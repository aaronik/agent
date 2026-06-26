use std::error::Error;
use std::time::{Duration, Instant};

use tokio::sync::mpsc;

use crate::agent::{AgentMessage, AssistantMessage, CancellationToken, ToolCall};
use crate::cli::EscAbortWatcher;
use crate::display::TerminalDisplay;
use crate::providers::{format_cost_and_context_line, parse_model_id};
use crate::session::{Session, SessionStore};
use crate::tools::ToolRegistry;
use crate::voice::audio::AudioIo;
use crate::voice::realtime::{
    RealtimeClient, RealtimeConfig, RealtimeError, RealtimeEvent, voice_instructions,
};

const DEFAULT_REALTIME_MODEL: &str = "gpt-realtime-2";
const PLAYBACK_ECHO_SUPPRESSION_HANGOVER: Duration = Duration::from_millis(900);
const DEFAULT_BARGE_IN_RMS_THRESHOLD: f64 = 900.0;
const DEFAULT_BARGE_IN_VOLUME_SCALE: f64 = 2_400.0;
const SYSTEM_VOLUME_CACHE_TTL: Duration = Duration::from_millis(500);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum InputAudioAction {
    Forward,
    SuppressEcho,
    BargeIn,
}

pub async fn run_talk_session(
    store: &SessionStore,
    session: &mut Session,
    model_name: &str,
    base_system_prompt: &str,
) -> Result<(), Box<dyn Error>> {
    let config = talk_config(model_name, base_system_prompt)?;
    println!(
        "\n[agent voice]\nmodel: {}\nvoice: {}\nspeed: {:.2}x",
        config.model, config.voice, config.voice_speed
    );
    println!(
        "Speak normally. Interrupt by talking over the assistant. Press Esc to cancel the current response. Press Ctrl-C to exit.\n"
    );

    let (audio_tx, mut audio_rx) = mpsc::unbounded_channel::<Vec<i16>>();
    let audio = AudioIo::start(audio_tx)?;
    audio.keepalive();
    let playback = audio.playback();
    let mut client = RealtimeClient::connect(&config).await?;
    let tools = ToolRegistry::new();
    let display = TerminalDisplay::new();
    let mut input_gate = InputAudioGate::default();
    let mut cancellation_token = CancellationToken::new();
    let mut esc_abort = EscAbortWatcher::spawn(cancellation_token.clone());

    loop {
        tokio::select! {
            Some(samples) = audio_rx.recv() => {
                match input_gate.action(&playback, &samples) {
                    InputAudioAction::SuppressEcho => continue,
                    InputAudioAction::BargeIn => {
                        audio.clear_playback();
                        let _ = client.cancel_response().await;
                    }
                    InputAudioAction::Forward => {}
                }
                client.append_input_audio(&samples).await?;
            }
            event = client.next_event() => {
                let Some(event) = event? else {
                    println!("voice session ended by server");
                    break;
                };
                let mut event_context = RealtimeEventContext {
                    client: &mut client,
                    tools: &tools,
                    display: &display,
                    audio: &audio,
                    playback: &playback,
                    model_name: config.model.as_str(),
                    store,
                    session,
                };
                handle_realtime_event(event, &mut event_context).await?;
            }
            _ = cancellation_token.cancelled() => {
                print_status("cancelled");
                audio.clear_playback();
                let _ = client.cancel_response().await;
                esc_abort.stop().await;
                cancellation_token = CancellationToken::new();
                esc_abort = EscAbortWatcher::spawn(cancellation_token.clone());
            }
            _ = tokio::signal::ctrl_c() => {
                println!("\nending voice session");
                audio.clear_playback();
                break;
            }
        }
    }

    esc_abort.stop().await;
    Ok(())
}

pub fn talk_model_name(model_name: &str) -> String {
    let parsed = parse_model_id(model_name);
    if parsed.provider == "mock" || parsed.provider == "ollama" {
        return DEFAULT_REALTIME_MODEL.to_string();
    }
    if let Ok(model) = std::env::var("AGENT_VOICE_MODEL")
        && !model.trim().is_empty()
    {
        return model;
    }
    if parsed.model.contains("realtime") {
        parsed.model
    } else {
        DEFAULT_REALTIME_MODEL.to_string()
    }
}

pub fn talk_config(
    model_name: &str,
    base_system_prompt: &str,
) -> Result<RealtimeConfig, Box<dyn Error>> {
    let parsed = parse_model_id(model_name);
    if parsed.provider != "openai" && !model_name.contains("realtime") {
        return Err("--talk currently supports OpenAI realtime models only".into());
    }
    let api_key = std::env::var("OPENAI_API_KEY")
        .map_err(|_| "--talk requires OPENAI_API_KEY for OpenAI Realtime")?;
    Ok(RealtimeConfig::new(
        talk_model_name(model_name),
        api_key,
        voice_instructions(base_system_prompt),
    )
    .with_tools(ToolRegistry::new().definitions().to_vec()))
}

struct RealtimeEventContext<'a> {
    client: &'a mut RealtimeClient,
    tools: &'a ToolRegistry,
    display: &'a TerminalDisplay,
    audio: &'a AudioIo,
    playback: &'a crate::voice::audio::PlaybackQueue,
    model_name: &'a str,
    store: &'a SessionStore,
    session: &'a mut Session,
}

async fn handle_realtime_event(
    event: RealtimeEvent,
    context: &mut RealtimeEventContext<'_>,
) -> Result<(), Box<dyn Error>> {
    match event {
        RealtimeEvent::AudioDelta(samples) => context.playback.push_pcm16(&samples),
        RealtimeEvent::SpeechStarted => {
            context.audio.clear_playback();
            print_status("interrupted");
        }
        RealtimeEvent::UserTranscript(transcript) => {
            if !transcript.trim().is_empty() {
                println!("\nYou: {}", transcript.trim());
                context.session.messages.push(AgentMessage::User {
                    content: transcript,
                });
                context.store.save(context.session)?;
            }
        }
        RealtimeEvent::AssistantTranscript(transcript) => {
            if !transcript.trim().is_empty() {
                println!("Agent: {}", transcript.trim());
                context
                    .session
                    .messages
                    .push(assistant_message(transcript, None));
                context.store.save(context.session)?;
            }
        }
        RealtimeEvent::Error(message) if is_benign_realtime_error(&message) => {
            eprintln!("[voice warning] {message}");
        }
        RealtimeEvent::Error(message) => return Err(RealtimeError::Connection(message).into()),
        RealtimeEvent::ResponseDone { tool_calls, usage } => {
            if tool_calls.is_empty() {
                attach_usage_to_latest_voice_assistant(context.session, usage);
                context.store.save(context.session)?;
            } else {
                handle_tool_calls(
                    tool_calls,
                    usage,
                    context.client,
                    context.tools,
                    context.display,
                    context.store,
                    context.session,
                )
                .await?;
            }
            print_cost_and_context(context.session, context.model_name);
        }
        RealtimeEvent::Other(_) => {}
    }
    Ok(())
}

fn print_status(status: &str) {
    println!("\n[{status}]");
}

fn assistant_message(content: String, usage: Option<crate::agent::Usage>) -> AgentMessage {
    AgentMessage::Assistant(AssistantMessage {
        content,
        tool_calls: Vec::new(),
        usage,
        metadata: serde_json::Map::new(),
    })
}

fn attach_usage_to_latest_voice_assistant(
    session: &mut Session,
    usage: Option<crate::agent::Usage>,
) {
    let Some(usage) = usage else {
        return;
    };

    if let Some(AgentMessage::Assistant(assistant)) = session
        .messages
        .iter_mut()
        .rev()
        .find(|message| matches!(message, AgentMessage::Assistant(_)))
        && assistant.usage.is_none()
    {
        assistant.usage = Some(usage);
        return;
    }

    session
        .messages
        .push(assistant_message(String::new(), Some(usage)));
}

fn print_cost_and_context(session: &Session, model_name: &str) {
    println!(
        "\n{}",
        format_cost_and_context_line(&session.messages, model_name)
    );
}

fn is_benign_realtime_error(message: &str) -> bool {
    message.contains("Cancellation failed: no active response found")
        || message.contains("no active response found")
}

async fn handle_tool_calls(
    tool_calls: Vec<ToolCall>,
    usage: Option<crate::agent::Usage>,
    client: &mut RealtimeClient,
    tools: &ToolRegistry,
    display: &TerminalDisplay,
    store: &SessionStore,
    session: &mut Session,
) -> Result<(), Box<dyn Error>> {
    if tool_calls.is_empty() {
        return Ok(());
    }

    let assistant_tool_calls = tool_calls.clone();
    session
        .messages
        .push(AgentMessage::Assistant(AssistantMessage {
            content: String::new(),
            tool_calls: assistant_tool_calls,
            usage,
            metadata: serde_json::Map::new(),
        }));

    for call in tool_calls {
        display.render_tool_start(&call);

        let result = tools
            .execute(call.id.clone(), &call.name, call.arguments.clone())
            .await;
        display.render_tool_result(&result);
        client
            .send_function_call_output(&result.tool_call_id, &result.content)
            .await?;
        session.messages.push(AgentMessage::Tool(result));
    }

    store.save(session)?;
    client.create_response().await?;
    Ok(())
}

#[derive(Debug, Default)]
struct InputAudioGate {
    system_volume: CachedSystemVolume,
}

impl InputAudioGate {
    fn action(
        &mut self,
        playback: &crate::voice::audio::PlaybackQueue,
        samples: &[i16],
    ) -> InputAudioAction {
        input_audio_action_with_volume(playback, samples, self.system_volume.current())
    }
}

#[derive(Debug, Default)]
struct CachedSystemVolume {
    value: Option<f64>,
    fetched_at: Option<Instant>,
}

impl CachedSystemVolume {
    fn current(&mut self) -> Option<f64> {
        if self
            .fetched_at
            .is_some_and(|fetched_at| fetched_at.elapsed() <= SYSTEM_VOLUME_CACHE_TTL)
        {
            return self.value;
        }

        self.value = system_output_volume();
        self.fetched_at = Some(Instant::now());
        self.value
    }
}

fn input_audio_action_with_volume(
    playback: &crate::voice::audio::PlaybackQueue,
    samples: &[i16],
    system_volume: Option<f64>,
) -> InputAudioAction {
    if !playback.is_active_within(PLAYBACK_ECHO_SUPPRESSION_HANGOVER) {
        return InputAudioAction::Forward;
    }
    if rms(samples) < volume_adjusted_barge_in_rms_threshold(system_volume) {
        InputAudioAction::SuppressEcho
    } else {
        InputAudioAction::BargeIn
    }
}

fn volume_adjusted_barge_in_rms_threshold(system_volume: Option<f64>) -> f64 {
    barge_in_rms_threshold()
        + system_volume.unwrap_or(0.0).clamp(0.0, 1.0) * barge_in_volume_scale()
}

fn barge_in_rms_threshold() -> f64 {
    std::env::var("AGENT_BARGE_IN_RMS_THRESHOLD")
        .ok()
        .and_then(|value| value.parse::<f64>().ok())
        .filter(|value| *value >= 0.0)
        .unwrap_or(DEFAULT_BARGE_IN_RMS_THRESHOLD)
}

fn barge_in_volume_scale() -> f64 {
    std::env::var("AGENT_BARGE_IN_VOLUME_SCALE")
        .ok()
        .and_then(|value| value.parse::<f64>().ok())
        .filter(|value| *value >= 0.0)
        .unwrap_or(DEFAULT_BARGE_IN_VOLUME_SCALE)
}

fn system_output_volume() -> Option<f64> {
    platform_system_output_volume()
}

#[cfg(target_os = "macos")]
fn platform_system_output_volume() -> Option<f64> {
    let output = std::process::Command::new("osascript")
        .args(["-e", "output volume of (get volume settings)"])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let raw = String::from_utf8(output.stdout).ok()?;
    let volume = raw.trim().parse::<f64>().ok()?;
    Some((volume / 100.0).clamp(0.0, 1.0))
}

#[cfg(not(target_os = "macos"))]
fn platform_system_output_volume() -> Option<f64> {
    None
}

fn rms(samples: &[i16]) -> f64 {
    if samples.is_empty() {
        return 0.0;
    }
    let sum_squares = samples
        .iter()
        .map(|sample| {
            let sample = f64::from(*sample);
            sample * sample
        })
        .sum::<f64>();
    (sum_squares / samples.len() as f64).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn talk_model_uses_voice_default_for_text_models() {
        assert_eq!(talk_model_name("gpt-5.5"), DEFAULT_REALTIME_MODEL);
        assert_eq!(talk_model_name("mock"), DEFAULT_REALTIME_MODEL);
    }

    #[test]
    fn talk_model_keeps_explicit_realtime_model() {
        assert_eq!(talk_model_name("openai:gpt-realtime-2"), "gpt-realtime-2");
    }

    #[test]
    fn response_usage_attaches_to_latest_voice_assistant() {
        let mut session = Session::new(
            "s1".to_string(),
            vec![AgentMessage::Assistant(AssistantMessage {
                content: "hello".to_string(),
                tool_calls: Vec::new(),
                usage: None,
                metadata: serde_json::Map::new(),
            })],
        );

        attach_usage_to_latest_voice_assistant(
            &mut session,
            Some(crate::agent::Usage {
                input_tokens: 10,
                output_tokens: 3,
                raw: Some(serde_json::json!({"total_cost": 0.0123})),
            }),
        );

        let AgentMessage::Assistant(assistant) = &session.messages[0] else {
            panic!("expected assistant message");
        };
        assert_eq!(assistant.usage.as_ref().expect("usage").input_tokens, 10);
    }

    #[test]
    fn response_usage_without_assistant_creates_cost_carrier_message() {
        let mut session = Session::new("s1".to_string(), Vec::new());

        attach_usage_to_latest_voice_assistant(
            &mut session,
            Some(crate::agent::Usage {
                input_tokens: 10,
                output_tokens: 3,
                raw: Some(serde_json::json!({"total_cost": 0.0123})),
            }),
        );

        let AgentMessage::Assistant(assistant) = &session.messages[0] else {
            panic!("expected assistant message");
        };
        assert_eq!(assistant.content, "");
        assert_eq!(assistant.usage.as_ref().expect("usage").output_tokens, 3);
    }

    #[test]
    fn input_echo_is_suppressed_while_playback_is_active() {
        let playback = crate::voice::audio::PlaybackQueue::default();
        playback.push_pcm16(&[1000; 240]);

        assert_eq!(
            input_audio_action_with_volume(&playback, &[200; 240], None),
            InputAudioAction::SuppressEcho
        );
        assert_eq!(
            input_audio_action_with_volume(&playback, &[8000; 240], None),
            InputAudioAction::BargeIn
        );
    }

    #[test]
    fn higher_system_volume_raises_barge_in_threshold() {
        assert!(
            volume_adjusted_barge_in_rms_threshold(Some(1.0))
                > volume_adjusted_barge_in_rms_threshold(Some(0.0))
        );
    }

    #[test]
    fn loud_output_volume_suppresses_louder_echo() {
        let playback = crate::voice::audio::PlaybackQueue::default();
        playback.push_pcm16(&[1000; 240]);

        assert_eq!(
            input_audio_action_with_volume(&playback, &[2_000; 240], Some(0.0)),
            InputAudioAction::BargeIn
        );
        assert_eq!(
            input_audio_action_with_volume(&playback, &[2_000; 240], Some(1.0)),
            InputAudioAction::SuppressEcho
        );
    }

    #[test]
    fn input_audio_is_not_suppressed_when_playback_is_inactive() {
        let playback = crate::voice::audio::PlaybackQueue::default();

        assert_eq!(
            input_audio_action_with_volume(&playback, &[200; 240], None),
            InputAudioAction::Forward
        );
    }

    #[test]
    fn no_active_response_cancel_error_is_benign() {
        assert!(is_benign_realtime_error(
            "Cancellation failed: no active response found"
        ));
        assert!(!is_benign_realtime_error("Invalid session payload"));
    }

    #[test]
    fn rms_measures_sample_energy() {
        assert_eq!(rms(&[]), 0.0);
        assert_eq!(rms(&[3, 4]), 3.5355339059327378);
    }
}
