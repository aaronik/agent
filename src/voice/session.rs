use std::error::Error;
use std::time::Duration;

use tokio::sync::mpsc;

use crate::agent::{AgentMessage, AssistantMessage, ToolCall};
use crate::display::TerminalDisplay;
use crate::providers::parse_model_id;
use crate::session::{Session, SessionStore};
use crate::tools::ToolRegistry;
use crate::voice::audio::AudioIo;
use crate::voice::realtime::{
    RealtimeClient, RealtimeConfig, RealtimeError, RealtimeEvent, voice_instructions,
};

const DEFAULT_REALTIME_MODEL: &str = "gpt-realtime-2";
const PLAYBACK_ECHO_SUPPRESSION_HANGOVER: Duration = Duration::from_millis(900);
const DEFAULT_BARGE_IN_RMS_THRESHOLD: f64 = 1_200.0;

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
    println!("Speak normally. Interrupt by talking over the assistant. Press Ctrl-C to exit.\n");

    let (audio_tx, mut audio_rx) = mpsc::unbounded_channel::<Vec<i16>>();
    let audio = AudioIo::start(audio_tx)?;
    audio.keepalive();
    let playback = audio.playback();
    let mut client = RealtimeClient::connect(&config).await?;
    let tools = ToolRegistry::new();
    let display = TerminalDisplay::new();

    loop {
        tokio::select! {
            Some(samples) = audio_rx.recv() => {
                match input_audio_action(&playback, &samples) {
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
                    store,
                    session,
                };
                handle_realtime_event(event, &mut event_context).await?;
            }
            _ = tokio::signal::ctrl_c() => {
                println!("\nending voice session");
                audio.clear_playback();
                break;
            }
        }
    }

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
                    .push(AgentMessage::Assistant(AssistantMessage {
                        content: transcript,
                        tool_calls: Vec::new(),
                        usage: None,
                        metadata: serde_json::Map::new(),
                    }));
                context.store.save(context.session)?;
            }
        }
        RealtimeEvent::Error(message) if is_benign_realtime_error(&message) => {
            eprintln!("[voice warning] {message}");
        }
        RealtimeEvent::Error(message) => return Err(RealtimeError::Connection(message).into()),
        RealtimeEvent::ResponseDone { tool_calls } => {
            handle_tool_calls(
                tool_calls,
                context.client,
                context.tools,
                context.display,
                context.store,
                context.session,
            )
            .await?;
        }
        RealtimeEvent::Other(_) => {}
    }
    Ok(())
}

fn print_status(status: &str) {
    println!("\n[{status}]");
}

fn is_benign_realtime_error(message: &str) -> bool {
    message.contains("Cancellation failed: no active response found")
        || message.contains("no active response found")
}

async fn handle_tool_calls(
    tool_calls: Vec<ToolCall>,
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
            usage: None,
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

fn input_audio_action(
    playback: &crate::voice::audio::PlaybackQueue,
    samples: &[i16],
) -> InputAudioAction {
    if !playback.is_active_within(PLAYBACK_ECHO_SUPPRESSION_HANGOVER) {
        return InputAudioAction::Forward;
    }
    if rms(samples) < barge_in_rms_threshold() {
        InputAudioAction::SuppressEcho
    } else {
        InputAudioAction::BargeIn
    }
}

fn barge_in_rms_threshold() -> f64 {
    std::env::var("AGENT_BARGE_IN_RMS_THRESHOLD")
        .ok()
        .and_then(|value| value.parse::<f64>().ok())
        .filter(|value| *value >= 0.0)
        .unwrap_or(DEFAULT_BARGE_IN_RMS_THRESHOLD)
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
    fn input_echo_is_suppressed_while_playback_is_active() {
        let playback = crate::voice::audio::PlaybackQueue::default();
        playback.push_pcm16(&[1000; 240]);

        assert_eq!(
            input_audio_action(&playback, &[200; 240]),
            InputAudioAction::SuppressEcho
        );
        assert_eq!(
            input_audio_action(&playback, &[8000; 240]),
            InputAudioAction::BargeIn
        );
    }

    #[test]
    fn input_audio_is_not_suppressed_when_playback_is_inactive() {
        let playback = crate::voice::audio::PlaybackQueue::default();

        assert_eq!(
            input_audio_action(&playback, &[200; 240]),
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
