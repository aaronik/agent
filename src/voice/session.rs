use std::error::Error;
use std::io::IsTerminal;
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};
use std::time::{Duration, Instant};

use crossterm::event::{self, Event, KeyCode, KeyEventKind, KeyModifiers};
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

const DEFAULT_REALTIME_MODEL: &str = "gpt-realtime";
const PLAYBACK_ECHO_SUPPRESSION_HANGOVER: Duration = Duration::from_millis(900);
const DEFAULT_BARGE_IN_RMS_THRESHOLD: f64 = 900.0;
const DEFAULT_BARGE_IN_VOLUME_SCALE: f64 = 2_400.0;
const SYSTEM_VOLUME_CACHE_TTL: Duration = Duration::from_millis(500);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TalkSessionExit {
    ToggleText,
    Ended,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum VoiceControlEvent {
    ToggleText,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum InputAudioAction {
    Forward,
    SuppressEcho,
    BargeIn,
}

enum TalkLoopAction {
    Continue,
    Reconnect(RealtimeError),
    ToggleText,
    Ended,
    Error(Box<dyn Error>),
}

pub async fn run_talk_session(
    store: &SessionStore,
    session: &mut Session,
    model_name: &str,
    base_system_prompt: &str,
) -> Result<TalkSessionExit, Box<dyn Error>> {
    let config =
        talk_config(model_name, base_system_prompt)?.with_history(session.messages.clone());
    println!(
        "\n[agent voice]\nmodel: {}\nvoice: {}\nspeed: {:.2}x",
        config.model, config.voice, config.voice_speed
    );
    println!(
        "Speak normally. Interrupt by talking over the assistant. Press Esc to cancel the current response. Press Ctrl-T to return to text mode. Press Ctrl-C to exit.\n"
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
    let (control_tx, mut control_rx) = mpsc::unbounded_channel();
    let control_watcher = VoiceControlWatcher::spawn(control_tx);

    loop {
        match talk_loop_tick(TalkLoopTick {
            client: &mut client,
            audio_rx: &mut audio_rx,
            control_rx: &mut control_rx,
            input_gate: &mut input_gate,
            audio: &audio,
            playback: &playback,
            tools: &tools,
            display: &display,
            model_name: config.model.as_str(),
            store,
            session,
            cancellation_token: &cancellation_token,
        })
        .await
        {
            TalkLoopAction::Continue => {}
            TalkLoopAction::Reconnect(err) => {
                eprintln!("[voice warning] {err}; reconnecting...");
                audio.clear_playback();
                match RealtimeClient::connect(&reconnect_config(&config, session)).await {
                    Ok(reconnected_client) => client = reconnected_client,
                    Err(err) => {
                        esc_abort.stop().await;
                        control_watcher.stop().await;
                        return Err(err.into());
                    }
                }
                print_status("reconnected");
            }
            TalkLoopAction::ToggleText => {
                esc_abort.stop().await;
                control_watcher.stop().await;
                return Ok(TalkSessionExit::ToggleText);
            }
            TalkLoopAction::Ended => break,
            TalkLoopAction::Error(err) => {
                esc_abort.stop().await;
                control_watcher.stop().await;
                return Err(err);
            }
        }

        if cancellation_token.is_cancelled() {
            print_status("cancelled");
            audio.clear_playback();
            let _ = client.cancel_response().await;
            esc_abort.stop().await;
            cancellation_token = CancellationToken::new();
            esc_abort = EscAbortWatcher::spawn(cancellation_token.clone());
        }
    }

    esc_abort.stop().await;
    control_watcher.stop().await;
    Ok(TalkSessionExit::Ended)
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

fn reconnect_config(config: &RealtimeConfig, session: &Session) -> RealtimeConfig {
    config.clone().with_history(session.messages.clone())
}

struct TalkLoopTick<'a> {
    client: &'a mut RealtimeClient,
    audio_rx: &'a mut mpsc::UnboundedReceiver<Vec<i16>>,
    control_rx: &'a mut mpsc::UnboundedReceiver<VoiceControlEvent>,
    input_gate: &'a mut InputAudioGate,
    audio: &'a AudioIo,
    playback: &'a crate::voice::audio::PlaybackQueue,
    tools: &'a ToolRegistry,
    display: &'a TerminalDisplay,
    model_name: &'a str,
    store: &'a SessionStore,
    session: &'a mut Session,
    cancellation_token: &'a CancellationToken,
}

async fn talk_loop_tick(context: TalkLoopTick<'_>) -> TalkLoopAction {
    tokio::select! {
        Some(samples) = context.audio_rx.recv() => {
            match context.input_gate.action(context.playback, &samples) {
                InputAudioAction::SuppressEcho => return TalkLoopAction::Continue,
                InputAudioAction::BargeIn => {
                    context.audio.clear_playback();
                    if let Err(err) = context.client.cancel_response().await
                        && is_recoverable_realtime_error(&err)
                    {
                        return TalkLoopAction::Reconnect(err);
                    }
                }
                InputAudioAction::Forward => {}
            }
            match context.client.append_input_audio(&samples).await {
                Ok(()) => TalkLoopAction::Continue,
                Err(err) if is_recoverable_realtime_error(&err) => TalkLoopAction::Reconnect(err),
                Err(err) => TalkLoopAction::Error(err.into()),
            }
        }
        event = context.client.next_event() => {
            let event = match event {
                Ok(Some(event)) => event,
                Ok(None) => {
                    println!("voice session ended by server");
                    return TalkLoopAction::Ended;
                }
                Err(err) if is_recoverable_realtime_error(&err) => {
                    return TalkLoopAction::Reconnect(err);
                }
                Err(err) => return TalkLoopAction::Error(err.into()),
            };
            let mut event_context = RealtimeEventContext {
                client: context.client,
                tools: context.tools,
                display: context.display,
                audio: context.audio,
                playback: context.playback,
                model_name: context.model_name,
                store: context.store,
                session: context.session,
                cancellation_token: context.cancellation_token,
            };
            match handle_realtime_event(event, &mut event_context).await {
                Ok(()) => TalkLoopAction::Continue,
                Err(err) if boxed_error_is_recoverable_realtime_connection(err.as_ref()) => {
                    TalkLoopAction::Reconnect(RealtimeError::Connection(err.to_string()))
                }
                Err(err) => TalkLoopAction::Error(err),
            }
        }
        Some(control) = context.control_rx.recv() => {
            match control {
                VoiceControlEvent::ToggleText => {
                    println!("\nreturning to text mode");
                    context.audio.clear_playback();
                    let _ = context.client.cancel_response().await;
                    TalkLoopAction::ToggleText
                }
            }
        }
        _ = context.cancellation_token.cancelled() => TalkLoopAction::Continue,
        _ = tokio::signal::ctrl_c() => {
            println!("\nending voice session");
            context.audio.clear_playback();
            TalkLoopAction::Ended
        }
    }
}

fn is_recoverable_realtime_error(err: &RealtimeError) -> bool {
    match err {
        RealtimeError::Connection(message) => is_recoverable_realtime_connection_message(message),
        RealtimeError::InvalidEvent(_) | RealtimeError::InvalidRequest(_) => false,
    }
}

fn boxed_error_is_recoverable_realtime_connection(err: &(dyn Error + 'static)) -> bool {
    err.downcast_ref::<RealtimeError>()
        .is_some_and(is_recoverable_realtime_error)
        || is_recoverable_realtime_connection_message(&err.to_string())
}

fn is_recoverable_realtime_connection_message(message: &str) -> bool {
    let message = message.to_ascii_lowercase();
    message.contains("connection reset")
        || message.contains("without closing handshake")
        || message.contains("broken pipe")
        || message.contains("connection closed")
        || message.contains("websocket protocol error")
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
    cancellation_token: &'a CancellationToken,
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
                handle_tool_calls(tool_calls, usage, context).await?;
            }
            print_cost_and_context(context.session, context.model_name);
        }
        RealtimeEvent::Other(_) => {}
    }
    Ok(())
}

struct VoiceControlWatcher {
    stop: Arc<AtomicBool>,
    handle: Option<tokio::task::JoinHandle<()>>,
}

impl VoiceControlWatcher {
    fn spawn(tx: mpsc::UnboundedSender<VoiceControlEvent>) -> Self {
        let stop = Arc::new(AtomicBool::new(false));
        let handle = std::io::stdin().is_terminal().then(|| {
            let stop_watcher = Arc::clone(&stop);
            tokio::task::spawn_blocking(move || {
                while !stop_watcher.load(Ordering::SeqCst) {
                    match event::poll(Duration::from_millis(50)) {
                        Ok(true) => match event::read() {
                            Ok(Event::Key(key))
                                if is_toggle_text_key(key.code, key.modifiers, key.kind) =>
                            {
                                let _ = tx.send(VoiceControlEvent::ToggleText);
                                break;
                            }
                            Ok(_) => {}
                            Err(_) => break,
                        },
                        Ok(false) => {}
                        Err(_) => break,
                    }
                }
            })
        });
        Self { stop, handle }
    }

    async fn stop(self) {
        self.stop.store(true, Ordering::SeqCst);
        if let Some(handle) = self.handle {
            let _ = handle.await;
        }
    }
}

fn is_toggle_text_key(code: KeyCode, modifiers: KeyModifiers, kind: KeyEventKind) -> bool {
    matches!(code, KeyCode::Char('t' | 'T'))
        && modifiers.contains(KeyModifiers::CONTROL)
        && kind == KeyEventKind::Press
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
    context: &mut RealtimeEventContext<'_>,
) -> Result<(), Box<dyn Error>> {
    if tool_calls.is_empty() {
        return Ok(());
    }

    let assistant_tool_calls = tool_calls.clone();
    context
        .session
        .messages
        .push(AgentMessage::Assistant(AssistantMessage {
            content: String::new(),
            tool_calls: assistant_tool_calls,
            usage,
            metadata: serde_json::Map::new(),
        }));

    for call in tool_calls {
        context.display.render_tool_start(&call);

        let result = context
            .tools
            .execute_cancellable(
                call.id.clone(),
                &call.name,
                call.arguments.clone(),
                context.cancellation_token,
            )
            .await;
        context.display.render_tool_result(&result);
        if context.cancellation_token.is_cancelled() {
            return Ok(());
        }
        context
            .client
            .send_function_call_output(&result.tool_call_id, &result.content)
            .await?;
        context.session.messages.push(AgentMessage::Tool(result));
    }

    context.store.save(context.session)?;
    context.client.create_response().await?;
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
        assert_eq!(talk_model_name("openai:gpt-realtime"), "gpt-realtime");
    }

    #[test]
    fn ctrl_t_is_voice_toggle_key() {
        assert!(is_toggle_text_key(
            KeyCode::Char('t'),
            KeyModifiers::CONTROL,
            KeyEventKind::Press,
        ));
        assert!(is_toggle_text_key(
            KeyCode::Char('T'),
            KeyModifiers::CONTROL | KeyModifiers::SHIFT,
            KeyEventKind::Press,
        ));
        assert!(!is_toggle_text_key(
            KeyCode::Char('t'),
            KeyModifiers::NONE,
            KeyEventKind::Press,
        ));
        assert!(!is_toggle_text_key(
            KeyCode::Char('t'),
            KeyModifiers::CONTROL,
            KeyEventKind::Release,
        ));
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
    fn websocket_reset_error_is_recoverable() {
        let err = RealtimeError::Connection(
            "WebSocket protocol error: Connection reset without closing handshake".to_string(),
        );

        assert!(is_recoverable_realtime_error(&err));
        assert!(boxed_error_is_recoverable_realtime_connection(&err));
        assert!(!is_recoverable_realtime_error(&RealtimeError::Connection(
            "Invalid session payload".to_string(),
        )));
    }

    #[test]
    fn reconnect_config_uses_latest_session_history() {
        let original = RealtimeConfig::new(
            "gpt-realtime".to_string(),
            "sk-test".to_string(),
            "instructions".to_string(),
        )
        .with_history(vec![AgentMessage::User {
            content: "old".to_string(),
        }]);
        let session = Session::new(
            "s1".to_string(),
            vec![AgentMessage::User {
                content: "latest".to_string(),
            }],
        );

        let reconnect = reconnect_config(&original, &session);

        assert_eq!(reconnect.model, original.model);
        assert_eq!(reconnect.history, session.messages);
    }

    #[test]
    fn rms_measures_sample_energy() {
        assert_eq!(rms(&[]), 0.0);
        assert_eq!(rms(&[3, 4]), 3.5355339059327378);
    }
}
