use base64::Engine;
use clap::Parser;
use crossterm::event::{
    self, Event, KeyCode as CrosstermKeyCode, KeyEventKind, KeyModifiers as CrosstermKeyModifiers,
};
#[cfg(not(unix))]
use crossterm::terminal::{disable_raw_mode, enable_raw_mode};
use reedline::{
    Completer, CompletionResult, DefaultPrompt, EditCommand, EditMode, FileBackedHistory, History,
    KeyCode, KeyModifiers, Keybindings, ListMenu, MenuBuilder, PromptEditMode, PromptViMode,
    Reedline, ReedlineEvent, ReedlineMenu, ReedlineRawEvent, SearchDirection, SearchQuery, Signal,
    Span, Suggestion, Vi, default_vi_insert_keybindings, default_vi_normal_keybindings,
};
use std::cell::RefCell;
use std::error::Error;
use std::io::{IsTerminal, Read};
use std::path::{Path, PathBuf};
use std::sync::{
    Arc, RwLock,
    atomic::{AtomicBool, Ordering},
};
use std::time::Duration;

use crate::agent::{AgentLoop, AgentLoopConfig, AgentMessage, CancellationToken};
use crate::display::TerminalDisplay;
use crate::memory::load_all_agents_memory;
use crate::pricing::refresh_pricing_cache;
use crate::providers::{
    Provider, build_provider, effective_model_name, format_cost_and_context_line, list_models,
};
use crate::session::{Session, SessionStore};
use crate::tools::ToolRegistry;

const COMPLETION_MENU_NAME: &str = "completion_menu";
const TOGGLE_TALK_HOST_COMMAND: &str = "agent:toggle-talk";
const SPINNER_UPDATE_INTERVAL: Duration = Duration::from_millis(128);

#[derive(Debug, Clone, PartialEq, Eq)]
enum PromptInput {
    Text(String),
    ToggleTalk,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum InteractionMode {
    Text,
    Talk,
}

#[derive(Debug, Parser)]
#[command(name = "agent", about = "Personal Command Line Agent")]
pub struct Args {
    #[arg(short, long, help = "Model id, optionally provider-prefixed")]
    pub model: Option<String>,
    #[arg(short, long, help = "List available OpenAI and Ollama models")]
    pub list_models: bool,
    #[arg(
        short,
        long,
        help = "Download and cache LiteLLM pricing data, then exit"
    )]
    pub update_pricing: bool,
    #[arg(short, long, help = "Run one turn and exit")]
    pub single: bool,
    #[arg(short = 't', long = "talk", help = "Start a realtime voice session")]
    pub talk: bool,
    #[arg(
        short = 'c',
        long = "command",
        help = "Run one turn and print only the final response for zsh print -z wrappers"
    )]
    pub command: bool,
    #[arg(
        short,
        long,
        num_args = 0..=1,
        default_missing_value = "__LATEST__",
        help = "Resume latest session or the provided session id"
    )]
    pub resume: Option<String>,
    #[arg(short = 'n', long, help = "Start a new session instead of resuming")]
    pub new: bool,
    #[arg(
        short = 'i',
        long = "image",
        value_name = "PATH",
        action = clap::ArgAction::Append,
        help = "Attach an image to the initial text message (repeatable)"
    )]
    pub images: Vec<std::path::PathBuf>,
    #[arg(long, help = "Allow git commands that modify repositories")]
    pub allow_git: bool,
    #[arg(long, help = "Disable the sound played after a successful turn")]
    pub no_completion_sound: bool,
    #[arg(long, help = "Disable spawning subagents for this run")]
    pub no_subagent: bool,
    #[arg(help = "Initial user message")]
    pub query: Vec<String>,
}

pub async fn run() -> Result<(), Box<dyn Error>> {
    let args = Args::parse();
    let prompt_prefill = capture_piped_prompt_prefill(&args)?;
    run_with_args_and_prefill(args, prompt_prefill).await
}

fn validate_args(args: &Args) -> Result<(), Box<dyn Error>> {
    if args.new && args.resume.is_some() {
        return Err("--new cannot be used with --resume".into());
    }
    if args.talk && args.command {
        return Err("--talk cannot be combined with --command".into());
    }
    if args.talk && args.single && args.query.is_empty() {
        return Err("--talk --single requires an initial query".into());
    }
    if !args.images.is_empty() && args.query.is_empty() && !args.talk {
        return Err("--image requires an initial query outside talk mode".into());
    }
    Ok(())
}

pub async fn run_with_args(args: Args) -> Result<(), Box<dyn Error>> {
    run_with_args_and_prefill(args, None).await
}

async fn run_with_args_and_prefill(
    args: Args,
    mut prompt_prefill: Option<String>,
) -> Result<(), Box<dyn Error>> {
    validate_args(&args)?;

    if args.update_pricing {
        let store = SessionStore::new()?;
        println!("{}", refresh_pricing_cache(store.root()).await?);
        return Ok(());
    }

    if args.list_models {
        for model in list_models().await {
            println!("{model}");
        }
        return Ok(());
    }

    if args.command {
        return run_command_mode(&args).await;
    }

    let store = SessionStore::new()?;
    let display = Arc::new(TerminalDisplay::new());
    let mut model_name = effective_model_name(args.model.as_deref());
    let mut loop_runner: Option<AgentLoop<Box<dyn Provider>>> = None;
    let mut allow_git_writes = args.allow_git;

    let mut session = load_or_create_session(&args, &store)?;
    print_agent_header(&model_name);
    replay_session(&session, &display);

    if args.talk && (!args.images.is_empty() || !args.query.is_empty()) {
        let parsed_input = parse_user_input(&args.query.join(" "), &args.images)?;
        let user_message = if parsed_input.images.is_empty() {
            AgentMessage::User {
                content: parsed_input.content,
            }
        } else {
            AgentMessage::UserWithImages {
                content: parsed_input.content,
                images: parsed_input.images,
            }
        };
        display.render_new_message(&user_message);
        session.messages.push(user_message);
        store.save(&session)?;
    }

    let mut interaction_mode = if args.talk {
        InteractionMode::Talk
    } else {
        InteractionMode::Text
    };

    let mut first_input = if args.talk || args.query.is_empty() {
        None
    } else {
        Some(args.query.join(" "))
    };
    let mut pending_images = Some(args.images.as_slice());

    loop {
        if interaction_mode == InteractionMode::Talk && first_input.is_none() {
            match crate::voice::session::run_talk_session(
                &store,
                &mut session,
                &model_name,
                &system_prompt(),
                args.single,
                allow_git_writes,
            )
            .await?
            {
                crate::voice::session::TalkSessionExit::ToggleText => {
                    interaction_mode = InteractionMode::Text;
                    loop_runner = None;
                    print_agent_header(&model_name);
                    continue;
                }
                crate::voice::session::TalkSessionExit::Ended => break,
            }
        }

        let (user_input, rendered_by_reedline) = match first_input.take() {
            Some(input) => (input, false),
            None => {
                if args.single && !session.messages.is_empty() {
                    break;
                }
                match prompt_for_input(
                    &store,
                    &session,
                    &model_name,
                    allow_git_writes,
                    prompt_prefill.take().as_deref(),
                )
                .await
                {
                    Ok(PromptInput::Text(input)) => (input, true),
                    Ok(PromptInput::ToggleTalk) => {
                        interaction_mode = InteractionMode::Talk;
                        loop_runner = None;
                        continue;
                    }
                    Err(err) if err.to_string() == "Goodbye!" => break,
                    Err(err) => return Err(err),
                }
            }
        };

        let image_paths = pending_images.take().unwrap_or_default();
        let mut parsed_input = parse_user_input(&user_input, image_paths)?;
        if parsed_input.images.is_empty() {
            match handle_slash_command(
                &user_input,
                &args,
                &store,
                &mut session,
                &display,
                &model_name,
                &mut allow_git_writes,
            )
            .await?
            {
                SlashCommandResult::NotCommand => {}
                SlashCommandResult::InvokeSkill(content) => {
                    parsed_input.content = content;
                }
                SlashCommandResult::Handled => {
                    loop_runner = None;
                    if args.single {
                        break;
                    }
                    continue;
                }
                SlashCommandResult::SwitchModel(new_model) => {
                    model_name = new_model;
                    loop_runner = None;
                    print_agent_header(&model_name);
                    if args.single {
                        break;
                    }
                    continue;
                }
            }
        }

        let user_message = if parsed_input.images.is_empty() {
            AgentMessage::User {
                content: parsed_input.content,
            }
        } else {
            AgentMessage::UserWithImages {
                content: parsed_input.content,
                images: parsed_input.images,
            }
        };
        if !rendered_by_reedline {
            display.render_new_message(&user_message);
        }
        session.messages.push(user_message);
        store.save(&session)?;

        if loop_runner.is_none() {
            loop_runner = Some(build_loop_runner(
                &model_name,
                allow_git_writes,
                args.no_subagent,
            )?);
        }
        let status_line =
            format_cost_and_context_line(&session.messages, &model_name, allow_git_writes);
        display.render_turn_submitted(&status_line);
        let cancellation_token = CancellationToken::new();
        let esc_abort = if args.single {
            EscAbortWatcher::disabled()
        } else {
            EscAbortWatcher::spawn_with_display(
                cancellation_token.clone(),
                Some(Arc::clone(&display)),
                load_prompt_history(&store).unwrap_or_default(),
            )
        };
        let observed_messages = RefCell::new(Vec::new());
        let streamed_assistant = Arc::new(std::sync::Mutex::new(AssistantStreamBuffer::default()));
        let persistence_error = RefCell::new(None);
        let streamed_assistant_for_events = Arc::clone(&streamed_assistant);
        let result = tokio::select! {
            result = loop_runner
                .as_ref()
                .expect("loop runner initialized")
                .run_turn_cancellable_with_event_observer(
                    &session.messages,
                    &cancellation_token,
                    |event| {
                        if let crate::agent::ProviderEvent::TextDelta { text } = event
                            && let Ok(mut assistant) = streamed_assistant_for_events.lock()
                            && let Some(text) = assistant.push(text)
                        {
                            display.render_assistant_delta(&text);
                        }
                    },
                    |message| {
                        let streamed_content = streamed_assistant
                            .lock()
                            .map(|mut assistant| {
                                let pending = assistant.finish();
                                let rendered = assistant.rendered().to_string();
                                (rendered, pending)
                            })
                            .unwrap_or_default();
                        if let AgentMessage::Assistant(assistant) = message {
                            if !streamed_content.1.is_empty() {
                                display.render_assistant_delta(&streamed_content.1);
                            }
                            let remainder = TerminalDisplay::assistant_stream_remainder(
                                &streamed_content.0,
                                &assistant.content,
                            );
                            if !streamed_content.0.is_empty() {
                                if !remainder.is_empty() {
                                    display.render_assistant_delta(&remainder);
                                }
                            } else {
                                display.render_new_message(message);
                            }
                            display.finish_assistant_stream();
                            if let Ok(mut assistant) = streamed_assistant.lock() {
                                assistant.reset();
                            }
                        } else {
                            if let AgentMessage::Tool(result) = message {
                                display.render_tool_result(result);
                            }
                            display.render_new_message(message);
                        }
                        observed_messages.borrow_mut().push(message.clone());
                        let mut status_messages = session.messages.clone();
                        status_messages.extend(observed_messages.borrow().iter().cloned());
                        display.update_working_footer(&format_cost_and_context_line(
                            &status_messages,
                            &model_name,
                            allow_git_writes,
                        ));
                        let can_save = persistence_error.borrow().is_none();
                        if can_save {
                            let mut checkpoint = session.clone();
                            checkpoint.messages.extend(observed_messages.borrow().iter().cloned());
                            checkpoint.replace_messages(checkpoint.messages.clone());
                            if let Err(error) = store.save(&checkpoint) {
                                *persistence_error.borrow_mut() = Some(error);
                            }
                        }
                    },
                    |call| display.render_tool_start(call),
                ) => result,
            _ = cancellation_token.cancelled() => Err(crate::providers::ProviderError::Cancelled),
            _ = tokio::signal::ctrl_c() => {
                cancellation_token.cancel();
                let _ = esc_abort.stop().await;
                display.finish_turn();
                return Err("cancelled".into());
            }
        };
        let typed_ahead = esc_abort.stop().await;
        if !typed_ahead.is_empty() {
            prompt_prefill = Some(typed_ahead);
        }
        display.finish_turn();

        if let Some(error) = persistence_error.into_inner() {
            return Err(error.into());
        }
        let observed_messages = observed_messages.into_inner();
        session.messages.extend(observed_messages.iter().cloned());
        if !observed_messages.is_empty() {
            session.replace_messages(session.messages.clone());
            store.save(&session)?;
        }

        match result {
            Ok(result) => {
                debug_assert_eq!(result.new_messages, observed_messages);
                if !args.no_completion_sound {
                    play_turn_completed_sound();
                }
            }
            Err(crate::providers::ProviderError::Cancelled)
                if cancellation_token.is_cancelled() =>
            {
                eprintln!("turn aborted");
            }
            Err(crate::providers::ProviderError::ContextLengthExceeded(error)) => {
                eprintln!("{}", context_limit_notice(&error));
            }
            Err(error @ crate::providers::ProviderError::Request(_)) if !args.single => {
                eprintln!("{error}");
            }
            Err(err) => return Err(err.into()),
        }

        if args.single {
            break;
        }
    }

    println!("sessionId: {}", session.session_id);
    Ok(())
}

#[derive(Debug, Default)]
struct AssistantStreamBuffer {
    rendered: String,
    pending_repeat: String,
}

impl AssistantStreamBuffer {
    fn push(&mut self, delta: &str) -> Option<String> {
        if self.rendered.is_empty() {
            self.rendered.push_str(delta);
            return Some(delta.to_string());
        }

        self.pending_repeat.push_str(delta);
        if self.rendered.starts_with(&self.pending_repeat) {
            if self.pending_repeat.len() == self.rendered.len() {
                self.pending_repeat.clear();
            }
            return None;
        }

        let output = if let Some(suffix) = self.pending_repeat.strip_prefix(&self.rendered) {
            suffix.to_string()
        } else {
            self.pending_repeat.clone()
        };
        self.rendered.push_str(&output);
        self.pending_repeat.clear();
        (!output.is_empty()).then_some(output)
    }

    fn finish(&mut self) -> String {
        if self.pending_repeat.is_empty() || self.pending_repeat == self.rendered {
            self.pending_repeat.clear();
            return String::new();
        }
        let output = std::mem::take(&mut self.pending_repeat);
        self.rendered.push_str(&output);
        output
    }

    fn rendered(&self) -> &str {
        &self.rendered
    }

    fn reset(&mut self) {
        self.rendered.clear();
        self.pending_repeat.clear();
    }
}

struct ParsedUserInput {
    content: String,
    images: Vec<crate::agent::ImageAttachment>,
}

fn parse_user_input(
    input: &str,
    explicit_images: &[std::path::PathBuf],
) -> Result<ParsedUserInput, Box<dyn Error>> {
    let mut image_paths = explicit_images.to_vec();
    let mut text_parts = Vec::new();
    let mut found_dragged_image = false;
    for part in shell_words(input) {
        let candidate = part.trim_end_matches(['.', ',', ';', ':']);
        let path = std::path::PathBuf::from(candidate);
        if is_supported_image_path(&path) && path.is_file() {
            image_paths.push(path);
            found_dragged_image = true;
        } else {
            text_parts.push(part);
        }
    }
    let content = if found_dragged_image {
        text_parts.join(" ")
    } else {
        input.to_string()
    };
    Ok(ParsedUserInput {
        content: if content.trim().is_empty() && !image_paths.is_empty() {
            "Describe this image.".to_string()
        } else {
            content
        },
        images: load_image_attachments(&image_paths)?,
    })
}

fn shell_words(input: &str) -> Vec<String> {
    let mut words = Vec::new();
    let mut word = String::new();
    let mut quote = None;
    let mut escaped = false;
    for character in input.chars() {
        if escaped {
            word.push(character);
            escaped = false;
        } else if character == '\\' && quote != Some('\'') {
            escaped = true;
        } else if matches!(character, '\'' | '"') {
            if quote == Some(character) {
                quote = None;
            } else if quote.is_none() {
                quote = Some(character);
            } else {
                word.push(character);
            }
        } else if character.is_ascii_whitespace() && quote.is_none() {
            if !word.is_empty() {
                words.push(std::mem::take(&mut word));
            }
        } else {
            word.push(character);
        }
    }
    if escaped {
        word.push('\\');
    }
    if !word.is_empty() {
        words.push(word);
    }
    words
}

fn is_supported_image_path(path: &std::path::Path) -> bool {
    matches!(
        path.extension()
            .and_then(|extension| extension.to_str())
            .map(str::to_ascii_lowercase)
            .as_deref(),
        Some("png" | "jpg" | "jpeg" | "gif" | "webp")
    )
}

fn load_image_attachments(
    paths: &[std::path::PathBuf],
) -> Result<Vec<crate::agent::ImageAttachment>, Box<dyn Error>> {
    paths
        .iter()
        .map(|path| {
            let media_type = match path
                .extension()
                .and_then(|extension| extension.to_str())
                .map(str::to_ascii_lowercase)
                .as_deref()
            {
                Some("png") => "image/png",
                Some("jpg" | "jpeg") => "image/jpeg",
                Some("gif") => "image/gif",
                Some("webp") => "image/webp",
                _ => return Err(format!("unsupported image format: {}", path.display()).into()),
            };
            let bytes = std::fs::read(path)
                .map_err(|err| format!("could not read image {}: {err}", path.display()))?;
            Ok(crate::agent::ImageAttachment {
                media_type: media_type.to_string(),
                data: base64::engine::general_purpose::STANDARD.encode(bytes),
            })
        })
        .collect()
}

fn context_limit_notice(_provider_error: &str) -> &'static str {
    "The conversation is too large for the model's context window. Run /compact, then try again."
}

fn play_turn_completed_sound() {
    #[cfg(target_os = "macos")]
    {
        const COMPLETION_SOUND: &str = "/System/Library/Sounds/Ping.aiff";
        let _ = std::process::Command::new("afplay")
            .arg(COMPLETION_SOUND)
            .spawn();
    }
}

async fn run_command_mode(args: &Args) -> Result<(), Box<dyn Error>> {
    if args.query.is_empty() {
        return Err("command mode requires a query".into());
    }

    let store = SessionStore::new()?;
    let mut session = load_or_create_session(
        &Args {
            model: args.model.clone(),
            list_models: false,
            update_pricing: false,
            single: true,
            talk: false,
            command: false,
            resume: None,
            new: true,
            images: Vec::new(),
            allow_git: args.allow_git,
            no_completion_sound: args.no_completion_sound,
            no_subagent: false,
            query: Vec::new(),
        },
        &store,
    )?;
    session.messages.push(AgentMessage::System {
        content: command_mode_system_prompt(),
    });
    let command_content = args.query.join(" ");
    session.messages.push(if args.images.is_empty() {
        AgentMessage::User {
            content: command_content,
        }
    } else {
        AgentMessage::UserWithImages {
            content: command_content,
            images: load_image_attachments(&args.images)?,
        }
    });

    let model_name = effective_model_name(args.model.as_deref());
    let assistant = build_provider(&model_name)?
        .complete(&session.messages, &[])
        .await?;

    println!("{}", assistant.content.trim());
    Ok(())
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
enum WorkingVimMode {
    #[default]
    Insert,
    Normal,
}

impl WorkingVimMode {
    fn label(self) -> &'static str {
        match self {
            Self::Insert => "INSERT",
            Self::Normal => "NORMAL",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum WorkingInputAction {
    Redraw,
    Abort,
    Ignored,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum FindDirection {
    Forward,
    Backward,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct FindMotion {
    direction: FindDirection,
    till: bool,
    character: char,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct PendingFind {
    direction: FindDirection,
    till: bool,
}

#[derive(Debug, Default)]
struct WorkingVimEditor {
    characters: Vec<char>,
    cursor: usize,
    mode: WorkingVimMode,
    pending_find: Option<PendingFind>,
    last_find: Option<FindMotion>,
    history: Vec<String>,
    history_index: Option<usize>,
    history_draft: Option<Vec<char>>,
}

impl WorkingVimEditor {
    fn with_history(history: Vec<String>) -> Self {
        Self {
            history,
            ..Self::default()
        }
    }

    fn text(&self) -> String {
        self.characters.iter().collect()
    }

    fn apply(&mut self, event: Event) -> WorkingInputAction {
        if matches!(event, Event::Resize(..)) {
            return WorkingInputAction::Redraw;
        }
        if let Event::Paste(text) = event {
            if self.mode == WorkingVimMode::Insert {
                for character in text.chars() {
                    self.characters.insert(self.cursor, character);
                    self.cursor += 1;
                }
                return WorkingInputAction::Redraw;
            }
            return WorkingInputAction::Ignored;
        }
        let Event::Key(key) = event else {
            return WorkingInputAction::Ignored;
        };
        if !matches!(key.kind, KeyEventKind::Press | KeyEventKind::Repeat) {
            return WorkingInputAction::Ignored;
        }
        if key.modifiers == CrosstermKeyModifiers::CONTROL
            && self.mode == WorkingVimMode::Insert
            && matches!(
                key.code,
                CrosstermKeyCode::Char('w') | CrosstermKeyCode::Char('W')
            )
        {
            self.delete_previous_word();
            return WorkingInputAction::Redraw;
        }
        if key.modifiers != CrosstermKeyModifiers::NONE
            && key.modifiers != CrosstermKeyModifiers::SHIFT
        {
            return WorkingInputAction::Ignored;
        }
        match self.mode {
            WorkingVimMode::Insert => self.apply_insert(key.code),
            WorkingVimMode::Normal => self.apply_normal(key.code),
        }
    }

    fn apply_insert(&mut self, code: CrosstermKeyCode) -> WorkingInputAction {
        match code {
            CrosstermKeyCode::Esc => self.mode = WorkingVimMode::Normal,
            CrosstermKeyCode::Up => self.history_previous(),
            CrosstermKeyCode::Down => self.history_next(),
            CrosstermKeyCode::Char(character) => {
                self.characters.insert(self.cursor, character);
                self.cursor += 1;
            }
            CrosstermKeyCode::Backspace if self.cursor > 0 => {
                self.cursor -= 1;
                self.characters.remove(self.cursor);
            }
            CrosstermKeyCode::Left if self.cursor > 0 => self.cursor -= 1,
            CrosstermKeyCode::Right if self.cursor < self.characters.len() => self.cursor += 1,
            CrosstermKeyCode::Home => self.cursor = 0,
            CrosstermKeyCode::End => self.cursor = self.characters.len(),
            CrosstermKeyCode::Tab => {
                self.characters.insert(self.cursor, '\t');
                self.cursor += 1;
            }
            _ => return WorkingInputAction::Ignored,
        }
        WorkingInputAction::Redraw
    }

    fn apply_normal(&mut self, code: CrosstermKeyCode) -> WorkingInputAction {
        if let Some(pending) = self.pending_find.take() {
            return match code {
                CrosstermKeyCode::Esc => WorkingInputAction::Redraw,
                CrosstermKeyCode::Char(character) => {
                    let motion = FindMotion {
                        direction: pending.direction,
                        till: pending.till,
                        character,
                    };
                    self.apply_find(motion);
                    self.last_find = Some(motion);
                    WorkingInputAction::Redraw
                }
                _ => WorkingInputAction::Ignored,
            };
        }
        match code {
            CrosstermKeyCode::Esc => return WorkingInputAction::Abort,
            CrosstermKeyCode::Up => self.history_previous(),
            CrosstermKeyCode::Down => self.history_next(),
            CrosstermKeyCode::Char('h') | CrosstermKeyCode::Left if self.cursor > 0 => {
                self.cursor -= 1
            }
            CrosstermKeyCode::Char('l') | CrosstermKeyCode::Right
                if self.cursor < self.characters.len() =>
            {
                self.cursor += 1
            }
            CrosstermKeyCode::Char('0') | CrosstermKeyCode::Home => self.cursor = 0,
            CrosstermKeyCode::Char('$') | CrosstermKeyCode::End => {
                self.cursor = self.characters.len()
            }
            CrosstermKeyCode::Char('w') => self.move_word_forward(false),
            CrosstermKeyCode::Char('W') => self.move_word_forward(true),
            CrosstermKeyCode::Char('b') => self.move_word_backward(false),
            CrosstermKeyCode::Char('B') => self.move_word_backward(true),
            CrosstermKeyCode::Char('e') => self.move_word_end(false),
            CrosstermKeyCode::Char('E') => self.move_word_end(true),
            CrosstermKeyCode::Char('f') => self.begin_find(FindDirection::Forward, false),
            CrosstermKeyCode::Char('F') => self.begin_find(FindDirection::Backward, false),
            CrosstermKeyCode::Char('t') => self.begin_find(FindDirection::Forward, true),
            CrosstermKeyCode::Char('T') => self.begin_find(FindDirection::Backward, true),
            CrosstermKeyCode::Char(';') => {
                if let Some(motion) = self.last_find {
                    self.apply_find(motion);
                }
            }
            CrosstermKeyCode::Char(',') => {
                if let Some(mut motion) = self.last_find {
                    motion.direction = reverse_find_direction(motion.direction);
                    self.apply_find(motion);
                }
            }
            CrosstermKeyCode::Char('i') => self.mode = WorkingVimMode::Insert,
            CrosstermKeyCode::Char('a') => {
                self.cursor = (self.cursor + 1).min(self.characters.len());
                self.mode = WorkingVimMode::Insert;
            }
            CrosstermKeyCode::Char('I') => {
                self.cursor = 0;
                self.mode = WorkingVimMode::Insert;
            }
            CrosstermKeyCode::Char('A') => {
                self.cursor = self.characters.len();
                self.mode = WorkingVimMode::Insert;
            }
            CrosstermKeyCode::Char('D') => self.characters.truncate(self.cursor),
            CrosstermKeyCode::Char('C') => {
                self.characters.truncate(self.cursor);
                self.mode = WorkingVimMode::Insert;
            }
            CrosstermKeyCode::Char('x') if self.cursor < self.characters.len() => {
                self.characters.remove(self.cursor);
            }
            _ => return WorkingInputAction::Ignored,
        }
        WorkingInputAction::Redraw
    }

    fn history_previous(&mut self) {
        if self.history.is_empty() {
            return;
        }
        let index = match self.history_index {
            Some(0) => 0,
            Some(index) => index - 1,
            None => {
                self.history_draft = Some(self.characters.clone());
                self.history.len() - 1
            }
        };
        self.set_history_entry(index);
    }

    fn history_next(&mut self) {
        let Some(index) = self.history_index else {
            return;
        };
        if index + 1 < self.history.len() {
            self.set_history_entry(index + 1);
        } else {
            self.characters = self.history_draft.take().unwrap_or_default();
            self.cursor = self.characters.len();
            self.history_index = None;
        }
    }

    fn set_history_entry(&mut self, index: usize) {
        self.characters = self.history[index].chars().collect();
        self.cursor = self.characters.len();
        self.history_index = Some(index);
    }

    fn delete_previous_word(&mut self) {
        let mut start = self.cursor;
        while start > 0 && self.characters[start - 1].is_whitespace() {
            start -= 1;
        }
        while start > 0 && !self.characters[start - 1].is_whitespace() {
            start -= 1;
        }
        self.characters.drain(start..self.cursor);
        self.cursor = start;
    }

    fn begin_find(&mut self, direction: FindDirection, till: bool) {
        self.pending_find = Some(PendingFind { direction, till });
    }

    fn apply_find(&mut self, motion: FindMotion) {
        let found = match motion.direction {
            FindDirection::Forward => ((self.cursor + 1)..self.characters.len())
                .find(|&index| self.characters[index] == motion.character),
            FindDirection::Backward => (0..self.cursor)
                .rev()
                .find(|&index| self.characters[index] == motion.character),
        };
        if let Some(index) = found {
            self.cursor = match (motion.direction, motion.till) {
                (FindDirection::Forward, true) => index.saturating_sub(1),
                (FindDirection::Backward, true) => (index + 1).min(self.characters.len()),
                _ => index,
            };
        }
    }

    fn move_word_forward(&mut self, big_word: bool) {
        let len = self.characters.len();
        if self.cursor >= len {
            return;
        }
        let class = word_class(self.characters[self.cursor], big_word);
        let mut index = self.cursor + 1;
        while index < len && word_class(self.characters[index], big_word) == class {
            index += 1;
        }
        while index < len && self.characters[index].is_whitespace() {
            index += 1;
        }
        self.cursor = index.min(len);
    }

    fn move_word_backward(&mut self, big_word: bool) {
        if self.cursor == 0 {
            return;
        }
        let mut index = self.cursor - 1;
        while index > 0 && self.characters[index].is_whitespace() {
            index -= 1;
        }
        let class = word_class(self.characters[index], big_word);
        while index > 0 && word_class(self.characters[index - 1], big_word) == class {
            index -= 1;
        }
        self.cursor = index;
    }

    fn move_word_end(&mut self, big_word: bool) {
        let len = self.characters.len();
        if self.cursor >= len {
            return;
        }
        let mut index = self.cursor;
        if !self.characters[index].is_whitespace() {
            index += 1;
        }
        while index < len && self.characters[index].is_whitespace() {
            index += 1;
        }
        if index >= len {
            self.cursor = len.saturating_sub(1);
            return;
        }
        let class = word_class(self.characters[index], big_word);
        while index + 1 < len && word_class(self.characters[index + 1], big_word) == class {
            index += 1;
        }
        self.cursor = index;
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum WordClass {
    Whitespace,
    Keyword,
    Punctuation,
}

fn word_class(character: char, big_word: bool) -> WordClass {
    if character.is_whitespace() {
        WordClass::Whitespace
    } else if big_word || character.is_alphanumeric() || character == '_' {
        WordClass::Keyword
    } else {
        WordClass::Punctuation
    }
}

fn reverse_find_direction(direction: FindDirection) -> FindDirection {
    match direction {
        FindDirection::Forward => FindDirection::Backward,
        FindDirection::Backward => FindDirection::Forward,
    }
}

pub struct EscAbortWatcher {
    stop: Arc<AtomicBool>,
    editor: Arc<std::sync::Mutex<WorkingVimEditor>>,
    handle: Option<tokio::task::JoinHandle<()>>,
}

impl EscAbortWatcher {
    pub fn disabled() -> Self {
        Self {
            stop: Arc::new(AtomicBool::new(true)),
            editor: Arc::new(std::sync::Mutex::new(WorkingVimEditor::default())),
            handle: None,
        }
    }

    pub fn spawn(cancellation_token: CancellationToken) -> Self {
        Self::spawn_with_display(cancellation_token, None, Vec::new())
    }

    pub fn spawn_with_display(
        cancellation_token: CancellationToken,
        display: Option<Arc<TerminalDisplay>>,
        history: Vec<String>,
    ) -> Self {
        let stop = Arc::new(AtomicBool::new(false));
        let editor = Arc::new(std::sync::Mutex::new(WorkingVimEditor::with_history(
            history,
        )));
        let handle = std::io::stdin().is_terminal().then(|| {
            let stop_watcher = Arc::clone(&stop);
            let watcher_editor = Arc::clone(&editor);
            tokio::task::spawn_blocking(move || {
                if InputModeGuard::enable().is_err() {
                    return;
                }
                let _input_mode = InputModeGuard;
                let mut spinner_frame = 0;
                let mut last_spinner_update = std::time::Instant::now();
                while !stop_watcher.load(Ordering::SeqCst) && !cancellation_token.is_cancelled() {
                    if last_spinner_update.elapsed() >= SPINNER_UPDATE_INTERVAL {
                        spinner_frame += 1;
                        if let Some(display) = &display {
                            display.update_spinner(spinner_frame);
                        }
                        last_spinner_update = std::time::Instant::now();
                    }
                    let next_event = {
                        let _input = crate::display::TERMINAL_INPUT
                            .lock()
                            .expect("terminal input lock poisoned");
                        match event::poll(Duration::from_millis(20)) {
                            Ok(true) => event::read().map(Some),
                            Ok(false) => Ok(None),
                            Err(error) => Err(error),
                        }
                    };
                    // Release the input lock before acquiring display/editor
                    // locks: a render transaction may query cursor position.
                    match next_event {
                        Ok(Some(event)) => {
                            if let Ok(mut editor) = watcher_editor.lock() {
                                match editor.apply(event) {
                                    WorkingInputAction::Abort => {
                                        cancellation_token.cancel();
                                        break;
                                    }
                                    WorkingInputAction::Redraw => {
                                        if let Some(display) = &display {
                                            display.update_working_input(
                                                &editor.text(),
                                                editor.cursor,
                                                editor.mode.label(),
                                            );
                                        }
                                    }
                                    WorkingInputAction::Ignored => {}
                                }
                            }
                        }
                        Ok(None) => {}
                        Err(_) => break,
                    }
                }
            })
        });
        Self {
            stop,
            editor,
            handle,
        }
    }

    pub async fn stop(self) -> String {
        self.stop.store(true, Ordering::SeqCst);
        if let Some(handle) = self.handle {
            let _ = handle.await;
        }
        self.editor
            .lock()
            .map(|editor| editor.text())
            .unwrap_or_default()
    }
}

#[cfg(unix)]
struct InputModeGuard;

#[cfg(unix)]
impl InputModeGuard {
    fn enable() -> std::io::Result<()> {
        use std::mem::MaybeUninit;

        let mut termios = MaybeUninit::<libc::termios>::uninit();
        // SAFETY: termios points to valid uninitialized storage for tcgetattr to fill.
        let result = unsafe { libc::tcgetattr(libc::STDIN_FILENO, termios.as_mut_ptr()) };
        if result != 0 {
            return Err(std::io::Error::last_os_error());
        }
        // SAFETY: tcgetattr succeeded, so termios has been initialized.
        let original = unsafe { termios.assume_init() };
        let mut input_mode = original;
        apply_esc_abort_input_mode(&mut input_mode);
        // SAFETY: input_mode is a valid termios value for stdin.
        if unsafe { libc::tcsetattr(libc::STDIN_FILENO, libc::TCSANOW, &input_mode) } != 0 {
            return Err(std::io::Error::last_os_error());
        }
        *terminal_mode_original()
            .lock()
            .expect("terminal mode lock poisoned") = Some(original);
        Ok(())
    }
}

#[cfg(unix)]
impl Drop for InputModeGuard {
    fn drop(&mut self) {
        if let Some(original) = terminal_mode_original()
            .lock()
            .expect("terminal mode lock poisoned")
            .take()
        {
            // SAFETY: original was captured from tcgetattr for stdin.
            let _ = unsafe { libc::tcsetattr(libc::STDIN_FILENO, libc::TCSANOW, &original) };
        }
    }
}

#[cfg(unix)]
fn terminal_mode_original() -> &'static std::sync::Mutex<Option<libc::termios>> {
    static ORIGINAL: std::sync::OnceLock<std::sync::Mutex<Option<libc::termios>>> =
        std::sync::OnceLock::new();
    ORIGINAL.get_or_init(|| std::sync::Mutex::new(None))
}

#[cfg(unix)]
pub fn restore_terminal_mode() {
    if let Some(original) = terminal_mode_original()
        .lock()
        .expect("terminal mode lock poisoned")
        .take()
    {
        // SAFETY: original was captured from tcgetattr for stdin.
        let _ = unsafe { libc::tcsetattr(libc::STDIN_FILENO, libc::TCSANOW, &original) };
    }
}

#[cfg(not(unix))]
pub fn restore_terminal_mode() {
    let _ = disable_raw_mode();
}

#[cfg(unix)]
fn apply_esc_abort_input_mode(termios: &mut libc::termios) {
    termios.c_iflag &= !(libc::BRKINT | libc::ICRNL | libc::INPCK | libc::ISTRIP | libc::IXON);
    termios.c_cflag |= libc::CS8;
    termios.c_lflag &= !(libc::ECHO | libc::ICANON | libc::IEXTEN);
    termios.c_cc[libc::VMIN] = 0;
    termios.c_cc[libc::VTIME] = 1;
}

#[cfg(not(unix))]
struct InputModeGuard;

#[cfg(not(unix))]
impl InputModeGuard {
    fn enable() -> std::io::Result<()> {
        enable_raw_mode()
    }
}

#[cfg(not(unix))]
impl Drop for InputModeGuard {
    fn drop(&mut self) {
        let _ = disable_raw_mode();
    }
}

fn capture_piped_prompt_prefill(args: &Args) -> Result<Option<String>, Box<dyn Error>> {
    if !args.query.is_empty()
        || args.command
        || args.talk
        || args.list_models
        || args.update_pricing
    {
        return Ok(None);
    }
    if std::io::stdin().is_terminal() {
        return Ok(None);
    }
    if !std::io::stdout().is_terminal() {
        return Ok(None);
    }

    let mut input = String::new();
    std::io::stdin().read_to_string(&mut input)?;
    let input = input.trim_end_matches(['\r', '\n']).to_string();
    if input.is_empty() {
        return Ok(None);
    }

    attach_stdin_to_stdout_tty()?;
    Ok(Some(input))
}

#[cfg(unix)]
fn attach_stdin_to_stdout_tty() -> Result<(), Box<dyn Error>> {
    use std::os::fd::AsRawFd;

    if unsafe { libc::dup2(std::io::stdout().as_raw_fd(), libc::STDIN_FILENO) } == -1 {
        Err(std::io::Error::last_os_error().into())
    } else {
        Ok(())
    }
}

#[cfg(not(unix))]
fn attach_stdin_to_stdout_tty() -> Result<(), Box<dyn Error>> {
    Ok(())
}

fn load_or_create_session(args: &Args, store: &SessionStore) -> Result<Session, Box<dyn Error>> {
    if !args.new
        && let Some(resume) = args.resume.as_deref()
    {
        let id = if resume == "__LATEST__" {
            None
        } else {
            Some(resume)
        };
        return Ok(store.load(id)?);
    }

    let mut messages = Vec::new();
    messages.push(AgentMessage::System {
        content: system_prompt(),
    });
    let memory = load_all_agents_memory(None);
    if !memory.is_empty() {
        messages.push(AgentMessage::System { content: memory });
    }
    messages.push(AgentMessage::System {
        content: format!("[SYSTEM INFO] pwd: {}", std::env::current_dir()?.display()),
    });

    Ok(Session::new(store.new_session_id(), messages))
}

fn print_agent_header(model_name: &str) {
    println!("\n[agent]\nmodel: {model_name}");
}

fn replay_session(session: &Session, display: &TerminalDisplay) {
    let mut tool_calls = std::collections::HashMap::new();
    for message in &session.messages {
        display.render_new_message(message);
        match message {
            AgentMessage::Assistant(assistant) => {
                for call in &assistant.tool_calls {
                    tool_calls.insert(call.id.clone(), call.clone());
                }
            }
            AgentMessage::Tool(result) => {
                let call = tool_calls.remove(&result.tool_call_id);
                print!(
                    "{}",
                    display.format_tool_result_for_call(result, call.as_ref())
                );
            }
            _ => {}
        }
    }
}

fn load_prompt_history(store: &SessionStore) -> reedline::Result<Vec<String>> {
    let history = FileBackedHistory::with_file(10_000, store.prompt_history_path())?;
    Ok(history
        .search(SearchQuery::everything(SearchDirection::Forward, None))?
        .into_iter()
        .map(|item| item.command_line)
        .collect())
}

async fn prompt_for_input(
    store: &SessionStore,
    session: &Session,
    model_name: &str,
    allow_git_writes: bool,
    prefill: Option<&str>,
) -> Result<PromptInput, Box<dyn Error>> {
    if !std::io::stdin().is_terminal() || !std::io::stdout().is_terminal() {
        return Err(
            "interactive mode requires a TTY; pass a query argument or use --single with a query"
                .into(),
        );
    }

    store.ensure_dirs()?;
    let history = Box::new(FileBackedHistory::with_file(
        10_000,
        store.prompt_history_path(),
    )?);
    let dynamic_candidates = Arc::new(RwLock::new(Vec::new()));
    spawn_completion_refresh(Arc::clone(&dynamic_candidates), store.clone());
    let completer = Box::new(AgentCompleter::with_dynamic_candidates_and_dir(
        prompt_completion_candidates(store),
        dynamic_candidates,
        std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")),
    ));
    let mut line_editor = Reedline::create()
        .use_bracketed_paste(true)
        .with_history(history)
        .with_completer(completer)
        .with_menu(ReedlineMenu::EngineCompleter(Box::new(
            ListMenu::default()
                .with_name(COMPLETION_MENU_NAME)
                .with_page_size(12)
                .with_max_entry_lines(1)
                .with_only_buffer_difference(false),
        )))
        .with_quick_completions(false)
        .with_partial_completions(false)
        .with_edit_mode(Box::new(agent_vi_mode()));
    set_cursor_style_for_mode(PromptEditMode::Vi(PromptViMode::Insert));
    let prompt = DefaultPrompt::default();
    print!(
        "{}",
        TerminalDisplay::format_standalone_footer(&format_cost_and_context_line(
            &session.messages,
            model_name,
            allow_git_writes
        ))
    );
    seed_line_editor_prefill(&mut line_editor, prefill);

    match line_editor.read_line(&prompt)? {
        Signal::Success(input) => Ok(PromptInput::Text(input)),
        Signal::HostCommand(command) if command == TOGGLE_TALK_HOST_COMMAND => {
            Ok(PromptInput::ToggleTalk)
        }
        Signal::CtrlD | Signal::CtrlC => Err("Goodbye!".into()),
        _ => Err("unsupported prompt signal".into()),
    }
}

fn seed_line_editor_prefill(line_editor: &mut Reedline, prefill: Option<&str>) {
    if let Some(prefill) = prefill.filter(|input| !input.is_empty()) {
        line_editor.run_edit_commands(&[
            EditCommand::InsertString(prefill.to_string()),
            EditCommand::MoveToEnd { select: false },
        ]);
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum SlashCommandResult {
    NotCommand,
    InvokeSkill(String),
    Handled,
    SwitchModel(String),
}

async fn handle_slash_command(
    input: &str,
    args: &Args,
    store: &SessionStore,
    session: &mut Session,
    display: &TerminalDisplay,
    model_name: &str,
    allow_git_writes: &mut bool,
) -> Result<SlashCommandResult, Box<dyn Error>> {
    let trimmed = input.trim();
    if !trimmed.starts_with('/') {
        return Ok(SlashCommandResult::NotCommand);
    }

    let (command, rest) = match trimmed.split_once(char::is_whitespace) {
        Some((command, rest)) => (command, rest.trim()),
        None => (trimmed, ""),
    };

    match command {
        "/help" => {
            println!("{}", slash_help(store)?);
        }
        "/session" => {
            println!(
                "{}",
                format_session_info(session, model_name, *allow_git_writes)
            );
        }
        "/compact" => {
            compact_session(store, session, model_name, rest).await?;
        }
        "/allow-git" => {
            *allow_git_writes = !*allow_git_writes;
            println!(
                "git write commands {} for this session",
                if *allow_git_writes {
                    "allowed"
                } else {
                    "blocked"
                }
            );
        }
        "/clear" | "/new" => {
            let new_session = load_or_create_session(
                &Args {
                    new: true,
                    resume: None,
                    images: Vec::new(),
                    query: Vec::new(),
                    single: args.single,
                    model: args.model.clone(),
                    list_models: false,
                    update_pricing: false,
                    talk: false,
                    command: false,
                    allow_git: *allow_git_writes,
                    no_completion_sound: args.no_completion_sound,
                    no_subagent: args.no_subagent,
                },
                store,
            )?;
            *session = new_session;
            store.save(session)?;
            println!("cleared");
            println!("sessionId: {}", session.session_id);
        }
        "/find" => {
            if rest.is_empty() {
                println!("Usage: /find <query>");
            } else {
                let matches =
                    store.find_sessions_excluding(rest, 160, Some(&session.session_id), 10)?;
                if matches.is_empty() {
                    println!("No conversations found for: {rest}");
                } else {
                    for found in matches {
                        println!(
                            "{}\n  {}\n  /resume {}",
                            found.session_id, found.excerpt, found.session_id
                        );
                    }
                }
            }
        }
        "/models" => {
            if rest.is_empty() {
                for model in list_models().await {
                    println!("{model}");
                }
            } else {
                return Ok(SlashCommandResult::SwitchModel(rest.to_string()));
            }
        }
        "/resume" => {
            let resume_id = match rest {
                "" | "latest" => None,
                other => Some(other.split('\t').next().unwrap_or(other)),
            };
            *session = store.load(resume_id)?;
            replay_session(session, display);
            println!("sessionId: {}", session.session_id);
        }
        "/pricing" => match rest {
            "refresh" => {
                println!("{}", refresh_pricing_cache(store.root()).await?);
            }
            "" | "help" => {
                println!("Usage: /pricing refresh");
            }
            other => {
                println!("Unknown pricing command: {other}. Try /pricing refresh");
            }
        },
        _ => {
            let skill_name = command.trim_start_matches('/');
            let project_dir = std::env::current_dir()?;
            if let Some(skill) = crate::skills::find(store.root(), &project_dir, skill_name)? {
                return Ok(SlashCommandResult::InvokeSkill(
                    crate::skills::invocation_prompt(&skill, rest),
                ));
            }
            println!("Unknown command: {command}. Try /help");
        }
    }
    Ok(SlashCommandResult::Handled)
}

async fn compact_session(
    store: &SessionStore,
    session: &mut Session,
    model_name: &str,
    focus: &str,
) -> Result<(), Box<dyn Error>> {
    const RECENT_USER_TURNS: usize = 2;

    let system_messages = session
        .messages
        .iter()
        .filter(|message| {
            matches!(message, AgentMessage::System { content } if !content.starts_with("[COMPACTED CONTEXT]"))
        })
        .cloned()
        .collect::<Vec<_>>();
    let user_starts = session
        .messages
        .iter()
        .enumerate()
        .filter_map(|(index, message)| {
            matches!(
                message,
                AgentMessage::User { .. } | AgentMessage::UserWithImages { .. }
            )
            .then_some(index)
        })
        .collect::<Vec<_>>();
    if user_starts.len() <= RECENT_USER_TURNS {
        println!("Nothing to compact; fewer than three user turns are present.");
        return Ok(());
    }

    let recent_start = user_starts[user_starts.len() - RECENT_USER_TURNS];
    let old_messages = &session.messages[..recent_start];
    let recent_messages = session.messages[recent_start..].to_vec();
    let transcript = old_messages
        .iter()
        .filter(|message| !matches!(message, AgentMessage::System { content } if !content.starts_with("[COMPACTED CONTEXT]")))
        .map(compaction_message_text)
        .collect::<Vec<_>>()
        .join("\n\n");
    if transcript.trim().is_empty() {
        println!("Nothing to compact.");
        return Ok(());
    }

    let focus_instruction = if focus.is_empty() {
        String::new()
    } else {
        format!("\nUser-requested focus: {focus}")
    };
    let summary_request = vec![
        AgentMessage::System {
            content: "Summarize conversation history into durable working memory for another AI agent. Preserve goals, requirements, decisions, verified facts, files changed, command/test results, unresolved issues, and next steps. Distinguish facts from assumptions. Be concise and do not include hidden reasoning. Return only the structured summary.".to_string(),
        },
        AgentMessage::User {
            content: format!(
                "Create a structured compacted context from this older transcript:{focus_instruction}\n\n{transcript}"
            ),
        },
    ];
    let summary = build_provider(model_name)?
        .complete(&summary_request, &[])
        .await?
        .content;
    if summary.trim().is_empty() {
        return Err("provider returned an empty compacted context".into());
    }

    let old_tokens = crate::agent::count_tokens(&session.messages, model_name);
    store.archive_before_compaction(session)?;
    let mut compacted = system_messages;
    compacted.push(AgentMessage::System {
        content: format!("[COMPACTED CONTEXT]\n{}", summary.trim()),
    });
    compacted.extend(recent_messages);
    let new_tokens = crate::agent::count_tokens(&compacted, model_name);
    let removed_messages = session.messages.len().saturating_sub(compacted.len());
    session.replace_messages(compacted);
    store.save(session)?;
    println!(
        "Compacted context: removed {removed_messages} messages; reclaimed approximately {} tokens ({old_tokens} -> {new_tokens}).",
        old_tokens.saturating_sub(new_tokens)
    );
    Ok(())
}

fn compaction_message_text(message: &AgentMessage) -> String {
    match message {
        AgentMessage::System { content } => format!("SYSTEM MEMORY:\n{content}"),
        AgentMessage::User { content } | AgentMessage::UserWithImages { content, .. } => {
            format!("USER:\n{content}")
        }
        AgentMessage::Assistant(assistant) => {
            let calls = assistant
                .tool_calls
                .iter()
                .map(|call| format!("tool call {}: {}", call.name, call.arguments))
                .collect::<Vec<_>>()
                .join("\n");
            format!("ASSISTANT:\n{}\n{calls}", assistant.content)
        }
        AgentMessage::Tool(result) => {
            format!(
                "TOOL {} ({:?}):\n{}",
                result.name, result.status, result.content
            )
        }
    }
}

fn format_session_info(session: &Session, model_name: &str, allow_git_writes: bool) -> String {
    format!(
        "Session ID: {}\nCreated: {}\nUpdated: {}\nMessages: {}\n{}",
        session.session_id,
        session.created_at.to_rfc3339(),
        session.updated_at.to_rfc3339(),
        session.messages.len(),
        format_cost_and_context_line(&session.messages, model_name, allow_git_writes),
    )
}

fn slash_help(store: &SessionStore) -> Result<String, Box<dyn Error>> {
    let mut help = "Available commands:\n  /allow-git\n      Toggle git commands that modify repositories for this session.\n  /clear, /new\n      Clear the UI and start a new conversation/session.\n  /compact [focus]\n      Summarize older turns into compact working context.\n  /find <query>\n      Search saved conversation histories.\n  /help\n      Show this help.\n  /models [<model_id>]\n      List models or switch the active model.\n  /pricing refresh\n      Download and cache LiteLLM pricing data.\n  /resume [latest|<session_id>]\n      Resume a saved conversation/session.\n  /session\n      Show information about the current session.\n\nSkills:\n  /<skill-name> [arguments]\n      Invoke a skill from .agent/skills or ~/.agent/skills.\n  Create ~/.agent/skills/<name>/SKILL.md (user) or .agent/skills/<name>/SKILL.md (project).\n  The agent can create these files with its file tools too.\n\n"
        .to_string();
    let project_dir = std::env::current_dir()?;
    for skill in crate::skills::discover(store.root(), &project_dir)? {
        help.push_str(&format!("  /{}\n", skill.name));
        if !skill.description.is_empty() {
            help.push_str(&format!("      {}\n", skill.description));
        }
    }
    Ok(help)
}

fn build_tool_registry(allow_git_writes: bool, include_spawn: bool) -> ToolRegistry {
    match (allow_git_writes, include_spawn) {
        (true, true) => ToolRegistry::new_with_git_write_access(),
        (false, true) => ToolRegistry::new(),
        (_, false) => ToolRegistry::without_spawn(),
    }
}

fn build_loop_runner(
    model_name: &str,
    allow_git_writes: bool,
    no_subagent: bool,
) -> Result<AgentLoop<Box<dyn Provider>>, Box<dyn Error>> {
    let tools = build_tool_registry(allow_git_writes, !no_subagent);
    Ok(AgentLoop::new(
        build_provider(model_name)?,
        tools,
        AgentLoopConfig {
            model: model_name.to_string(),
            ..AgentLoopConfig::default()
        },
    ))
}

fn system_prompt() -> String {
    "You are a highly autonomous AI command line agent designed to help users with software engineering tasks, system operations, research, and problem-solving. Be concise, direct, and action-oriented. You may create reusable skills as SKILL.md files under ~/.agent/skills/<name>/ for the user or .agent/skills/<name>/ for the current project. A skill may start with YAML-like frontmatter containing name and description, followed by its instructions; users invoke it as /<name> [arguments].".to_string()
}

fn command_mode_system_prompt() -> String {
    "Command-buffer mode: produce the text the user wants placed into their zsh prompt. Prefer a single bash/zsh command when the user is asking for a command. Return only the command/text to insert, with no Markdown fences or explanatory prose.".to_string()
}

fn spawn_completion_refresh(dynamic_candidates: Arc<RwLock<Vec<String>>>, store: SessionStore) {
    spawn_detached_completion_refresh(Arc::clone(&dynamic_candidates), move || {
        completion_candidates(&store, &[])
    });
    tokio::spawn(async move {
        merge_completion_candidates(
            &dynamic_candidates,
            list_models()
                .await
                .into_iter()
                .map(|model| format!("/models {model}")),
        );
    });
}

fn spawn_detached_completion_refresh<F>(dynamic_candidates: Arc<RwLock<Vec<String>>>, refresh: F)
where
    F: FnOnce() -> Vec<String> + Send + 'static,
{
    std::thread::spawn(move || {
        merge_completion_candidates(&dynamic_candidates, refresh());
    });
}

fn merge_completion_candidates(
    dynamic_candidates: &RwLock<Vec<String>>,
    candidates: impl IntoIterator<Item = String>,
) {
    if let Ok(mut refreshed) = dynamic_candidates.write() {
        refreshed.extend(candidates);
        *refreshed = dedup_preserving_order(std::mem::take(&mut *refreshed));
    }
}

fn prompt_completion_candidates(store: &SessionStore) -> Vec<String> {
    let mut candidates = vec![
        "/allow-git".to_string(),
        "/clear".to_string(),
        "/compact".to_string(),
        "/find".to_string(),
        "/help".to_string(),
        "/models".to_string(),
        "/new".to_string(),
        "/pricing refresh".to_string(),
        "/resume latest".to_string(),
        "/session".to_string(),
    ];
    candidates.extend(
        model_completion_values(store, &[])
            .into_iter()
            .map(|model| format!("/models {model}")),
    );
    candidates
}

fn completion_candidates(store: &SessionStore, available_models: &[String]) -> Vec<String> {
    completion_candidates_in_dir(
        store,
        available_models,
        &std::env::current_dir().unwrap_or_else(|_| std::path::PathBuf::from(".")),
    )
}

fn completion_candidates_in_dir(
    store: &SessionStore,
    available_models: &[String],
    project_dir: &std::path::Path,
) -> Vec<String> {
    let mut candidates = vec![
        "/allow-git".to_string(),
        "/clear".to_string(),
        "/compact".to_string(),
        "/find".to_string(),
        "/help".to_string(),
        "/models".to_string(),
        "/new".to_string(),
        "/pricing refresh".to_string(),
        "/resume".to_string(),
        "/session".to_string(),
    ];

    candidates.extend(
        model_completion_values(store, available_models)
            .into_iter()
            .map(|model| format!("/models {model}")),
    );

    if let Ok(labels) = store.list_session_labels(80) {
        candidates.extend(labels.into_iter().map(|label| format!("/resume {label}")));
    }
    candidates.push("/resume latest".to_string());
    if let Ok(skills) = crate::skills::discover(store.root(), project_dir) {
        candidates.extend(skills.into_iter().map(|skill| format!("/{}", skill.name)));
    }
    candidates
}

fn model_completion_values(store: &SessionStore, available_models: &[String]) -> Vec<String> {
    let mut models = vec![
        crate::providers::configuration::DEFAULT_MODEL.to_string(),
        format!("openai:{}", crate::providers::configuration::DEFAULT_MODEL),
    ];

    models.extend(available_models.iter().cloned());

    for env_name in ["AGENT_MODEL", "OPENAI_MODEL"] {
        if let Ok(model) = std::env::var(env_name)
            && !model.trim().is_empty()
        {
            models.push(model);
        }
    }
    if let Ok(model) = std::env::var("OLLAMA_MODEL")
        && !model.trim().is_empty()
    {
        models.push(format!("ollama:{model}"));
    }

    let _ = store;
    models.sort();
    models.dedup();
    models
}

pub fn completion_values_for_line(store: &SessionStore, line: &str, pos: usize) -> Vec<String> {
    completion_values_for_line_with_models(store, line, pos, &[])
}

pub fn completion_values_for_line_in_dir(
    store: &SessionStore,
    project_dir: &std::path::Path,
    line: &str,
    pos: usize,
) -> Vec<String> {
    AgentCompleter::new(completion_candidates_in_dir(store, &[], project_dir))
        .complete(line, pos)
        .suggestions()
        .iter()
        .map(|suggestion| suggestion.value.clone())
        .collect()
}

pub fn completion_values_for_line_with_models(
    store: &SessionStore,
    line: &str,
    pos: usize,
    available_models: &[String],
) -> Vec<String> {
    AgentCompleter::new(completion_candidates(store, available_models))
        .complete(line, pos)
        .suggestions()
        .iter()
        .map(|suggestion| suggestion.value.clone())
        .collect()
}

#[derive(Clone, Debug)]
struct AgentCompleter {
    candidates: Vec<String>,
    dynamic_candidates: Option<Arc<RwLock<Vec<String>>>>,
    working_dir: PathBuf,
}

impl AgentCompleter {
    fn new(candidates: Vec<String>) -> Self {
        Self::from_parts(candidates, None, current_working_dir())
    }

    #[cfg(test)]
    fn with_dynamic_candidates(
        candidates: Vec<String>,
        dynamic_candidates: Arc<RwLock<Vec<String>>>,
    ) -> Self {
        Self::from_parts(candidates, Some(dynamic_candidates), current_working_dir())
    }

    fn with_dynamic_candidates_and_dir(
        candidates: Vec<String>,
        dynamic_candidates: Arc<RwLock<Vec<String>>>,
        working_dir: PathBuf,
    ) -> Self {
        Self::from_parts(candidates, Some(dynamic_candidates), working_dir)
    }

    fn from_parts(
        candidates: Vec<String>,
        dynamic_candidates: Option<Arc<RwLock<Vec<String>>>>,
        working_dir: PathBuf,
    ) -> Self {
        Self {
            candidates: dedup_preserving_order(candidates),
            dynamic_candidates,
            working_dir,
        }
    }

    fn candidates(&self) -> Vec<String> {
        let mut candidates = self.candidates.clone();
        if let Some(dynamic_candidates) = &self.dynamic_candidates
            && let Ok(refreshed) = dynamic_candidates.read()
        {
            candidates.extend(refreshed.iter().cloned());
        }
        dedup_preserving_order(candidates)
    }
}

fn dedup_preserving_order(candidates: Vec<String>) -> Vec<String> {
    let mut seen = std::collections::HashSet::new();
    candidates
        .into_iter()
        .filter(|candidate| seen.insert(candidate.clone()))
        .collect()
}

impl AgentCompleter {
    fn complete_command(&self, prefix: &str, pos: usize) -> Vec<Suggestion> {
        let span = Span::new(0, pos);
        let candidates = self.candidates();
        let mut matches = candidates
            .iter()
            .enumerate()
            .filter_map(|(index, candidate)| {
                let score = completion_score(candidate, prefix)?;
                Some((score, index, candidate))
            })
            .collect::<Vec<_>>();
        matches.sort_by(
            |(left_score, left_index, _), (right_score, right_index, _)| {
                left_score
                    .cmp(right_score)
                    .then_with(|| left_index.cmp(right_index))
            },
        );

        matches
            .into_iter()
            .map(|(_, _, candidate)| Suggestion {
                value: candidate.clone(),
                span,
                append_whitespace: false,
                ..Default::default()
            })
            .collect()
    }

    fn complete_filename(&self, prefix: &str, pos: usize) -> Vec<Suggestion> {
        let token_start = prefix
            .char_indices()
            .rev()
            .find(|(_, character)| character.is_whitespace())
            .map_or(0, |(index, character)| index + character.len_utf8());
        let token = &prefix[token_start..];
        if token.is_empty() {
            return Vec::new();
        }
        let token = token.trim_end_matches('/');
        let token = if prefix[token_start..].ends_with('/') {
            format!("{token}/")
        } else {
            token.to_string()
        };

        let path = Path::new(&token);
        let (parent, name_prefix) = if token.ends_with('/') {
            (path, "")
        } else {
            (
                path.parent().unwrap_or_else(|| Path::new("")),
                path.file_name()
                    .and_then(|name| name.to_str())
                    .unwrap_or(""),
            )
        };
        let search_dir = self.working_dir.join(parent);
        let Ok(entries) = std::fs::read_dir(search_dir) else {
            return Vec::new();
        };

        let mut values = entries
            .filter_map(Result::ok)
            .filter_map(|entry| {
                let name = entry.file_name().into_string().ok()?;
                if name.starts_with('.') && !name_prefix.starts_with('.') {
                    return None;
                }
                if !name.to_lowercase().starts_with(&name_prefix.to_lowercase()) {
                    return None;
                }
                let mut value = parent.join(&name).to_string_lossy().into_owned();
                if entry.file_type().ok()?.is_dir() {
                    value.push('/');
                }
                Some(value)
            })
            .collect::<Vec<_>>();
        values.sort();

        values
            .into_iter()
            .map(|value| Suggestion {
                value,
                span: Span::new(token_start, pos),
                append_whitespace: false,
                ..Default::default()
            })
            .collect()
    }
}

fn current_working_dir() -> PathBuf {
    std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."))
}

impl Completer for AgentCompleter {
    fn complete(&mut self, line: &str, pos: usize) -> CompletionResult {
        let Some(prefix) = line.get(..pos) else {
            return CompletionResult::fresh(Vec::new());
        };
        CompletionResult::fresh(if prefix.starts_with('/') {
            self.complete_command(prefix, pos)
        } else {
            self.complete_filename(prefix, pos)
        })
    }
}

fn agent_vi_mode() -> SlashCompletionVi {
    let mut insert_keybindings = default_vi_insert_keybindings();
    let mut normal_keybindings = default_vi_normal_keybindings();
    add_completion_keybindings(&mut insert_keybindings);
    add_completion_keybindings(&mut normal_keybindings);
    SlashCompletionVi::new(Vi::new(insert_keybindings, normal_keybindings))
}

fn cursor_style_for_mode(mode: PromptEditMode) -> crossterm::cursor::SetCursorStyle {
    use crossterm::cursor::SetCursorStyle;

    match mode {
        PromptEditMode::Vi(PromptViMode::Insert) => SetCursorStyle::BlinkingBar,
        _ => SetCursorStyle::SteadyBlock,
    }
}

fn set_cursor_style_for_mode(mode: PromptEditMode) {
    use std::io::{IsTerminal, stdout};

    let mut out = stdout();
    if !out.is_terminal() {
        return;
    }

    let _ = crossterm::execute!(out, cursor_style_for_mode(mode));
}

struct SlashCompletionVi {
    inner: Vi,
    slash_completion_active: bool,
}

impl SlashCompletionVi {
    fn new(inner: Vi) -> Self {
        Self {
            inner,
            slash_completion_active: false,
        }
    }

    fn handle_event(&mut self, event: ReedlineEvent) -> ReedlineEvent {
        match event {
            ReedlineEvent::Edit(commands) if inserts_slash(&commands) => {
                self.slash_completion_active = true;
                ReedlineEvent::Multiple(vec![
                    ReedlineEvent::Edit(commands),
                    ReedlineEvent::Menu(COMPLETION_MENU_NAME.to_string()),
                ])
            }
            ReedlineEvent::Enter | ReedlineEvent::Submit if self.slash_completion_active => {
                self.slash_completion_active = false;
                ReedlineEvent::Enter
            }
            ReedlineEvent::Esc => {
                self.slash_completion_active = false;
                ReedlineEvent::Esc
            }
            ReedlineEvent::Multiple(events) => ReedlineEvent::Multiple(
                events
                    .into_iter()
                    .map(|event| self.handle_event(event))
                    .collect(),
            ),
            other => other,
        }
    }
}

impl EditMode for SlashCompletionVi {
    fn parse_event(&mut self, event: ReedlineRawEvent) -> ReedlineEvent {
        let crossterm_event: Event = event.into();
        if is_toggle_talk_key(&crossterm_event) {
            return ReedlineEvent::ExecuteHostCommand(TOGGLE_TALK_HOST_COMMAND.to_string());
        }

        let newline_key = matches!(
            &crossterm_event,
            Event::Key(key)
                if (key.code == CrosstermKeyCode::Enter && !key.modifiers.is_empty())
                    || (key.code == CrosstermKeyCode::Char('j')
                        && key.modifiers.contains(CrosstermKeyModifiers::CONTROL))
        );
        let was_insert_mode = matches!(
            self.inner.edit_mode(),
            PromptEditMode::Vi(PromptViMode::Insert)
        );

        if newline_key && was_insert_mode {
            return ReedlineEvent::Edit(vec![EditCommand::InsertNewline]);
        }

        let event = ReedlineRawEvent::try_from(crossterm_event)
            .map(|event| self.inner.parse_event(event))
            .unwrap_or(ReedlineEvent::None);

        set_cursor_style_for_mode(self.inner.edit_mode());
        self.handle_event(event)
    }

    fn edit_mode(&self) -> PromptEditMode {
        self.inner.edit_mode()
    }
}

fn is_toggle_talk_key(event: &Event) -> bool {
    matches!(
        event,
        Event::Key(key)
            if key.kind == KeyEventKind::Press
                && matches!(key.code, CrosstermKeyCode::Char('t' | 'T'))
                && key.modifiers.contains(CrosstermKeyModifiers::CONTROL)
    )
}

fn inserts_slash(commands: &[EditCommand]) -> bool {
    commands.iter().any(|command| match command {
        EditCommand::InsertChar('/') => true,
        EditCommand::InsertString(text) => text.starts_with('/'),
        _ => false,
    })
}
fn add_completion_keybindings(keybindings: &mut Keybindings) {
    keybindings.add_binding(
        KeyModifiers::CONTROL,
        KeyCode::Char('t'),
        ReedlineEvent::ExecuteHostCommand(TOGGLE_TALK_HOST_COMMAND.to_string()),
    );
    keybindings.add_binding(
        KeyModifiers::NONE,
        KeyCode::Up,
        ReedlineEvent::UntilFound(vec![ReedlineEvent::MenuPrevious, ReedlineEvent::Up]),
    );
    keybindings.add_binding(
        KeyModifiers::NONE,
        KeyCode::Down,
        ReedlineEvent::UntilFound(vec![ReedlineEvent::MenuNext, ReedlineEvent::Down]),
    );
    keybindings.add_binding(
        KeyModifiers::NONE,
        KeyCode::Tab,
        ReedlineEvent::UntilFound(vec![
            ReedlineEvent::Menu(COMPLETION_MENU_NAME.to_string()),
            ReedlineEvent::MenuNext,
            ReedlineEvent::Edit(vec![EditCommand::Complete]),
        ]),
    );
    keybindings.add_binding(
        KeyModifiers::SHIFT,
        KeyCode::BackTab,
        ReedlineEvent::MenuPrevious,
    );
}

fn completion_score(candidate: &str, query: &str) -> Option<usize> {
    let candidate = candidate.to_lowercase();
    let query = query.to_lowercase();
    if candidate.starts_with(&query) {
        return Some(0);
    }

    fuzzy_subsequence_score(&candidate, &query).map(|score| score + 1_000)
}

fn fuzzy_subsequence_score(candidate: &str, query: &str) -> Option<usize> {
    let mut score = 0;
    let mut search_start = 0;
    for query_char in query.chars() {
        let rest = candidate.get(search_start..)?;
        let (offset, _) = rest.char_indices().find(|(_, ch)| *ch == query_char)?;
        score += offset;
        search_start += offset + query_char.len_utf8();
    }
    Some(score)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_subagent_agents_do_not_receive_the_spawn_tool() {
        let registry = build_tool_registry(false, false);
        let names = registry
            .definitions()
            .iter()
            .map(|definition| definition.name.as_str())
            .collect::<Vec<_>>();

        assert!(!names.contains(&"spawn"));
    }

    #[test]
    fn primary_agents_receive_the_spawn_tool() {
        let registry = build_tool_registry(false, true);
        let names = registry
            .definitions()
            .iter()
            .map(|definition| definition.name.as_str())
            .collect::<Vec<_>>();

        assert!(names.contains(&"spawn"));
    }

    #[tokio::test]
    async fn allow_git_slash_command_toggles_git_write_access() {
        let temp = tempfile::tempdir().expect("temp dir");
        let store = SessionStore::with_root(temp.path().join(".agent"));
        let args = Args::parse_from(["agent"]);
        let display = TerminalDisplay::new();
        let mut session = Session::new("test-session".to_string(), Vec::new());
        let mut allow_git_writes = false;

        assert_eq!(
            handle_slash_command(
                "/allow-git",
                &args,
                &store,
                &mut session,
                &display,
                "mock",
                &mut allow_git_writes,
            )
            .await
            .expect("enable command"),
            SlashCommandResult::Handled
        );
        assert!(allow_git_writes);

        handle_slash_command(
            "/allow-git",
            &args,
            &store,
            &mut session,
            &display,
            "mock",
            &mut allow_git_writes,
        )
        .await
        .expect("disable command");
        assert!(!allow_git_writes);
    }

    #[test]
    fn spinner_animation_advances_at_a_relaxed_cadence() {
        assert_eq!(SPINNER_UPDATE_INTERVAL, Duration::from_millis(128));
    }

    #[test]
    fn detached_completion_work_does_not_delay_tokio_shutdown() {
        let runtime = tokio::runtime::Runtime::new().expect("runtime");
        let candidates = Arc::new(RwLock::new(Vec::new()));
        let (started_tx, started_rx) = std::sync::mpsc::channel();
        let (release_tx, release_rx) = std::sync::mpsc::channel();

        spawn_detached_completion_refresh(Arc::clone(&candidates), move || {
            started_tx.send(()).expect("signal start");
            release_rx.recv().expect("wait for release");
            vec!["/resume refreshed".to_string()]
        });
        started_rx.recv().expect("refresh started");

        let releaser = std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(300));
            release_tx.send(()).expect("release refresh");
        });
        let shutdown_started = std::time::Instant::now();
        drop(runtime);
        assert!(
            shutdown_started.elapsed() < Duration::from_millis(100),
            "runtime shutdown waited for completion refresh"
        );

        releaser.join().expect("releaser");
    }

    #[test]
    fn prompt_completion_candidates_defer_saved_session_labels() {
        let temp = tempfile::tempdir().expect("temp dir");
        let store = SessionStore::with_root(temp.path().join(".agent"));
        store
            .save(&Session::new(
                "saved-session".to_string(),
                vec![AgentMessage::User {
                    content: "saved conversation".to_string(),
                }],
            ))
            .expect("save session");

        let candidates = prompt_completion_candidates(&store);

        assert!(candidates.contains(&"/resume latest".to_string()));
        assert!(
            !candidates
                .iter()
                .any(|candidate| candidate.contains("saved-session"))
        );
    }

    #[test]
    fn parses_requested_short_flags() {
        let args = Args::parse_from(["agent", "-l", "-u", "-r", "session-id"]);

        assert!(args.list_models);
        assert!(args.update_pricing);
        assert_eq!(args.resume.as_deref(), Some("session-id"));
    }

    #[test]
    fn parses_new_short_flag() {
        let args = Args::parse_from(["agent", "-n"]);
        assert!(args.new);
    }

    #[test]
    fn talk_single_accepts_an_initial_query() {
        let args = Args::parse_from(["agent", "--talk", "--single", "hello"]);

        assert!(validate_args(&args).is_ok());
    }

    #[test]
    fn talk_single_requires_an_initial_query() {
        let args = Args::parse_from(["agent", "--talk", "--single"]);

        assert_eq!(
            validate_args(&args).unwrap_err().to_string(),
            "--talk --single requires an initial query"
        );
    }

    #[test]
    fn talk_mode_accepts_image_argument() {
        let args = Args::parse_from(["agent", "--talk", "-i", "photo.png"]);

        assert!(args.talk);
        assert_eq!(args.images, vec![std::path::PathBuf::from("photo.png")]);
    }

    #[test]
    fn dragged_image_path_becomes_an_attachment() {
        let temp = tempfile::tempdir().expect("temp dir");
        let image = temp.path().join("screen shot.png");
        std::fs::write(&image, b"png").expect("image");
        let escaped = image.display().to_string().replace(' ', "\\ ");

        let parsed = parse_user_input(&format!("what is this? {escaped}"), &[])
            .expect("parsed dragged image");

        assert_eq!(parsed.content, "what is this?");
        assert_eq!(parsed.images.len(), 1);
        assert_eq!(parsed.images[0].media_type, "image/png");
    }

    #[test]
    fn dragged_macos_screenshot_with_narrow_no_break_space_is_attached() {
        let temp = tempfile::tempdir().expect("temp dir");
        let image = temp
            .path()
            .join("Screenshot 2026-07-06 at 9.48.19\u{202f}AM.png");
        std::fs::write(&image, b"png").expect("image");
        let escaped = image.display().to_string().replace(' ', "\\ ");

        let parsed = parse_user_input(&format!("{escaped}."), &[]).expect("parsed dragged image");

        assert_eq!(parsed.content, "Describe this image.");
        assert_eq!(parsed.images.len(), 1);
    }

    #[test]
    fn dragging_only_an_image_supplies_a_default_prompt() {
        let temp = tempfile::tempdir().expect("temp dir");
        let image = temp.path().join("photo.jpg");
        std::fs::write(&image, b"jpg").expect("image");

        let parsed =
            parse_user_input(&format!("'{}'", image.display()), &[]).expect("parsed dragged image");

        assert_eq!(parsed.content, "Describe this image.");
        assert_eq!(parsed.images.len(), 1);
    }

    #[test]
    fn talk_resume_loads_existing_session_history() {
        let temp = tempfile::tempdir().expect("temp dir");
        let store = SessionStore::with_root(temp.path().join(".agent"));
        store
            .save(&Session::new(
                "voice-session".to_string(),
                vec![AgentMessage::User {
                    content: "remember blue".to_string(),
                }],
            ))
            .expect("save session");
        for argv in [
            vec!["agent", "--talk", "--resume", "voice-session"],
            vec!["agent", "-t", "--resume"],
        ] {
            let args = Args::parse_from(argv);
            let session = load_or_create_session(&args, &store).expect("loaded session");

            assert_eq!(session.session_id, "voice-session");
            assert_eq!(
                session.messages,
                vec![AgentMessage::User {
                    content: "remember blue".to_string()
                }]
            );
        }
    }

    #[test]
    fn text_without_resume_starts_a_new_session() {
        let temp = tempfile::tempdir().expect("temp dir");
        let store = SessionStore::with_root(temp.path().join(".agent"));
        store
            .save(&Session::new(
                "other-session".to_string(),
                vec![AgentMessage::Assistant(crate::agent::AssistantMessage {
                    content: "stale output from another session".to_string(),
                    tool_calls: Vec::new(),
                    usage: None,
                    model: None,
                    metadata: serde_json::Map::new(),
                })],
            ))
            .expect("save session");
        let args = Args::parse_from(["agent"]);

        let session = load_or_create_session(&args, &store).expect("new session");

        assert_ne!(session.session_id, "other-session");
        assert!(
            session
                .messages
                .iter()
                .all(|message| { message.content() != "stale output from another session" })
        );
    }

    #[test]
    fn assistant_stream_buffer_suppresses_a_repeated_response() {
        let mut buffer = AssistantStreamBuffer::default();

        assert_eq!(buffer.push("Fixed. "), Some("Fixed. ".to_string()));
        assert_eq!(buffer.push("Done."), Some("Done.".to_string()));
        assert_eq!(buffer.push("Fixed. "), None);
        assert_eq!(buffer.push("Done."), None);
        assert_eq!(buffer.finish(), "");
        assert_eq!(buffer.rendered(), "Fixed. Done.");
    }

    #[test]
    fn assistant_stream_buffer_preserves_nonduplicate_continuation() {
        let mut buffer = AssistantStreamBuffer::default();

        assert_eq!(buffer.push("Fixed. "), Some("Fixed. ".to_string()));
        assert_eq!(
            buffer.push("Still working."),
            Some("Still working.".to_string())
        );
        assert_eq!(buffer.rendered(), "Fixed. Still working.");
    }

    #[test]
    fn talk_without_resume_starts_a_new_session() {
        let temp = tempfile::tempdir().expect("temp dir");
        let store = SessionStore::with_root(temp.path().join(".agent"));
        store
            .save(&Session::new(
                "latest-voice-session".to_string(),
                vec![AgentMessage::User {
                    content: "prior voice turn".to_string(),
                }],
            ))
            .expect("save session");
        for flag in ["-t", "--talk"] {
            let args = Args::parse_from(["agent", flag]);

            let session = load_or_create_session(&args, &store).expect("new session");

            assert_ne!(session.session_id, "latest-voice-session");
            assert!(
                session
                    .messages
                    .iter()
                    .all(|message| message.content() != "prior voice turn")
            );
        }
    }

    #[test]
    fn working_vim_arrows_navigate_prompt_history_and_restore_draft() {
        let mut editor = WorkingVimEditor::with_history(vec![
            "first prompt".to_string(),
            "second prompt".to_string(),
        ]);
        editor.apply(Event::Paste("draft".to_string()));

        editor.apply(key_event(CrosstermKeyCode::Up));
        assert_eq!(editor.text(), "second prompt");
        editor.apply(key_event(CrosstermKeyCode::Up));
        assert_eq!(editor.text(), "first prompt");
        editor.apply(key_event(CrosstermKeyCode::Up));
        assert_eq!(editor.text(), "first prompt");

        editor.apply(key_event(CrosstermKeyCode::Down));
        assert_eq!(editor.text(), "second prompt");
        editor.apply(key_event(CrosstermKeyCode::Down));
        assert_eq!(editor.text(), "draft");
    }

    #[test]
    fn working_vim_normal_mode_arrows_navigate_prompt_history() {
        let mut editor = WorkingVimEditor::with_history(vec!["previous".to_string()]);
        editor.apply(key_event(CrosstermKeyCode::Esc));

        assert_eq!(
            editor.apply(key_event(CrosstermKeyCode::Up)),
            WorkingInputAction::Redraw
        );
        assert_eq!(editor.text(), "previous");
        assert_eq!(editor.cursor, "previous".chars().count());
        assert_eq!(editor.mode, WorkingVimMode::Normal);
    }

    #[test]
    fn working_vim_insert_esc_enters_normal_then_normal_esc_aborts() {
        let mut editor = WorkingVimEditor::default();
        assert_eq!(
            editor.apply(key_event(CrosstermKeyCode::Char('h'))),
            WorkingInputAction::Redraw
        );
        assert_eq!(editor.text(), "h");
        assert_eq!(editor.mode, WorkingVimMode::Insert);

        assert_eq!(
            editor.apply(key_event(CrosstermKeyCode::Esc)),
            WorkingInputAction::Redraw
        );
        assert_eq!(editor.mode, WorkingVimMode::Normal);
        assert_eq!(
            editor.apply(key_event(CrosstermKeyCode::Esc)),
            WorkingInputAction::Abort
        );
    }

    #[test]
    fn working_vim_normal_mode_supports_navigation_editing_and_insert() {
        let mut editor = WorkingVimEditor::default();
        for character in "helo".chars() {
            editor.apply(key_event(CrosstermKeyCode::Char(character)));
        }
        editor.apply(key_event(CrosstermKeyCode::Esc));
        editor.apply(key_event(CrosstermKeyCode::Char('h')));
        editor.apply(key_event(CrosstermKeyCode::Char('i')));
        editor.apply(key_event(CrosstermKeyCode::Char('l')));

        assert_eq!(editor.text(), "hello");
        assert_eq!(editor.cursor, 4);
        assert_eq!(editor.mode, WorkingVimMode::Insert);
    }

    #[test]
    fn working_vim_normal_mode_supports_change_and_delete_to_line_end() {
        let mut editor = editor_in_normal_mode("hello world");
        editor.cursor = 6;

        editor.apply(key_event(CrosstermKeyCode::Char('D')));
        assert_eq!(editor.text(), "hello ");
        assert_eq!(editor.cursor, 6);
        assert_eq!(editor.mode, WorkingVimMode::Normal);

        editor = editor_in_normal_mode("hello world");
        editor.cursor = 6;
        editor.apply(key_event(CrosstermKeyCode::Char('C')));
        assert_eq!(editor.text(), "hello ");
        assert_eq!(editor.cursor, 6);
        assert_eq!(editor.mode, WorkingVimMode::Insert);
    }

    #[test]
    fn working_vim_insert_mode_ctrl_w_deletes_the_previous_word() {
        let mut editor = WorkingVimEditor::default();
        editor.apply(Event::Paste("hello, world".to_string()));

        assert_eq!(
            editor.apply(Event::Key(crossterm::event::KeyEvent::new(
                CrosstermKeyCode::Char('w'),
                CrosstermKeyModifiers::CONTROL,
            ))),
            WorkingInputAction::Redraw
        );
        assert_eq!(editor.text(), "hello, ");
        assert_eq!(editor.cursor, 7);

        editor.apply(Event::Key(crossterm::event::KeyEvent::new(
            CrosstermKeyCode::Char('w'),
            CrosstermKeyModifiers::CONTROL,
        )));
        assert_eq!(editor.text(), "");
        assert_eq!(editor.cursor, 0);
    }

    #[test]
    fn working_vim_supports_word_motions() {
        let mut editor = editor_in_normal_mode("one two-three FOUR");

        editor.apply(key_event(CrosstermKeyCode::Char('w')));
        assert_eq!(editor.cursor, 4);
        editor.apply(key_event(CrosstermKeyCode::Char('e')));
        assert_eq!(editor.cursor, 6);
        editor.apply(key_event(CrosstermKeyCode::Char('w')));
        assert_eq!(editor.cursor, 7);
        editor.apply(key_event(CrosstermKeyCode::Char('W')));
        assert_eq!(editor.cursor, 14);
        editor.apply(key_event(CrosstermKeyCode::Char('B')));
        assert_eq!(editor.cursor, 4);
        editor.apply(key_event(CrosstermKeyCode::Char('b')));
        assert_eq!(editor.cursor, 0);
        editor.apply(key_event(CrosstermKeyCode::Char('E')));
        assert_eq!(editor.cursor, 2);
    }

    #[test]
    fn working_vim_supports_find_till_and_repeat_motions() {
        let mut editor = editor_in_normal_mode("abc def ghi def");

        editor.apply(key_event(CrosstermKeyCode::Char('f')));
        assert_eq!(
            editor.apply(key_event(CrosstermKeyCode::Char('d'))),
            WorkingInputAction::Redraw
        );
        assert_eq!(editor.cursor, 4);
        editor.apply(key_event(CrosstermKeyCode::Char(';')));
        assert_eq!(editor.cursor, 12);
        editor.apply(key_event(CrosstermKeyCode::Char(',')));
        assert_eq!(editor.cursor, 4);

        editor.apply(key_event(CrosstermKeyCode::Char('t')));
        editor.apply(key_event(CrosstermKeyCode::Char('g')));
        assert_eq!(editor.cursor, 7);
        editor.apply(key_event(CrosstermKeyCode::Char('F')));
        editor.apply(key_event(CrosstermKeyCode::Char('a')));
        assert_eq!(editor.cursor, 0);
        editor.cursor = 4;
        editor.apply(key_event(CrosstermKeyCode::Char('T')));
        editor.apply(key_event(CrosstermKeyCode::Char('a')));
        assert_eq!(editor.cursor, 1);
    }

    #[test]
    fn working_vim_esc_cancels_pending_find_before_aborting() {
        let mut editor = editor_in_normal_mode("abc");
        editor.apply(key_event(CrosstermKeyCode::Char('f')));

        assert_eq!(
            editor.apply(key_event(CrosstermKeyCode::Esc)),
            WorkingInputAction::Redraw
        );
        assert_eq!(
            editor.apply(key_event(CrosstermKeyCode::Esc)),
            WorkingInputAction::Abort
        );
    }

    fn editor_in_normal_mode(text: &str) -> WorkingVimEditor {
        let mut editor = WorkingVimEditor::default();
        editor.apply(Event::Paste(text.to_string()));
        editor.apply(key_event(CrosstermKeyCode::Esc));
        editor.cursor = 0;
        editor
    }

    #[test]
    fn working_input_resize_redraws_without_changing_the_draft() {
        let mut editor = WorkingVimEditor::default();
        editor.apply(Event::Paste("draft 漢字".into()));
        let cursor = editor.cursor;
        assert_eq!(
            editor.apply(Event::Resize(40, 12)),
            WorkingInputAction::Redraw
        );
        assert_eq!(editor.text(), "draft 漢字");
        assert_eq!(editor.cursor, cursor);
    }

    #[test]
    fn working_vim_accepts_paste_and_backspace_in_insert_mode() {
        let mut editor = WorkingVimEditor::default();
        assert_eq!(
            editor.apply(Event::Paste("hello".to_string())),
            WorkingInputAction::Redraw
        );
        editor.apply(key_event(CrosstermKeyCode::Backspace));
        assert_eq!(editor.text(), "hell");
    }

    #[test]
    fn working_vim_ignores_control_shortcuts() {
        let mut editor = WorkingVimEditor::default();
        assert_eq!(
            editor.apply(Event::Key(crossterm::event::KeyEvent::new(
                CrosstermKeyCode::Char('c'),
                CrosstermKeyModifiers::CONTROL,
            ))),
            WorkingInputAction::Ignored
        );
        assert!(editor.text().is_empty());
    }

    fn key_event(code: CrosstermKeyCode) -> Event {
        Event::Key(crossterm::event::KeyEvent::new(
            code,
            CrosstermKeyModifiers::NONE,
        ))
    }

    #[test]
    fn esc_abort_input_mode_preserves_output_processing() {
        // SAFETY: the test fills the fields it asserts against before reading them.
        let mut termios = unsafe { std::mem::zeroed::<libc::termios>() };
        termios.c_iflag = libc::BRKINT | libc::ICRNL | libc::INPCK | libc::ISTRIP | libc::IXON;
        termios.c_oflag = libc::OPOST;
        termios.c_lflag = libc::ECHO | libc::ICANON | libc::IEXTEN | libc::ISIG;

        super::apply_esc_abort_input_mode(&mut termios);

        assert_eq!(termios.c_oflag & libc::OPOST, libc::OPOST);
        assert_eq!(termios.c_lflag & libc::ICANON, 0);
        assert_eq!(termios.c_lflag & libc::ECHO, 0);
        assert_eq!(termios.c_lflag & libc::IEXTEN, 0);
        assert_eq!(termios.c_lflag & libc::ISIG, libc::ISIG);
        assert_eq!(termios.c_cc[libc::VMIN], 0);
        assert_eq!(termios.c_cc[libc::VTIME], 1);
    }
}

#[cfg(test)]
mod completion_input_tests {
    use super::*;
    use crossterm::event::{Event, KeyEvent};

    fn key(code: KeyCode) -> ReedlineRawEvent {
        modified_key(code, KeyModifiers::NONE)
    }

    fn modified_key(code: KeyCode, modifiers: KeyModifiers) -> ReedlineRawEvent {
        ReedlineRawEvent::try_from(Event::Key(KeyEvent::new(code, modifiers)))
            .expect("reedline raw event")
    }

    #[test]
    fn prefill_seeds_existing_reedline_buffer_at_end() {
        let mut line_editor = Reedline::create();

        seed_line_editor_prefill(&mut line_editor, Some("alpha\nbeta"));

        assert_eq!(line_editor.current_buffer_contents(), "alpha\nbeta");
        assert_eq!(line_editor.current_insertion_point(), "alpha\nbeta".len());
    }

    #[test]
    fn cursor_style_for_insert_and_normal_modes() {
        assert!(matches!(
            cursor_style_for_mode(PromptEditMode::Vi(PromptViMode::Insert)),
            crossterm::cursor::SetCursorStyle::BlinkingBar
        ));
        assert!(matches!(
            cursor_style_for_mode(PromptEditMode::Vi(PromptViMode::Normal)),
            crossterm::cursor::SetCursorStyle::SteadyBlock
        ));
    }

    #[test]
    fn ctrl_t_returns_toggle_talk_host_command() {
        let mut mode = agent_vi_mode();

        assert_eq!(
            mode.parse_event(modified_key(KeyCode::Char('t'), KeyModifiers::CONTROL)),
            ReedlineEvent::ExecuteHostCommand(TOGGLE_TALK_HOST_COMMAND.to_string())
        );
    }

    #[test]
    fn shift_ctrl_t_returns_toggle_talk_host_command() {
        let mut mode = agent_vi_mode();

        assert_eq!(
            mode.parse_event(modified_key(
                KeyCode::Char('T'),
                KeyModifiers::CONTROL | KeyModifiers::SHIFT,
            )),
            ReedlineEvent::ExecuteHostCommand(TOGGLE_TALK_HOST_COMMAND.to_string())
        );
    }

    #[test]
    fn down_arrow_cycles_completion_menu_when_active() {
        let mut mode = agent_vi_mode();

        assert_eq!(
            mode.parse_event(key(KeyCode::Down)),
            ReedlineEvent::UntilFound(vec![ReedlineEvent::MenuNext, ReedlineEvent::Down])
        );
    }

    #[test]
    fn up_arrow_cycles_completion_menu_when_active() {
        let mut mode = agent_vi_mode();

        assert_eq!(
            mode.parse_event(key(KeyCode::Up)),
            ReedlineEvent::UntilFound(vec![ReedlineEvent::MenuPrevious, ReedlineEvent::Up])
        );
    }

    #[test]
    fn completion_menu_queries_full_buffer_on_activation() {
        use reedline::{Editor, Menu, MenuEvent};

        let temp = tempfile::tempdir().expect("temp dir");
        let store = SessionStore::with_root(temp.path().join(".agent"));
        store
            .save(&Session::new(
                "recent".to_string(),
                vec![AgentMessage::User {
                    content: "recent conversation".to_string(),
                }],
            ))
            .expect("save session");
        let mut completer = AgentCompleter::new(completion_candidates(&store, &[]));
        let mut menu = ListMenu::default()
            .with_name(COMPLETION_MENU_NAME)
            .with_page_size(12)
            .with_max_entry_lines(1)
            .with_only_buffer_difference(false);
        let mut editor = Editor::default();
        editor.edit_buffer(
            |buffer| {
                buffer.set_buffer("/resume ".to_string());
                buffer.set_insertion_point("/resume ".len());
            },
            reedline::UndoBehavior::CreateUndoPoint,
        );

        menu.menu_event(MenuEvent::Activate(false));
        menu.update_values(&mut editor, &mut completer);

        assert!(!menu.get_values().is_empty());
    }

    #[test]
    fn enter_after_auto_opened_slash_completion_accepts_without_submitting() {
        let mut mode = agent_vi_mode();

        assert!(matches!(
            mode.parse_event(key(KeyCode::Char('/'))),
            ReedlineEvent::Multiple(events)
                if events == vec![
                    ReedlineEvent::Edit(vec![EditCommand::InsertChar('/')]),
                    ReedlineEvent::Menu(COMPLETION_MENU_NAME.to_string()),
                ]
        ));

        let event = mode.parse_event(key(KeyCode::Enter));
        assert!(
            matches!(event, ReedlineEvent::Enter)
                || matches!(event, ReedlineEvent::Multiple(events) if matches!(events.last(), Some(ReedlineEvent::Enter)))
        );
    }

    #[test]
    fn enter_submits_and_iterm_shift_enter_inserts_newline_in_insert_mode() {
        let mut mode = agent_vi_mode();

        let event = mode.parse_event(key(KeyCode::Enter));
        assert!(
            matches!(event, ReedlineEvent::Enter)
                || matches!(event, ReedlineEvent::Multiple(events) if matches!(events.last(), Some(ReedlineEvent::Enter)))
        );
        assert_eq!(
            mode.parse_event(modified_key(KeyCode::Char('j'), KeyModifiers::CONTROL)),
            ReedlineEvent::Edit(vec![EditCommand::InsertNewline])
        );
        assert_eq!(
            mode.parse_event(modified_key(KeyCode::Enter, KeyModifiers::SHIFT)),
            ReedlineEvent::Edit(vec![EditCommand::InsertNewline])
        );

        assert!(matches!(
            mode.parse_event(key(KeyCode::Esc)),
            ReedlineEvent::Multiple(events)
                if matches!(events.first(), Some(ReedlineEvent::Esc))
                    && matches!(events.last(), Some(ReedlineEvent::Repaint))
        ));
        let event = mode.parse_event(key(KeyCode::Enter));
        assert!(
            matches!(event, ReedlineEvent::Enter)
                || matches!(event, ReedlineEvent::Multiple(events) if matches!(events.last(), Some(ReedlineEvent::Enter)))
        );
    }

    #[test]
    fn bracketed_multiline_paste_inserts_text_without_submitting() {
        let mut mode = agent_vi_mode();
        let paste = "first line\r\n\tindented second line\rthird line";

        let event = ReedlineRawEvent::try_from(Event::Paste(paste.to_string()))
            .expect("paste event should convert");

        assert_eq!(
            mode.parse_event(event),
            ReedlineEvent::Edit(vec![EditCommand::InsertString(
                "first line\n\tindented second line\nthird line".to_string(),
            )])
        );
    }

    #[test]
    fn completer_picks_up_background_candidate_refreshes() {
        let dynamic_candidates = Arc::new(RwLock::new(Vec::new()));
        let mut completer = AgentCompleter::with_dynamic_candidates(
            vec!["/models".to_string(), "/models gpt-5.6-terra".to_string()],
            Arc::clone(&dynamic_candidates),
        );

        let initial_values = completer
            .complete("/models op", 10)
            .suggestions()
            .iter()
            .map(|suggestion| suggestion.value.clone())
            .collect::<Vec<_>>();
        assert!(!initial_values.contains(&"/models openai:gpt-5.2".to_string()));

        *dynamic_candidates.write().expect("dynamic candidates lock") =
            vec!["/models openai:gpt-5.2".to_string()];

        let refreshed_values = completer
            .complete("/models op", 10)
            .suggestions()
            .iter()
            .map(|suggestion| suggestion.value.clone())
            .collect::<Vec<_>>();
        assert_eq!(
            refreshed_values.first(),
            Some(&"/models openai:gpt-5.2".to_string())
        );
    }
}
