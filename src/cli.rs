use clap::Parser;
use crossterm::event::{
    self, Event, KeyCode as CrosstermKeyCode, KeyEventKind, KeyModifiers as CrosstermKeyModifiers,
};
#[cfg(not(unix))]
use crossterm::terminal::{disable_raw_mode, enable_raw_mode};
use reedline::{
    ColumnarMenu, Completer, DefaultPrompt, EditCommand, EditMode, FileBackedHistory, KeyCode,
    KeyModifiers, Keybindings, MenuBuilder, PromptEditMode, Reedline, ReedlineEvent, ReedlineMenu,
    ReedlineRawEvent, Signal, Span, Suggestion, Vi, default_vi_insert_keybindings,
    default_vi_normal_keybindings,
};
use std::error::Error;
use std::io::IsTerminal;
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
    Provider, build_provider, context_window_tokens, effective_model_name,
    format_cost_and_context_line,
};
use crate::session::{Session, SessionStore};
use crate::tools::ToolRegistry;

const COMPLETION_MENU_NAME: &str = "completion_menu";

#[derive(Debug, Parser)]
#[command(name = "agent", about = "Personal Command Line Agent")]
pub struct Args {
    #[arg(short, long, help = "Model id, optionally provider-prefixed")]
    pub model: Option<String>,
    #[arg(long, help = "List available OpenAI and Ollama models")]
    pub list_models: bool,
    #[arg(long, help = "Download and cache LiteLLM pricing data, then exit")]
    pub update_pricing: bool,
    #[arg(short, long, help = "Run one turn and exit")]
    pub single: bool,
    #[arg(
        short = 'c',
        long = "command",
        help = "Run one turn and print only the final response for zsh print -z wrappers"
    )]
    pub command: bool,
    #[arg(
        long,
        num_args = 0..=1,
        default_missing_value = "__LATEST__",
        help = "Resume latest session or the provided session id"
    )]
    pub resume: Option<String>,
    #[arg(long, help = "Start a new session instead of resuming")]
    pub new: bool,
    #[arg(help = "Initial user message")]
    pub query: Vec<String>,
}

pub async fn run() -> Result<(), Box<dyn Error>> {
    let args = Args::parse();
    run_with_args(args).await
}

pub async fn run_with_args(args: Args) -> Result<(), Box<dyn Error>> {
    if args.new && args.resume.is_some() {
        return Err("--new cannot be used with --resume".into());
    }

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
    let display = TerminalDisplay::new();
    let mut model_name = effective_model_name(args.model.as_deref());
    let mut loop_runner: Option<AgentLoop<Box<dyn Provider>>> = None;

    println!("\n[agent]\nmodel: {model_name}");

    let mut session = load_or_create_session(&args, &store)?;
    replay_session(&session, &display);

    let mut first_input = if args.query.is_empty() {
        None
    } else {
        Some(args.query.join(" "))
    };

    loop {
        let user_input = match first_input.take() {
            Some(input) => input,
            None => {
                if args.single && !session.messages.is_empty() {
                    break;
                }
                prompt_for_input(&store, &session, &model_name).await?
            }
        };

        match handle_slash_command(&user_input, &args, &store, &mut session).await? {
            SlashCommandResult::NotCommand => {}
            SlashCommandResult::Handled => {
                if args.single {
                    break;
                }
                continue;
            }
            SlashCommandResult::SwitchModel(new_model) => {
                model_name = new_model;
                loop_runner = None;
                println!("\n[agent]\nmodel: {model_name}");
                if args.single {
                    break;
                }
                continue;
            }
        }

        session.messages.push(AgentMessage::User {
            content: user_input,
        });
        store.save(&session)?;
        display.render_turn_submitted();

        if loop_runner.is_none() {
            loop_runner = Some(build_loop_runner(&model_name)?);
        }
        let cancellation_token = CancellationToken::new();
        let esc_abort = EscAbortWatcher::spawn(cancellation_token.clone());
        let result = tokio::select! {
            result = loop_runner
                .as_ref()
                .expect("loop runner initialized")
                .run_turn_cancellable_with_observer(
                    &session.messages,
                    &cancellation_token,
                    |message| {
                        if let AgentMessage::Tool(result) = message {
                            display.render_tool_result(result);
                        }
                        display.render_new_message(message);
                    },
                    |call| display.render_tool_start(call),
                ) => result,
            _ = cancellation_token.cancelled() => Err(crate::providers::ProviderError::Cancelled),
            _ = tokio::signal::ctrl_c() => {
                cancellation_token.cancel();
                esc_abort.stop().await;
                return Err("cancelled".into());
            }
        };
        esc_abort.stop().await;

        match result {
            Ok(result) => {
                session.messages.extend(result.new_messages);
                session.replace_messages(session.messages.clone());
                store.save(&session)?;
            }
            Err(crate::providers::ProviderError::Cancelled)
                if cancellation_token.is_cancelled() =>
            {
                eprintln!("turn aborted");
            }
            Err(err) => return Err(err.into()),
        }

        if args.single {
            break;
        }
    }

    Ok(())
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
            command: false,
            resume: None,
            new: true,
            query: Vec::new(),
        },
        &store,
    )?;
    session.messages.push(AgentMessage::System {
        content: command_mode_system_prompt(),
    });
    session.messages.push(AgentMessage::User {
        content: args.query.join(" "),
    });

    let model_name = effective_model_name(args.model.as_deref());
    let assistant = build_provider(&model_name)?
        .complete(&session.messages, &[])
        .await?;

    println!("{}", assistant.content.trim());
    Ok(())
}

struct EscAbortWatcher {
    stop: Arc<AtomicBool>,
    handle: Option<tokio::task::JoinHandle<()>>,
}

impl EscAbortWatcher {
    fn spawn(cancellation_token: CancellationToken) -> Self {
        let stop = Arc::new(AtomicBool::new(false));
        let handle = std::io::stdin().is_terminal().then(|| {
            let stop_watcher = Arc::clone(&stop);
            tokio::task::spawn_blocking(move || {
                if InputModeGuard::enable().is_err() {
                    return;
                }
                let _input_mode = InputModeGuard;
                while !stop_watcher.load(Ordering::SeqCst) && !cancellation_token.is_cancelled() {
                    match event::poll(Duration::from_millis(50)) {
                        Ok(true) => match event::read() {
                            Ok(Event::Key(key))
                                if key.code == CrosstermKeyCode::Esc
                                    && key.kind != KeyEventKind::Release
                                    && key.modifiers == CrosstermKeyModifiers::NONE =>
                            {
                                cancellation_token.cancel();
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
        TERMINAL_MODE_ORIGINAL.with(|saved| saved.set(Some(original)));
        Ok(())
    }
}

#[cfg(unix)]
impl Drop for InputModeGuard {
    fn drop(&mut self) {
        TERMINAL_MODE_ORIGINAL.with(|saved| {
            if let Some(original) = saved.take() {
                // SAFETY: original was captured from tcgetattr for stdin in this thread.
                let _ = unsafe { libc::tcsetattr(libc::STDIN_FILENO, libc::TCSANOW, &original) };
            }
        });
    }
}

#[cfg(unix)]
thread_local! {
    static TERMINAL_MODE_ORIGINAL: std::cell::Cell<Option<libc::termios>> = const { std::cell::Cell::new(None) };
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

fn load_or_create_session(args: &Args, store: &SessionStore) -> Result<Session, Box<dyn Error>> {
    if !args.new {
        if let Some(resume) = args.resume.as_deref() {
            let id = if resume == "__LATEST__" {
                None
            } else {
                Some(resume)
            };
            return Ok(store.load(id)?);
        }
        if !args.single
            && let Ok(session) = store.load(None)
        {
            return Ok(session);
        }
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

async fn prompt_for_input(
    store: &SessionStore,
    session: &Session,
    model_name: &str,
) -> Result<String, Box<dyn Error>> {
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
    let dynamic_models = Arc::new(RwLock::new(Vec::new()));
    spawn_model_completion_refresh(Arc::clone(&dynamic_models));
    let completer = Box::new(AgentCompleter::with_dynamic_models(
        completion_candidates(store, &[]),
        dynamic_models,
    ));
    let mut line_editor = Reedline::create()
        .with_history(history)
        .with_completer(completer)
        .with_menu(ReedlineMenu::EngineCompleter(Box::new(
            ColumnarMenu::default().with_name(COMPLETION_MENU_NAME),
        )))
        .with_quick_completions(false)
        .with_partial_completions(false)
        .with_edit_mode(Box::new(agent_vi_mode()));
    let prompt = DefaultPrompt::default();
    println!(
        "\n{}",
        format_cost_and_context_line(&session.messages, model_name)
    );
    match line_editor.read_line(&prompt)? {
        Signal::Success(input) => Ok(input),
        Signal::CtrlD | Signal::CtrlC => Err("Goodbye!".into()),
        _ => Err("unsupported prompt signal".into()),
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum SlashCommandResult {
    NotCommand,
    Handled,
    SwitchModel(String),
}

async fn handle_slash_command(
    input: &str,
    args: &Args,
    store: &SessionStore,
    session: &mut Session,
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
            println!("{}", slash_help());
        }
        "/clear" => {
            let new_session = load_or_create_session(
                &Args {
                    new: true,
                    resume: None,
                    query: Vec::new(),
                    single: args.single,
                    model: args.model.clone(),
                    list_models: false,
                    update_pricing: false,
                    command: false,
                },
                store,
            )?;
            *session = new_session;
            store.save(session)?;
            println!("cleared");
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
            println!("Unknown command: {command}. Try /help");
        }
    }
    Ok(SlashCommandResult::Handled)
}

fn slash_help() -> &'static str {
    "Available commands:\n  /clear\n      Clear the UI and start a new conversation/session.\n  /help\n      Show this help.\n  /models [<model_id>]\n      List models or switch the active model.\n  /pricing refresh\n      Download and cache LiteLLM pricing data.\n  /resume [latest|<session_id>]\n      Resume a saved conversation/session.\n"
}

fn build_loop_runner(model_name: &str) -> Result<AgentLoop<Box<dyn Provider>>, Box<dyn Error>> {
    Ok(AgentLoop::new(
        build_provider(model_name)?,
        ToolRegistry::new(),
        AgentLoopConfig {
            model: model_name.to_string(),
            max_context_tokens: context_window_tokens(model_name),
            ..AgentLoopConfig::default()
        },
    ))
}

async fn list_models() -> Vec<String> {
    let mut models = Vec::new();
    let client = match reqwest::Client::builder()
        .timeout(Duration::from_secs(2))
        .build()
    {
        Ok(client) => client,
        Err(_) => return models,
    };

    if let Ok(api_key) = std::env::var("OPENAI_API_KEY")
        && let Ok(response) = client
            .get("https://api.openai.com/v1/models")
            .bearer_auth(api_key)
            .send()
            .await
        && let Ok(value) = response.json::<serde_json::Value>().await
        && let Some(data) = value.get("data").and_then(|value| value.as_array())
    {
        models.extend(data.iter().filter_map(|model| {
            model
                .get("id")
                .and_then(|id| id.as_str())
                .filter(|id| id.starts_with("gpt-") || id.starts_with('o'))
                .map(|id| format!("openai:{id}"))
        }));
    }

    let ollama_url =
        std::env::var("OLLAMA_URL").unwrap_or_else(|_| "http://localhost:11434".to_string());
    if let Ok(response) = client
        .get(format!("{}/api/tags", ollama_url.trim_end_matches('/')))
        .send()
        .await
        && let Ok(value) = response.json::<serde_json::Value>().await
        && let Some(data) = value.get("models").and_then(|value| value.as_array())
    {
        models.extend(data.iter().filter_map(|model| {
            model
                .get("name")
                .and_then(|name| name.as_str())
                .map(|name| format!("ollama:{name}"))
        }));
    }

    models.sort();
    models.dedup();
    models
}

fn system_prompt() -> String {
    "You are a highly autonomous AI command line agent designed to help users with software engineering tasks, system operations, research, and problem-solving. Be concise, direct, and action-oriented.".to_string()
}

fn command_mode_system_prompt() -> String {
    "Command-buffer mode: produce the text the user wants placed into their zsh prompt. Prefer a single bash/zsh command when the user is asking for a command. Return only the command/text to insert, with no Markdown fences or explanatory prose.".to_string()
}

fn spawn_model_completion_refresh(dynamic_models: Arc<RwLock<Vec<String>>>) {
    tokio::spawn(async move {
        let available_models = list_models().await;
        if !available_models.is_empty()
            && let Ok(mut models) = dynamic_models.write()
        {
            *models = available_models;
        }
    });
}

fn completion_candidates(store: &SessionStore, available_models: &[String]) -> Vec<String> {
    let mut candidates = vec![
        "/clear".to_string(),
        "/help".to_string(),
        "/models".to_string(),
        "/pricing refresh".to_string(),
        "/resume".to_string(),
        "/resume latest".to_string(),
    ];

    candidates.extend(
        model_completion_values(store, available_models)
            .into_iter()
            .map(|model| format!("/models {model}")),
    );

    if let Ok(labels) = store.list_session_labels(80) {
        candidates.extend(labels.into_iter().map(|label| format!("/resume {label}")));
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

pub fn completion_values_for_line_with_models(
    store: &SessionStore,
    line: &str,
    pos: usize,
    available_models: &[String],
) -> Vec<String> {
    AgentCompleter::new(completion_candidates(store, available_models))
        .complete(line, pos)
        .into_iter()
        .map(|suggestion| suggestion.value)
        .collect()
}

#[derive(Clone, Debug)]
struct AgentCompleter {
    candidates: Vec<String>,
    dynamic_models: Option<Arc<RwLock<Vec<String>>>>,
}

impl AgentCompleter {
    fn new(candidates: Vec<String>) -> Self {
        Self::from_parts(candidates, None)
    }

    fn with_dynamic_models(
        candidates: Vec<String>,
        dynamic_models: Arc<RwLock<Vec<String>>>,
    ) -> Self {
        Self::from_parts(candidates, Some(dynamic_models))
    }

    fn from_parts(
        mut candidates: Vec<String>,
        dynamic_models: Option<Arc<RwLock<Vec<String>>>>,
    ) -> Self {
        candidates.sort();
        candidates.dedup();
        Self {
            candidates,
            dynamic_models,
        }
    }

    fn candidates(&self) -> Vec<String> {
        let mut candidates = self.candidates.clone();
        if let Some(dynamic_models) = &self.dynamic_models
            && let Ok(models) = dynamic_models.read()
        {
            candidates.extend(models.iter().map(|model| format!("/models {model}")));
        }
        candidates.sort();
        candidates.dedup();
        candidates
    }
}

impl Completer for AgentCompleter {
    fn complete(&mut self, line: &str, pos: usize) -> Vec<Suggestion> {
        let Some(prefix) = line.get(..pos) else {
            return Vec::new();
        };
        if !prefix.starts_with('/') {
            return Vec::new();
        }

        let span = Span::new(0, pos);
        let candidates = self.candidates();
        let mut matches = candidates
            .iter()
            .filter_map(|candidate| {
                let score = completion_score(candidate, prefix)?;
                Some((score, candidate))
            })
            .collect::<Vec<_>>();
        matches.sort_by(
            |(left_score, left_candidate), (right_score, right_candidate)| {
                left_score
                    .cmp(right_score)
                    .then_with(|| left_candidate.cmp(right_candidate))
            },
        );

        matches
            .into_iter()
            .map(|(_, candidate)| Suggestion {
                value: candidate.clone(),
                span,
                append_whitespace: false,
                ..Default::default()
            })
            .collect()
    }
}

fn agent_vi_mode() -> SlashCompletionVi {
    let mut insert_keybindings = default_vi_insert_keybindings();
    let mut normal_keybindings = default_vi_normal_keybindings();
    add_completion_keybindings(&mut insert_keybindings);
    add_completion_keybindings(&mut normal_keybindings);
    SlashCompletionVi::new(Vi::new(insert_keybindings, normal_keybindings))
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
                ReedlineEvent::Multiple(vec![ReedlineEvent::Enter, ReedlineEvent::Enter])
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
        let event = self.inner.parse_event(event);
        self.handle_event(event)
    }

    fn edit_mode(&self) -> PromptEditMode {
        self.inner.edit_mode()
    }
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
    #[cfg(unix)]
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
        ReedlineRawEvent::try_from(Event::Key(KeyEvent::new(code, KeyModifiers::NONE)))
            .expect("reedline raw event")
    }

    #[test]
    fn enter_after_auto_opened_slash_completion_accepts_and_submits() {
        let mut mode = agent_vi_mode();

        assert!(matches!(
            mode.parse_event(key(KeyCode::Char('/'))),
            ReedlineEvent::Multiple(events)
                if events == vec![
                    ReedlineEvent::Edit(vec![EditCommand::InsertChar('/')]),
                    ReedlineEvent::Menu(COMPLETION_MENU_NAME.to_string()),
                ]
        ));

        assert!(matches!(
            mode.parse_event(key(KeyCode::Enter)),
            ReedlineEvent::Multiple(events)
                if events == vec![ReedlineEvent::Enter, ReedlineEvent::Enter]
        ));
    }

    #[test]
    fn completer_picks_up_background_model_refreshes() {
        let dynamic_models = Arc::new(RwLock::new(Vec::new()));
        let mut completer = AgentCompleter::with_dynamic_models(
            vec!["/models".to_string(), "/models gpt-5.5".to_string()],
            Arc::clone(&dynamic_models),
        );

        let initial_values = completer
            .complete("/models op", 10)
            .into_iter()
            .map(|suggestion| suggestion.value)
            .collect::<Vec<_>>();
        assert!(!initial_values.contains(&"/models openai:gpt-5.2".to_string()));

        *dynamic_models.write().expect("dynamic models lock") = vec!["openai:gpt-5.2".to_string()];

        let refreshed_values = completer
            .complete("/models op", 10)
            .into_iter()
            .map(|suggestion| suggestion.value)
            .collect::<Vec<_>>();
        assert_eq!(
            refreshed_values.first(),
            Some(&"/models openai:gpt-5.2".to_string())
        );
    }
}
