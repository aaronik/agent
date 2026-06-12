use clap::Parser;
use reedline::{
    ColumnarMenu, Completer, DefaultPrompt, EditCommand, FileBackedHistory, KeyCode, KeyModifiers,
    Keybindings, MenuBuilder, Reedline, ReedlineEvent, ReedlineMenu, Signal, Span, Suggestion, Vi,
    default_vi_insert_keybindings, default_vi_normal_keybindings,
};
use std::error::Error;
use std::io::IsTerminal;
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
                prompt_for_input(&store, &session, &model_name)?
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

        if loop_runner.is_none() {
            loop_runner = Some(build_loop_runner(&model_name)?);
        }
        let cancellation_token = CancellationToken::new();
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
                ) => result?,
            _ = tokio::signal::ctrl_c() => {
                cancellation_token.cancel();
                return Err("cancelled".into());
            }
        };

        session.messages.extend(result.new_messages);
        session.replace_messages(session.messages.clone());
        store.save(&session)?;

        if args.single {
            break;
        }
    }

    Ok(())
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
    for message in &session.messages {
        display.render_new_message(message);
        if let AgentMessage::Tool(result) = message {
            display.render_tool_result(result);
        }
    }
}

fn prompt_for_input(
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
    let completer = Box::new(AgentCompleter::new(completion_candidates(store)));
    let mut line_editor = Reedline::create()
        .with_history(history)
        .with_completer(completer)
        .with_menu(ReedlineMenu::EngineCompleter(Box::new(
            ColumnarMenu::default().with_name(COMPLETION_MENU_NAME),
        )))
        .with_quick_completions(true)
        .with_partial_completions(true)
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

fn completion_candidates(store: &SessionStore) -> Vec<String> {
    let mut candidates = vec![
        "/clear".to_string(),
        "/help".to_string(),
        "/models".to_string(),
        "/models mock".to_string(),
        "/models openai:gpt-5.5".to_string(),
        "/pricing refresh".to_string(),
        "/resume".to_string(),
        "/resume latest".to_string(),
    ];
    if let Ok(labels) = store.list_session_labels(80) {
        candidates.extend(labels.into_iter().map(|label| format!("/resume {label}")));
    }
    candidates
}

pub fn completion_values_for_line(store: &SessionStore, line: &str, pos: usize) -> Vec<String> {
    AgentCompleter::new(completion_candidates(store))
        .complete(line, pos)
        .into_iter()
        .map(|suggestion| suggestion.value)
        .collect()
}

#[derive(Clone, Debug)]
struct AgentCompleter {
    candidates: Vec<String>,
}

impl AgentCompleter {
    fn new(mut candidates: Vec<String>) -> Self {
        candidates.sort();
        candidates.dedup();
        Self { candidates }
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
        let mut matches = self
            .candidates
            .iter()
            .filter_map(|candidate| {
                let score = completion_score(candidate, prefix)?;
                if candidate.len() <= prefix.len() && candidate == prefix {
                    return None;
                }
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

fn agent_vi_mode() -> Vi {
    let mut insert_keybindings = default_vi_insert_keybindings();
    let mut normal_keybindings = default_vi_normal_keybindings();
    add_completion_keybindings(&mut insert_keybindings);
    add_completion_keybindings(&mut normal_keybindings);
    Vi::new(insert_keybindings, normal_keybindings)
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
