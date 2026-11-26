use std::cell::LazyCell;
use std::env;

use crate::agent::{AgentMessage, Usage};
use crate::pricing::{
    PricingMap, cache_token_counts, context_window_from_pricing_map, cost_from_pricing_map,
    load_cached_pricing_map, service_tier,
};
use crate::providers::{
    MockProvider, OpenAiCompatibleProvider, Provider, ProviderConfig, ProviderFlavor,
};

pub const DEFAULT_MODEL: &str = "gpt-6.2-sol";
pub const DEFAULT_CONTEXT_TOKENS: usize = 16_384;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ParsedModelId {
    pub provider: String,
    pub model: String,
}

pub fn effective_model_name(model_override: Option<&str>) -> String {
    model_override
        .map(ToOwned::to_owned)
        .filter(|model| !model.is_empty())
        .or_else(|| non_empty_env("AGENT_MODEL"))
        .or_else(|| non_empty_env("OLLAMA_MODEL").map(|model| format!("ollama:{model}")))
        .or_else(|| non_empty_env("OPENAI_MODEL"))
        .unwrap_or_else(|| DEFAULT_MODEL.to_string())
}

fn non_empty_env(name: &str) -> Option<String> {
    env::var(name).ok().filter(|value| !value.is_empty())
}

pub fn parse_model_id(raw: &str) -> ParsedModelId {
    if raw == "mock" {
        return ParsedModelId {
            provider: "mock".to_string(),
            model: "mock".to_string(),
        };
    }

    match raw.split_once(':') {
        Some((provider, model)) => ParsedModelId {
            provider: provider.to_string(),
            model: model.to_string(),
        },
        None => ParsedModelId {
            provider: "openai".to_string(),
            model: raw.to_string(),
        },
    }
}

pub fn build_provider(raw_model: &str) -> Result<Box<dyn Provider>, Box<dyn std::error::Error>> {
    build_provider_with_fast_mode(raw_model, false)
}

pub(crate) fn build_provider_with_fast_mode(
    raw_model: &str,
    fast: bool,
) -> Result<Box<dyn Provider>, Box<dyn std::error::Error>> {
    let parsed = parse_model_id(raw_model);

    if parsed.provider == "mock" {
        return Ok(Box::new(MockProvider::default()));
    }

    let config = match parsed.provider.as_str() {
        "ollama" => ProviderConfig {
            provider: parsed.provider,
            model: parsed.model,
            base_url: format!(
                "{}/v1",
                env::var("OLLAMA_URL")
                    .unwrap_or_else(|_| "http://localhost:11434".to_string())
                    .trim_end_matches('/')
            ),
            api_key: "dummy".to_string(),
            flavor: ProviderFlavor::OpenAiChat,
        },
        "openai" => ProviderConfig {
            provider: parsed.provider,
            model: parsed.model,
            base_url: env::var("AGENT_BASE_URL")
                .or_else(|_| env::var("OPENAI_BASE_URL"))
                .unwrap_or_else(|_| "https://api.openai.com/v1".to_string()),
            api_key: env::var("OPENAI_API_KEY")?,
            flavor: ProviderFlavor::OpenAiResponses,
        },
        other => return Err(format!("unknown provider: {other}").into()),
    };

    Ok(Box::new(
        OpenAiCompatibleProvider::new(config).with_fast_mode(fast),
    ))
}

// One on-demand load (including a cache miss) per calculation. The next
// calculation uses a new snapshot so it sees refreshed prices.
type PricingSnapshot<F> = LazyCell<Option<PricingMap>, F>;

pub fn context_window_tokens(raw_model: &str) -> usize {
    context_window_tokens_with_pricing(raw_model, &PricingSnapshot::new(load_cached_pricing_map))
}

fn context_window_tokens_with_pricing(
    raw_model: &str,
    pricing: &PricingSnapshot<impl FnOnce() -> Option<PricingMap>>,
) -> usize {
    let parsed = parse_model_id(raw_model);
    let model = parsed.model.to_lowercase();

    if parsed.provider == "ollama" || parsed.provider == "mock" {
        return DEFAULT_CONTEXT_TOKENS;
    }
    if model.contains("gpt-5.6")
        || model.contains("gpt5.6")
        || model.contains("gpt-5.5")
        || model.contains("gpt5.5")
        || model.contains("gpt-5.2")
        || model.contains("gpt5.2")
    {
        return 400_000;
    }
    if model.contains("gpt-4o")
        || model.contains("gpt-4-turbo")
        || model.contains("gpt-4-1106")
        || model.contains("gpt-4-0125")
        || model.starts_with("o1")
    {
        return 128_000;
    }
    if model == "gpt-4" || model.contains("gpt-4-") {
        return 8_192;
    }
    if model.contains("gpt-3.5-turbo") {
        return 16_384;
    }
    if let Some(window) = pricing
        .as_ref()
        .and_then(|map| context_window_from_pricing_map(map, raw_model))
    {
        return window;
    }
    if model.contains("gpt-6") || model.contains("gpt6") {
        return 1_050_000;
    }

    DEFAULT_CONTEXT_TOKENS
}

pub fn format_cost_and_context_line(
    messages: &[AgentMessage],
    raw_model: &str,
    allow_git_writes: bool,
) -> String {
    format_cost_and_context_line_with_usages(messages, raw_model, allow_git_writes, &[])
}

pub fn format_cost_and_context_line_with_usages(
    messages: &[AgentMessage],
    raw_model: &str,
    allow_git_writes: bool,
    name_usages: &[crate::agent::SubagentUsage],
) -> String {
    format_cost_and_context_line_with_all_usages(
        messages,
        raw_model,
        allow_git_writes,
        name_usages,
        &[],
    )
}

pub fn format_cost_and_context_line_with_all_usages(
    messages: &[AgentMessage],
    raw_model: &str,
    allow_git_writes: bool,
    name_usages: &[crate::agent::SubagentUsage],
    compacted_usages: &[crate::agent::SubagentUsage],
) -> String {
    format_cost_and_context_line_with_pricing(
        messages,
        raw_model,
        allow_git_writes,
        name_usages,
        compacted_usages,
        load_cached_pricing_map,
    )
}

fn format_cost_and_context_line_with_pricing(
    messages: &[AgentMessage],
    raw_model: &str,
    allow_git_writes: bool,
    name_usages: &[crate::agent::SubagentUsage],
    compacted_usages: &[crate::agent::SubagentUsage],
    load_pricing: impl FnOnce() -> Option<PricingMap>,
) -> String {
    let pricing = PricingSnapshot::new(load_pricing);
    let max_context_tokens = context_window_tokens_with_pricing(raw_model, &pricing);
    let used = crate::agent::count_tokens(messages, raw_model);
    let remaining = max_context_tokens.saturating_sub(used);
    let pct = remaining
        .saturating_mul(100)
        .checked_div(max_context_tokens)
        .unwrap_or(0);

    let cached_tokens: u64 = messages
        .iter()
        .map(|message| match message {
            AgentMessage::Assistant(assistant) => assistant
                .usage
                .as_ref()
                .map(|usage| cache_token_counts(usage).0)
                .unwrap_or(0),
            AgentMessage::Tool(result) => result
                .subagent_usages
                .iter()
                .map(|subagent| cache_token_counts(&subagent.usage).0)
                .sum(),
            _ => 0,
        })
        .sum();
    let mut known_cost = 0.0;
    let mut unknown_cost = false;
    for (usage, model) in session_usages(messages, raw_model).chain(
        name_usages
            .iter()
            .chain(compacted_usages)
            .map(|entry| (&entry.usage, entry.model.as_str())),
    ) {
        match usage_cost_usd(usage, model, &pricing) {
            Some(cost) => known_cost += cost,
            None => unknown_cost = true,
        }
    }
    let cost = match (normalized_cost(known_cost), unknown_cost) {
        (0.0, true) => "unknown".to_string(),
        (known, true) => format!("${known:.4} + unknown"),
        (known, false) => format!("${known:.4}"),
    };
    format!(
        "Cost: {cost}   Context {}% ({} C/{} I/{} T)   Model: {}{}",
        pct,
        format_number(cached_tokens as usize),
        format_number(used),
        format_number(max_context_tokens),
        raw_model,
        if allow_git_writes {
            "   git:allowed"
        } else {
            ""
        }
    )
}

fn normalized_cost(cost: f64) -> f64 {
    if cost.abs() < 0.00005 { 0.0 } else { cost }
}

/// Sum the known costs. The footer separately indicates unpriced usage.
pub fn total_session_cost_usd(messages: &[AgentMessage], raw_model: &str) -> f64 {
    total_session_cost_usd_with_pricing(messages, raw_model, load_cached_pricing_map)
}

fn total_session_cost_usd_with_pricing(
    messages: &[AgentMessage],
    raw_model: &str,
    load_pricing: impl FnOnce() -> Option<PricingMap>,
) -> f64 {
    let pricing = PricingSnapshot::new(load_pricing);
    total_session_cost_usd_with(messages, raw_model, |usage, model| {
        usage_cost_usd(usage, model, &pricing).unwrap_or(0.0)
    })
}

pub fn total_session_cost_usd_with<F>(
    messages: &[AgentMessage],
    fallback_model: &str,
    mut cost_for_usage: F,
) -> f64
where
    F: FnMut(&Usage, &str) -> f64,
{
    session_usages(messages, fallback_model)
        .map(|(usage, model)| cost_for_usage(usage, model))
        .sum()
}

fn session_usages<'a>(
    messages: &'a [AgentMessage],
    fallback_model: &'a str,
) -> impl Iterator<Item = (&'a Usage, &'a str)> {
    messages.iter().flat_map(move |message| {
        let (assistant_usage, subagent_usages) = match message {
            AgentMessage::Assistant(assistant) => (
                assistant
                    .usage
                    .as_ref()
                    .map(|usage| (usage, assistant.model.as_deref().unwrap_or(fallback_model))),
                &[][..],
            ),
            AgentMessage::Tool(result) => (None, result.subagent_usages.as_slice()),
            _ => (None, &[][..]),
        };
        assistant_usage.into_iter().chain(
            subagent_usages
                .iter()
                .map(|entry| (&entry.usage, entry.model.as_str())),
        )
    })
}

fn usage_cost_usd(
    usage: &Usage,
    raw_model: &str,
    pricing: &PricingSnapshot<impl FnOnce() -> Option<PricingMap>>,
) -> Option<f64> {
    if let Some(raw_cost) = usage.raw.as_ref().and_then(cost_from_raw_usage) {
        return Some(raw_cost);
    }

    let parsed = parse_model_id(raw_model);
    if parsed.provider == "ollama" {
        return Some(0.0);
    }

    if let Some(cached_cost) = pricing
        .as_ref()
        .and_then(|map| cost_from_pricing_map(map, raw_model, usage))
    {
        return Some(cached_cost);
    }

    if !matches!(
        service_tier(usage),
        None | Some("default" | "standard" | "auto")
    ) {
        return None;
    }
    let Some((input_per_million, output_per_million)) = pricing_per_million() else {
        return Some(0.0);
    };
    Some(
        ((usage.input_tokens as f64 * input_per_million)
            + (usage.output_tokens as f64 * output_per_million))
            / 1_000_000.0,
    )
}

fn cost_from_raw_usage(raw: &serde_json::Value) -> Option<f64> {
    raw.get("total_cost")
        .or_else(|| raw.get("cost"))
        .and_then(serde_json::Value::as_f64)
        .or_else(|| {
            let input = raw
                .get("prompt_cost")
                .or_else(|| raw.get("input_cost"))
                .and_then(serde_json::Value::as_f64)?;
            let output = raw
                .get("completion_cost")
                .or_else(|| raw.get("output_cost"))
                .and_then(serde_json::Value::as_f64)?;
            Some(input + output)
        })
}

fn pricing_per_million() -> Option<(f64, f64)> {
    let input = env::var("AGENT_INPUT_COST_PER_MILLION")
        .ok()?
        .parse::<f64>()
        .ok()?;
    let output = env::var("AGENT_OUTPUT_COST_PER_MILLION")
        .ok()?
        .parse::<f64>()
        .ok()?;
    Some((input, output))
}

fn format_number(value: usize) -> String {
    let digits = value.to_string();
    let mut out = String::new();
    for (index, ch) in digits.chars().rev().enumerate() {
        if index > 0 && index % 3 == 0 {
            out.push(',');
        }
        out.push(ch);
    }
    out.chars().rev().collect()
}

#[cfg(test)]
mod pricing_tests {
    use super::*;
    use crate::agent::{AssistantMessage, SubagentUsage, ToolResult, ToolStatus};
    use crate::pricing::{load_pricing_map_at, pricing_cache_path};
    use std::{cell::Cell, fs};

    fn usage(input_tokens: u64) -> Usage {
        Usage {
            input_tokens,
            output_tokens: 0,
            raw: None,
        }
    }

    fn assistant(usage: Usage) -> AgentMessage {
        AgentMessage::Assistant(AssistantMessage {
            content: String::new(),
            tool_calls: Vec::new(),
            usage: Some(usage),
            model: None,
            metadata: Default::default(),
        })
    }

    fn subagent(model: &str, input_tokens: u64) -> SubagentUsage {
        SubagentUsage {
            model: model.to_string(),
            usage: usage(input_tokens),
        }
    }

    #[test]
    fn pricing_loads_once_for_footer_and_session_total_and_sees_cache_replacement() {
        let root = tempfile::tempdir().unwrap();
        let path = pricing_cache_path(root.path());
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        let write_prices = |context, input_rate| {
            fs::write(
                &path,
                serde_json::json!({
                    "main": {"max_input_tokens": context, "input_cost_per_token": input_rate},
                    "worker": {"input_cost_per_token": 0.25, "aliases": ["worker-alias"]},
                })
                .to_string(),
            )
            .unwrap();
        };
        let loads = Cell::new(0);
        let load = || {
            loads.set(loads.get() + 1);
            load_pricing_map_at(root.path())
        };
        let model = "openai:main";
        let messages = [
            assistant(usage(1)),
            AgentMessage::Tool(ToolResult {
                tool_call_id: "call".into(),
                name: "spawn".into(),
                status: ToolStatus::Success,
                content: String::new(),
                elapsed_ms: None,
                subagent_usages: vec![subagent("openai:worker-alias", 1)],
                images: Vec::new(),
            }),
        ];
        let name_usages = [subagent(model, 4)];
        let compacted_usages = [subagent("openai:worker-alias", 4)];
        let footer = || {
            format_cost_and_context_line_with_pricing(
                &messages,
                model,
                false,
                &name_usages,
                &compacted_usages,
                load,
            )
        };

        write_prices(48_000, 0.125);
        let line = footer();
        assert!(line.contains("Cost: $1.8750"), "{line}");
        assert!(line.contains("/48,000 T)"), "{line}");
        assert_eq!(loads.replace(0), 1, "footer pricing loads");

        assert_eq!(
            total_session_cost_usd_with_pricing(&messages, model, load),
            0.375
        );
        assert_eq!(loads.replace(0), 1, "session total pricing loads");

        write_prices(64_000, 0.25);
        let line = footer();
        assert!(line.contains("Cost: $2.5000"), "{line}");
        assert!(line.contains("/64,000 T)"), "{line}");
        assert_eq!(loads.get(), 1, "footer pricing loads after replacement");
    }

    #[test]
    fn pricing_does_not_retry_a_missing_cache_within_a_calculation() {
        let loads = Cell::new(0);
        let load = || {
            loads.set(loads.get() + 1);
            None
        };
        let model = "openai:gpt-6-test";
        let mut priority = usage(1);
        priority.raw = Some(serde_json::json!({"service_tier": "priority"}));
        let messages = [assistant(priority.clone()), assistant(priority)];

        let line =
            format_cost_and_context_line_with_pricing(&messages, model, false, &[], &[], load);
        assert!(line.contains("Cost: unknown"), "{line}");
        assert!(line.contains("/1,050,000 T)"), "{line}");
        assert_eq!(loads.replace(0), 1, "footer pricing loads");
        assert_eq!(
            total_session_cost_usd_with_pricing(&messages, model, load),
            0.0
        );
        assert_eq!(loads.get(), 1, "session total pricing loads");
    }

    #[test]
    fn pricing_is_not_loaded_for_raw_cost_or_free_ollama_with_builtin_context() {
        let load = || panic!("pricing was not needed");
        let model = "ollama:local";
        let mut reported = usage(1);
        reported.raw = Some(serde_json::json!({"total_cost": 0.125}));
        let messages = [assistant(reported), assistant(usage(1))];

        let line =
            format_cost_and_context_line_with_pricing(&messages, model, false, &[], &[], load);
        assert!(line.contains("Cost: $0.1250"), "{line}");
        assert!(line.contains("/16,384 T)"), "{line}");
        assert_eq!(
            total_session_cost_usd_with_pricing(&messages, model, load),
            0.125
        );
        assert_eq!(
            total_session_cost_usd_with_pricing(&[], "openai:main", load),
            0.0
        );
    }
}
