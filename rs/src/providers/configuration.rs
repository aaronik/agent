use std::env;

use crate::agent::{AgentMessage, Usage};
use crate::pricing::cost_from_cached_litellm_pricing;
use crate::providers::{
    MockProvider, OpenAiCompatibleProvider, Provider, ProviderConfig, ProviderFlavor,
};

pub const DEFAULT_MODEL: &str = "gpt-5.5";
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

    Ok(Box::new(OpenAiCompatibleProvider::new(config)))
}

pub fn context_window_tokens(raw_model: &str) -> usize {
    let parsed = parse_model_id(raw_model);
    let model = parsed.model.to_lowercase();

    if parsed.provider == "ollama" || parsed.provider == "mock" {
        return DEFAULT_CONTEXT_TOKENS;
    }
    if model.contains("gpt-5.5")
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

    DEFAULT_CONTEXT_TOKENS
}

pub fn format_cost_and_context_line(messages: &[AgentMessage], raw_model: &str) -> String {
    let max_context_tokens = context_window_tokens(raw_model);
    let used = crate::agent::count_tokens(messages, raw_model);
    let remaining = max_context_tokens.saturating_sub(used);
    let pct = remaining
        .saturating_mul(100)
        .checked_div(max_context_tokens)
        .unwrap_or(0);

    format!(
        "Cost: ${:.4}   Context {}% ({}/{} tokens)   Model: {}",
        normalized_cost(total_session_cost_usd(messages, raw_model)),
        pct,
        format_number(remaining),
        format_number(max_context_tokens),
        raw_model
    )
}

fn normalized_cost(cost: f64) -> f64 {
    if cost.abs() < 0.00005 { 0.0 } else { cost }
}

pub fn total_session_cost_usd(messages: &[AgentMessage], raw_model: &str) -> f64 {
    messages
        .iter()
        .filter_map(|message| match message {
            AgentMessage::Assistant(assistant) => assistant.usage.as_ref(),
            _ => None,
        })
        .map(|usage| usage_cost_usd(usage, raw_model))
        .sum()
}

fn usage_cost_usd(usage: &Usage, raw_model: &str) -> f64 {
    if let Some(raw_cost) = usage.raw.as_ref().and_then(cost_from_raw_usage) {
        return raw_cost;
    }

    let parsed = parse_model_id(raw_model);
    if parsed.provider == "ollama" {
        return 0.0;
    }

    if let Some(cached_cost) = cost_from_cached_litellm_pricing(raw_model, usage) {
        return cached_cost;
    }

    let Some((input_per_million, output_per_million)) = pricing_per_million() else {
        return 0.0;
    };
    ((usage.input_tokens as f64 * input_per_million)
        + (usage.output_tokens as f64 * output_per_million))
        / 1_000_000.0
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
