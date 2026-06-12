use std::collections::HashMap;
use std::fmt;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::time::Duration;

use serde::{Deserialize, Deserializer};
use serde_json::Value;
use thiserror::Error;

use crate::agent::Usage;

pub const DEFAULT_LITELLM_PRICING_URL: &str =
    "https://raw.githubusercontent.com/BerriAI/litellm/main/model_prices_and_context_window.json";
pub const PRICING_URL_ENV: &str = "AGENT_PRICING_URL";

const PRICING_DIR: &str = "pricing";
const PRICING_CACHE_FILE: &str = "model_prices_and_context_window.json";

#[derive(Debug, Error)]
pub enum PricingError {
    #[error("pricing request failed: {0}")]
    Request(#[from] reqwest::Error),
    #[error("pricing cache write failed: {0}")]
    Io(#[from] io::Error),
    #[error("pricing data was invalid: {0}")]
    InvalidData(String),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PricingRefreshReport {
    pub source_url: String,
    pub cache_path: PathBuf,
    pub model_count: usize,
    pub priced_model_count: usize,
}

impl fmt::Display for PricingRefreshReport {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "Pricing updated: {}/{} priced models cached at {}",
            self.priced_model_count,
            self.model_count,
            self.cache_path.display()
        )
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(transparent)]
pub struct PricingMap {
    models: HashMap<String, ModelPricing>,
}

impl PricingMap {
    pub fn model_count(&self) -> usize {
        self.models.len()
    }

    pub fn priced_model_count(&self) -> usize {
        self.models
            .values()
            .filter(|pricing| pricing.has_text_pricing())
            .count()
    }

    fn pricing_for_model(&self, raw_model: &str) -> Option<&ModelPricing> {
        let candidates = model_candidates(raw_model);
        for candidate in &candidates {
            if let Some(pricing) = self.models.get(candidate) {
                return Some(pricing);
            }
        }

        self.models.values().find(|pricing| {
            pricing
                .aliases
                .iter()
                .any(|alias| candidates.iter().any(|candidate| candidate == alias))
        })
    }
}

#[derive(Clone, Debug, Default, Deserialize, PartialEq)]
struct ModelPricing {
    #[serde(default, deserialize_with = "deserialize_optional_f64")]
    input_cost_per_token: Option<f64>,
    #[serde(default, deserialize_with = "deserialize_optional_f64")]
    output_cost_per_token: Option<f64>,
    #[serde(default)]
    aliases: Vec<String>,
}

impl ModelPricing {
    fn has_text_pricing(&self) -> bool {
        self.input_cost_per_token.is_some() || self.output_cost_per_token.is_some()
    }

    fn cost_usd(&self, usage: &Usage) -> Option<f64> {
        if !self.has_text_pricing() {
            return None;
        }

        Some(
            (usage.input_tokens as f64 * self.input_cost_per_token.unwrap_or(0.0))
                + (usage.output_tokens as f64 * self.output_cost_per_token.unwrap_or(0.0)),
        )
    }
}

pub async fn refresh_pricing_cache(root: &Path) -> Result<PricingRefreshReport, PricingError> {
    let source_url =
        std::env::var(PRICING_URL_ENV).unwrap_or_else(|_| DEFAULT_LITELLM_PRICING_URL.to_string());
    refresh_pricing_cache_from_url(root, &source_url).await
}

pub async fn refresh_pricing_cache_from_url(
    root: &Path,
    source_url: &str,
) -> Result<PricingRefreshReport, PricingError> {
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(20))
        .build()?;
    let response = client.get(source_url).send().await?;
    let status = response.status();
    if !status.is_success() {
        return Err(PricingError::InvalidData(format!(
            "source returned HTTP {}",
            status.as_u16()
        )));
    }

    let payload = response.text().await?;
    let pricing_map = parse_pricing_map(&payload)?;
    let priced_model_count = pricing_map.priced_model_count();
    if priced_model_count == 0 {
        return Err(PricingError::InvalidData(
            "cost map did not contain text-token pricing".to_string(),
        ));
    }

    let cache_path = pricing_cache_path(root);
    if let Some(parent) = cache_path.parent() {
        fs::create_dir_all(parent)?;
    }
    let tmp_path = cache_path.with_extension("json.tmp");
    fs::write(&tmp_path, format!("{payload}\n"))?;
    fs::rename(&tmp_path, &cache_path)?;

    Ok(PricingRefreshReport {
        source_url: source_url.to_string(),
        cache_path,
        model_count: pricing_map.model_count(),
        priced_model_count,
    })
}

pub fn pricing_cache_path(root: &Path) -> PathBuf {
    root.join(PRICING_DIR).join(PRICING_CACHE_FILE)
}

pub fn cost_from_cached_litellm_pricing(raw_model: &str, usage: &Usage) -> Option<f64> {
    let root = dirs::home_dir()?.join(".agent-rs");
    cost_from_cache_at(&root, raw_model, usage)
}

pub fn cost_from_cache_at(root: &Path, raw_model: &str, usage: &Usage) -> Option<f64> {
    let payload = fs::read_to_string(pricing_cache_path(root)).ok()?;
    let pricing_map = parse_pricing_map(&payload).ok()?;
    cost_from_pricing_map(&pricing_map, raw_model, usage)
}

pub fn cost_from_pricing_map(
    pricing_map: &PricingMap,
    raw_model: &str,
    usage: &Usage,
) -> Option<f64> {
    pricing_map.pricing_for_model(raw_model)?.cost_usd(usage)
}

pub fn parse_pricing_map(payload: &str) -> Result<PricingMap, PricingError> {
    let pricing_map: PricingMap =
        serde_json::from_str(payload).map_err(|err| PricingError::InvalidData(err.to_string()))?;
    if pricing_map.model_count() == 0 {
        return Err(PricingError::InvalidData("cost map was empty".to_string()));
    }
    Ok(pricing_map)
}

fn model_candidates(raw_model: &str) -> Vec<String> {
    let mut candidates = vec![raw_model.to_string()];
    if let Some((provider, model)) = raw_model.split_once(':') {
        candidates.push(model.to_string());
        candidates.push(format!("{provider}/{model}"));
    }
    candidates.dedup();
    candidates
}

fn deserialize_optional_f64<'de, D>(deserializer: D) -> Result<Option<f64>, D::Error>
where
    D: Deserializer<'de>,
{
    let value = Option::<Value>::deserialize(deserializer)?;
    match value {
        None | Some(Value::Null) => Ok(None),
        Some(Value::Number(number)) => number
            .as_f64()
            .ok_or_else(|| serde::de::Error::custom("number could not be represented as f64"))
            .map(Some),
        Some(Value::String(raw)) if raw.trim().is_empty() => Ok(None),
        Some(Value::String(raw)) => raw
            .parse::<f64>()
            .map(Some)
            .map_err(serde::de::Error::custom),
        Some(other) => Err(serde::de::Error::custom(format!(
            "expected number, string, or null; got {other}"
        ))),
    }
}
