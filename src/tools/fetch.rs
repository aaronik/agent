use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::time::Duration;

const MAX_RESPONSE_LENGTH: usize = 1_000_000;
const FETCH_TIMEOUT_SECONDS: u64 = 30;

#[derive(Clone, Debug, Deserialize, JsonSchema, Serialize)]
pub struct FetchArgs {
    pub url: String,
}

pub async fn fetch(args: FetchArgs) -> Result<String, String> {
    let jina_url = if is_jina_reader_url(&args.url) {
        args.url.clone()
    } else {
        format!("https://r.jina.ai/{}", args.url)
    };

    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(FETCH_TIMEOUT_SECONDS))
        .build()
        .map_err(|err| format!("Error fetching URL {}: {err}", args.url))?;
    let response = client
        .get(&jina_url)
        .send()
        .await
        .map_err(|err| format!("Error fetching URL {}: {err}", args.url))?;
    let status = response.status();
    let text = response
        .text()
        .await
        .map_err(|err| format!("Error fetching URL {}: {err}", args.url))?;
    if !status.is_success() {
        return Err(format!(
            "Error fetching URL {}: HTTP {status}\n{text}",
            args.url
        ));
    }

    if text.len() > MAX_RESPONSE_LENGTH {
        let truncated = text.chars().take(MAX_RESPONSE_LENGTH).collect::<String>();
        Ok(format!(
            "{}\n\n[Content truncated due to size limitations]",
            truncated
        ))
    } else {
        Ok(format!("[URL]: {}\n\n{text}", args.url))
    }
}

fn is_jina_reader_url(url: &str) -> bool {
    url.starts_with("https://r.jina.ai/") || url.starts_with("http://r.jina.ai/")
}
