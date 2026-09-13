use std::time::Duration;

/// Discover the available model IDs shared by `/models` and spawn validation.
pub async fn list_models() -> Vec<String> {
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
        && let Ok(response) = response.error_for_status()
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
        && let Ok(response) = response.error_for_status()
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
