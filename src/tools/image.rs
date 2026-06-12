use async_openai::{Client, config::OpenAIConfig};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::json;

#[derive(Clone, Debug, Deserialize, JsonSchema, Serialize)]
pub struct GenImageArgs {
    pub number: u32,
    pub model: String,
    pub size: String,
    pub prompt: String,
}

pub async fn gen_image(args: GenImageArgs) -> Result<String, String> {
    let api_key =
        std::env::var("OPENAI_API_KEY").map_err(|_| "OPENAI_API_KEY is not set".to_string())?;
    let base_url = std::env::var("AGENT_BASE_URL")
        .or_else(|_| std::env::var("OPENAI_BASE_URL"))
        .unwrap_or_else(|_| "https://api.openai.com/v1".to_string());
    let client = Client::with_config(
        OpenAIConfig::new()
            .with_api_base(base_url)
            .with_api_key(api_key),
    );
    let response: serde_json::Value = client
        .images()
        .generate_byot(json!({
            "n": args.number,
            "model": args.model,
            "size": args.size,
            "prompt": args.prompt,
        }))
        .await
        .map_err(|err| format!("Error creating images: {err}"))?;
    Ok(response.to_string())
}
