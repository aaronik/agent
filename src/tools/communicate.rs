use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Deserialize, JsonSchema, Serialize)]
pub struct CommunicateArgs {
    pub message: String,
}

pub async fn communicate(args: CommunicateArgs) -> Result<String, String> {
    Ok(args.message)
}
