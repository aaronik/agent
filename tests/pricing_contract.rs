use agent_rs::agent::Usage;
use agent_rs::pricing::{
    cost_from_cache_at, cost_from_pricing_map, parse_pricing_map, pricing_cache_path,
    refresh_pricing_cache_from_url,
};
use serde_json::json;
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

#[test]
fn litellm_pricing_map_prices_prefixed_models_and_aliases() {
    let pricing_map = parse_pricing_map(
        r#"{
            "gpt-5.2": {
                "input_cost_per_token": 0.000001,
                "output_cost_per_token": 0.000003,
                "aliases": ["gpt-5.2-latest"]
            }
        }"#,
    )
    .expect("pricing map");
    let usage = Usage {
        input_tokens: 1_000,
        output_tokens: 500,
        raw: None,
    };

    assert_eq!(
        cost_from_pricing_map(&pricing_map, "openai:gpt-5.2", &usage),
        Some(0.0025)
    );
    assert_eq!(
        cost_from_pricing_map(&pricing_map, "gpt-5.2-latest", &usage),
        Some(0.0025)
    );
}

#[test]
fn descriptive_token_limits_do_not_disable_cached_pricing_or_context() {
    let temp = tempfile::tempdir().expect("temp dir");
    let root = temp.path().join(".agent");
    let cache_path = pricing_cache_path(&root);
    std::fs::create_dir_all(cache_path.parent().unwrap()).unwrap();
    std::fs::write(
        &cache_path,
        r#"{
            "sample_spec": {"max_input_tokens": "max input tokens", "max_output_tokens": "max output tokens", "max_tokens": "legacy parameter"},
            "gpt-6-sol": {"input_cost_per_token": 0.000002, "output_cost_per_token": 0.00001, "max_input_tokens": 922000, "max_output_tokens": 128000}
        }"#,
    )
    .unwrap();
    let usage = Usage {
        input_tokens: 33_000,
        output_tokens: 100,
        raw: None,
    };
    assert_eq!(
        cost_from_cache_at(&root, "openai:gpt-6-sol", &usage),
        Some(0.067)
    );
    assert_eq!(
        agent_rs::pricing::context_window_from_cache_at(&root, "openai:gpt-6-sol"),
        Some(1_050_000)
    );
}

#[test]
fn cached_pricing_file_prices_sessions_without_network() {
    let temp = tempfile::tempdir().expect("temp dir");
    let root = temp.path().join(".agent");
    let cache_path = pricing_cache_path(&root);
    std::fs::create_dir_all(cache_path.parent().expect("cache parent")).expect("cache dir");
    std::fs::write(
        &cache_path,
        r#"{
            "openai/gpt-5.2": {
                "input_cost_per_token": "0.000001",
                "output_cost_per_token": "0.000003"
            }
        }"#,
    )
    .expect("write pricing cache");
    let usage = Usage {
        input_tokens: 1_000,
        output_tokens: 500,
        raw: None,
    };

    assert_eq!(
        cost_from_cache_at(&root, "openai:gpt-5.2", &usage),
        Some(0.0025)
    );
}

#[tokio::test]
async fn refresh_pricing_cache_writes_validated_litellm_map() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/pricing.json"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "gpt-5.2": {
                "input_cost_per_token": 0.000001,
                "output_cost_per_token": 0.000003
            },
            "text-embedding-3-large": {
                "input_cost_per_token": 0.00000013
            }
        })))
        .mount(&server)
        .await;
    let temp = tempfile::tempdir().expect("temp dir");
    let root = temp.path().join(".agent");

    let report = refresh_pricing_cache_from_url(&root, &format!("{}/pricing.json", server.uri()))
        .await
        .expect("refresh pricing");

    assert_eq!(report.model_count, 2);
    assert_eq!(report.priced_model_count, 2);
    assert!(pricing_cache_path(&root).exists());
}
