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

#[test]
fn openai_cache_usage_uses_litellm_read_and_write_rates() {
    let pricing_map = parse_pricing_map(
        r#"{"gpt-test":{"input_cost_per_token":0.000001,"cache_read_input_token_cost":0.0000001,"cache_creation_input_token_cost":0.00000125,"output_cost_per_token":0.000003}}"#,
    )
    .unwrap();
    let usage = Usage {
        input_tokens: 2_000,
        output_tokens: 100,
        raw: Some(json!({"input_tokens_details":{"cached_tokens":800,"cache_write_tokens":600}})),
    };
    let cost = cost_from_pricing_map(&pricing_map, "openai:gpt-test", &usage).unwrap();
    assert!((cost - 0.00173).abs() < 1e-12, "{cost}");

    let chat_usage = Usage {
        raw: Some(json!({"prompt_tokens_details":{"cached_tokens":800}})),
        ..usage.clone()
    };
    let cost = cost_from_pricing_map(&pricing_map, "openai:gpt-test", &chat_usage).unwrap();
    assert!((cost - 0.00158).abs() < 1e-12, "{cost}");
}

#[test]
fn missing_cache_price_or_invalid_counts_never_invent_discounts() {
    let pricing_map = parse_pricing_map(
        r#"{"gpt-test":{"input_cost_per_token":0.000001,"output_cost_per_token":0.000003}}"#,
    )
    .unwrap();
    for raw in [
        json!({"input_tokens_details":{"cached_tokens":800}}),
        json!({"input_tokens_details":{"cached_tokens":2001,"cache_write_tokens":10}}),
    ] {
        let usage = Usage {
            input_tokens: 2_000,
            output_tokens: 100,
            raw: Some(raw),
        };
        assert_eq!(
            cost_from_pricing_map(&pricing_map, "openai:gpt-test", &usage),
            Some(0.0023)
        );
    }
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

#[test]
fn model_price_table_resolves_provider_prefixes_aliases_and_small_rates() {
    let home = tempfile::tempdir().unwrap();
    let cache = pricing_cache_path(home.path());
    std::fs::create_dir_all(cache.parent().unwrap()).unwrap();
    std::fs::write(
        cache,
        json!({
            "gpt-test": {
                "aliases": ["gpt-test-latest"],
                "input_cost_per_token": "0.000000001",
                "cache_read_input_token_cost": "0.0000000001",
                "output_cost_per_token": 0.0000025
            }
        })
        .to_string(),
    )
    .unwrap();
    let table = agent_rs::pricing::format_model_prices(
        home.path(),
        &[
            "openai:gpt-test".into(),
            "gpt-test-latest".into(),
            "missing".into(),
        ],
    );
    for model in ["openai:gpt-test", "gpt-test-latest"] {
        let row = table.lines().find(|line| line.contains(model)).unwrap();
        assert!(
            row.contains("$0.001") && row.contains("$0.0001") && row.contains("$2.50"),
            "{row}"
        );
    }
    let missing = table.lines().find(|line| line.contains("missing")).unwrap();
    assert_eq!(missing.matches("unknown").count(), 3);
}

#[test]
fn fast_rates_price_cached_input_and_long_context_without_standard_fallback() {
    let map = parse_pricing_map(
        &json!({
            "gpt-test": {
                "input_cost_per_token": 0.000002,
                "output_cost_per_token": 0.00001,
                "input_cost_per_token_priority": 0.000004,
                "cache_read_input_token_cost_priority": 0.0000002,
                "cache_creation_input_token_cost_priority": 0.000005,
                "output_cost_per_token_priority": 0.00002,
                "input_cost_per_token_above_272k_tokens_priority": 0.000008,
                "cache_read_input_token_cost_above_272k_tokens_priority": 0.0000004,
                "output_cost_per_token_above_272k_tokens_priority": 0.00003
            },
            "gpt-standard-only": {
                "input_cost_per_token": 0.000002,
                "output_cost_per_token": 0.00001
            }
        })
        .to_string(),
    )
    .unwrap();
    for tier in ["priority", "fast"] {
        let usage = Usage {
            input_tokens: 272_000,
            output_tokens: 1000,
            raw: Some(
                json!({"service_tier": tier, "input_tokens_details": {"cached_tokens": 100_000, "cache_write_tokens": 20_000}}),
            ),
        };
        let cost = cost_from_pricing_map(&map, "openai:gpt-test", &usage).unwrap();
        assert!((cost - 0.748).abs() < 1e-12, "{cost}");
        assert_eq!(
            cost_from_pricing_map(&map, "gpt-standard-only", &usage),
            None
        );

        let long = Usage {
            input_tokens: 272_001,
            raw: Some(
                json!({"service_tier": tier, "input_tokens_details": {"cached_tokens": 100_000}}),
            ),
            ..usage
        };
        let cost = cost_from_pricing_map(&map, "openai:gpt-test", &long).unwrap();
        assert!((cost - 1.446008).abs() < 1e-12, "{cost}");
    }
}
