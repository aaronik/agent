use agent_rs::agent::{
    AgentLoop, AgentLoopConfig, AgentMessage, AssistantMessage, CancellationToken, ProviderEvent,
    ToolCall,
};
use agent_rs::providers::{
    OpenAiCompatibleProvider, Provider, ProviderConfig, ProviderError, ProviderFlavor,
};
use agent_rs::tools::ToolRegistry;
use async_trait::async_trait;
use serde_json::json;
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

#[tokio::test]
async fn chat_provider_parses_tool_calls() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {
                                    "name": "run_shell_command",
                                    "arguments": "{\"cmd\":\"echo hi\",\"timeout\":30}"
                                }
                            }
                        ]
                    }
                }
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 4
            }
        })))
        .mount(&server)
        .await;

    let provider = OpenAiCompatibleProvider::new(ProviderConfig {
        provider: "ollama".to_string(),
        model: "test-model".to_string(),
        base_url: server.uri(),
        api_key: "test-key".to_string(),
        flavor: ProviderFlavor::OpenAiChat,
    });

    let response = provider
        .complete(
            &[AgentMessage::User {
                content: "run echo hi".to_string(),
            }],
            ToolRegistry::new().definitions(),
        )
        .await
        .expect("provider response");

    assert_eq!(response.tool_calls.len(), 1);
    assert_eq!(response.tool_calls[0].name, "run_shell_command");
    assert_eq!(response.tool_calls[0].arguments["cmd"], "echo hi");
    assert_eq!(response.usage.expect("usage").input_tokens, 10);
}

#[tokio::test]
async fn chat_provider_streams_events() {
    let server = MockServer::start().await;
    let body = sse_body(vec![
        json!({
            "id": "chatcmpl_1",
            "object": "chat.completion.chunk",
            "created": 0,
            "model": "test-model",
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": "hel"}
                }
            ]
        }),
        json!({
            "id": "chatcmpl_1",
            "object": "chat.completion.chunk",
            "created": 0,
            "model": "test-model",
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": "lo"}
                }
            ]
        }),
        json!({
            "id": "chatcmpl_1",
            "object": "chat.completion.chunk",
            "created": 0,
            "model": "test-model",
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_1",
                                "type": "function",
                                "function": {
                                    "name": "run_shell_command",
                                    "arguments": "{\"cmd\""
                                }
                            }
                        ]
                    }
                }
            ]
        }),
        json!({
            "id": "chatcmpl_1",
            "object": "chat.completion.chunk",
            "created": 0,
            "model": "test-model",
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "function": {
                                    "arguments": ":\"echo hi\",\"timeout\":30}"
                                }
                            }
                        ]
                    }
                }
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5
            }
        }),
    ]);
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "text/event-stream")
                .set_body_string(body),
        )
        .mount(&server)
        .await;

    let provider = OpenAiCompatibleProvider::new(ProviderConfig {
        provider: "ollama".to_string(),
        model: "test-model".to_string(),
        base_url: server.uri(),
        api_key: "test-key".to_string(),
        flavor: ProviderFlavor::OpenAiChat,
    });

    let events = provider
        .events(
            &[AgentMessage::User {
                content: "run echo hi".to_string(),
            }],
            ToolRegistry::new().definitions(),
        )
        .await
        .expect("provider events");

    assert!(matches!(
        &events[0],
        ProviderEvent::TextDelta { text } if text == "hel"
    ));
    let final_message = events
        .iter()
        .find_map(|event| match event {
            ProviderEvent::FinalMessage { message } => Some(message),
            _ => None,
        })
        .expect("final message");
    assert_eq!(final_message.content, "hello");
    assert_eq!(final_message.tool_calls[0].name, "run_shell_command");
    assert_eq!(final_message.tool_calls[0].arguments["cmd"], "echo hi");
    assert_eq!(
        final_message.usage.as_ref().expect("usage").output_tokens,
        5
    );
}

#[tokio::test]
async fn responses_provider_parses_final_text_and_tool_calls() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/responses"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "output": [
                {
                    "type": "message",
                    "content": [
                        {
                            "type": "output_text",
                            "text": "hello"
                        }
                    ]
                },
                {
                    "type": "function_call",
                    "call_id": "call_2",
                    "name": "read_file",
                    "arguments": "{\"path\":\"./README.md\"}"
                }
            ],
            "usage": {
                "input_tokens": 7,
                "output_tokens": 3
            }
        })))
        .mount(&server)
        .await;

    let provider = OpenAiCompatibleProvider::new(ProviderConfig {
        provider: "openai".to_string(),
        model: "gpt-test".to_string(),
        base_url: server.uri(),
        api_key: "test-key".to_string(),
        flavor: ProviderFlavor::OpenAiResponses,
    });

    let response = provider
        .complete(
            &[AgentMessage::User {
                content: "read".to_string(),
            }],
            ToolRegistry::new().definitions(),
        )
        .await
        .expect("provider response");

    assert_eq!(response.content, "hello");
    assert_eq!(response.tool_calls.len(), 1);
    assert_eq!(response.tool_calls[0].name, "read_file");
    assert_eq!(response.tool_calls[0].arguments["path"], "./README.md");
    assert_eq!(response.usage.expect("usage").output_tokens, 3);
}

#[tokio::test]
async fn responses_provider_streams_events() {
    let server = MockServer::start().await;
    let body = sse_body(vec![
        json!({
            "type": "response.output_text.delta",
            "delta": "hi"
        }),
        json!({
            "type": "response.completed",
            "response": {
                "output": [
                    {
                        "type": "message",
                        "content": [
                            {
                                "type": "output_text",
                                "text": "hi"
                            }
                        ]
                    },
                    {
                        "type": "function_call",
                        "call_id": "call_2",
                        "name": "read_file",
                        "arguments": "{\"path\":\"./README.md\"}"
                    }
                ],
                "usage": {
                    "input_tokens": 7,
                    "output_tokens": 3
                }
            }
        }),
    ]);
    Mock::given(method("POST"))
        .and(path("/responses"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "text/event-stream")
                .set_body_string(body),
        )
        .mount(&server)
        .await;

    let provider = OpenAiCompatibleProvider::new(ProviderConfig {
        provider: "openai".to_string(),
        model: "gpt-test".to_string(),
        base_url: server.uri(),
        api_key: "test-key".to_string(),
        flavor: ProviderFlavor::OpenAiResponses,
    });

    let events = provider
        .events(
            &[AgentMessage::User {
                content: "read".to_string(),
            }],
            ToolRegistry::new().definitions(),
        )
        .await
        .expect("provider events");

    assert!(matches!(
        &events[0],
        ProviderEvent::TextDelta { text } if text == "hi"
    ));
    let final_message = events
        .iter()
        .find_map(|event| match event {
            ProviderEvent::FinalMessage { message } => Some(message),
            _ => None,
        })
        .expect("final message");
    assert_eq!(final_message.content, "hi");
    assert_eq!(final_message.tool_calls[0].name, "read_file");
    assert_eq!(final_message.tool_calls[0].arguments["path"], "./README.md");
    assert_eq!(final_message.usage.as_ref().expect("usage").input_tokens, 7);
}

fn sse_body(chunks: Vec<serde_json::Value>) -> String {
    let mut body = String::new();
    for chunk in chunks {
        body.push_str("data: ");
        body.push_str(&chunk.to_string());
        body.push_str("\n\n");
    }
    body.push_str("data: [DONE]\n\n");
    body
}

#[tokio::test]
async fn agent_loop_errors_on_max_turn_exhaustion() {
    #[derive(Clone, Debug)]
    struct LoopingProvider;

    #[async_trait]
    impl Provider for LoopingProvider {
        async fn complete(
            &self,
            _messages: &[AgentMessage],
            _tools: &[agent_rs::tools::ToolDefinition],
        ) -> Result<AssistantMessage, ProviderError> {
            Ok(AssistantMessage {
                content: String::new(),
                tool_calls: vec![ToolCall {
                    id: "call_1".to_string(),
                    name: "communicate".to_string(),
                    arguments: json!({"intent": "keep user updated", "message": "still working"}),
                }],
                usage: None,
                metadata: Default::default(),
            })
        }
    }

    let loop_runner = AgentLoop::new(
        LoopingProvider,
        ToolRegistry::new(),
        AgentLoopConfig {
            max_turns: 1,
            max_context_tokens: 16_384,
            model: "mock".to_string(),
        },
    );

    let err = loop_runner
        .run_turn(&[AgentMessage::User {
            content: "loop forever".to_string(),
        }])
        .await
        .expect_err("max-turn exhaustion should fail");

    assert!(matches!(err, ProviderError::MaxTurnsExceeded(1)));
}

#[tokio::test]
async fn agent_loop_respects_pre_cancelled_token() {
    let token = CancellationToken::new();
    token.cancel();
    let loop_runner = AgentLoop::new(
        agent_rs::providers::MockProvider::default(),
        ToolRegistry::new(),
        AgentLoopConfig {
            max_turns: 1,
            max_context_tokens: 16_384,
            model: "mock".to_string(),
        },
    );

    let err = loop_runner
        .run_turn_cancellable(
            &[AgentMessage::User {
                content: "run echo hi".to_string(),
            }],
            &token,
        )
        .await
        .expect_err("cancelled turn should fail");

    assert!(matches!(err, ProviderError::Cancelled));
}

#[tokio::test]
async fn agent_loop_aborts_in_flight_shell_tool_when_token_is_cancelled() {
    #[derive(Clone, Debug)]
    struct ShellToolProvider;

    #[async_trait]
    impl Provider for ShellToolProvider {
        async fn complete(
            &self,
            messages: &[AgentMessage],
            _tools: &[agent_rs::tools::ToolDefinition],
        ) -> Result<AssistantMessage, ProviderError> {
            let has_tool_result = messages
                .iter()
                .any(|message| matches!(message, AgentMessage::Tool(_)));
            Ok(AssistantMessage {
                content: if has_tool_result {
                    "too late".to_string()
                } else {
                    String::new()
                },
                tool_calls: if has_tool_result {
                    Vec::new()
                } else {
                    vec![ToolCall {
                        id: "call_sleep".to_string(),
                        name: "run_shell_command".to_string(),
                        arguments: json!({
                            "intent": "simulate a long running command",
                            "cmd": "/bin/sleep 60",
                            "timeout": 120,
                        }),
                    }]
                },
                usage: None,
                metadata: Default::default(),
            })
        }
    }

    let token = CancellationToken::new();
    let loop_runner = AgentLoop::new(
        ShellToolProvider,
        ToolRegistry::new(),
        AgentLoopConfig {
            max_turns: 2,
            max_context_tokens: 16_384,
            model: "mock".to_string(),
        },
    );

    let messages = [AgentMessage::User {
        content: "sleep".to_string(),
    }];
    let turn = loop_runner.run_turn_cancellable(&messages, &token);
    tokio::pin!(turn);

    tokio::time::sleep(std::time::Duration::from_millis(100)).await;
    token.cancel();

    let err = tokio::time::timeout(std::time::Duration::from_secs(1), turn)
        .await
        .expect("turn should abort promptly")
        .expect_err("cancelled turn should fail");

    assert!(matches!(err, ProviderError::Cancelled));
}

#[tokio::test]
async fn agent_loop_aborts_in_flight_provider_request_when_token_is_cancelled() {
    #[derive(Clone, Debug)]
    struct SlowProvider;

    #[async_trait]
    impl Provider for SlowProvider {
        async fn complete(
            &self,
            _messages: &[AgentMessage],
            _tools: &[agent_rs::tools::ToolDefinition],
        ) -> Result<AssistantMessage, ProviderError> {
            tokio::time::sleep(std::time::Duration::from_secs(60)).await;
            Ok(AssistantMessage {
                content: "too late".to_string(),
                tool_calls: Vec::new(),
                usage: None,
                metadata: Default::default(),
            })
        }
    }

    let token = CancellationToken::new();
    let loop_runner = AgentLoop::new(
        SlowProvider,
        ToolRegistry::new(),
        AgentLoopConfig {
            max_turns: 1,
            max_context_tokens: 16_384,
            model: "mock".to_string(),
        },
    );

    let messages = [AgentMessage::User {
        content: "wait".to_string(),
    }];
    let turn = loop_runner.run_turn_cancellable(&messages, &token);
    tokio::pin!(turn);

    tokio::time::sleep(std::time::Duration::from_millis(10)).await;
    token.cancel();

    let err = tokio::time::timeout(std::time::Duration::from_secs(1), turn)
        .await
        .expect("turn should abort promptly")
        .expect_err("cancelled turn should fail");

    assert!(matches!(err, ProviderError::Cancelled));
}
