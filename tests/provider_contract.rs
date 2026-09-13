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
async fn chat_provider_does_not_send_responses_text_verbosity_for_ollama() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "ok"
                    }
                }
            ],
            "usage": {
                "prompt_tokens": 1,
                "completion_tokens": 1
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
                content: "be brief".to_string(),
            }],
            ToolRegistry::new().definitions(),
        )
        .await
        .expect("provider response");

    assert_eq!(response.content, "ok");
    let requests = server.received_requests().await.expect("requests");
    let body: serde_json::Value = serde_json::from_slice(&requests[0].body).expect("json body");
    assert!(body.get("text").is_none());
}

#[tokio::test]
async fn responses_provider_requests_low_text_verbosity() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/responses"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "output_text": "ok",
            "usage": {
                "input_tokens": 1,
                "output_tokens": 1
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

    provider
        .complete(
            &[AgentMessage::User {
                content: "be brief".to_string(),
            }],
            ToolRegistry::new().definitions(),
        )
        .await
        .expect("provider response");

    let requests = server.received_requests().await.expect("requests");
    let body: serde_json::Value = serde_json::from_slice(&requests[0].body).expect("json body");
    assert_eq!(body["text"]["verbosity"], "low");
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
async fn agent_loop_observes_text_deltas_before_the_final_message() {
    #[derive(Clone, Debug)]
    struct StreamingProvider;

    #[async_trait]
    impl Provider for StreamingProvider {
        async fn complete(
            &self,
            _messages: &[AgentMessage],
            _tools: &[agent_rs::tools::ToolDefinition],
        ) -> Result<AssistantMessage, ProviderError> {
            unreachable!("the loop must use stream_events")
        }

        async fn stream_events(
            &self,
            _messages: &[AgentMessage],
            _tools: &[agent_rs::tools::ToolDefinition],
            on_event: &mut (dyn FnMut(ProviderEvent) + Send),
        ) -> Result<(), ProviderError> {
            on_event(ProviderEvent::TextDelta {
                text: "hel".to_string(),
            });
            tokio::time::sleep(std::time::Duration::from_millis(10)).await;
            on_event(ProviderEvent::TextDelta {
                text: "lo".to_string(),
            });
            on_event(ProviderEvent::FinalMessage {
                message: AssistantMessage {
                    content: "hello".to_string(),
                    tool_calls: Vec::new(),
                    usage: None,
                    model: None,
                    metadata: Default::default(),
                },
            });
            Ok(())
        }
    }

    let loop_runner = AgentLoop::new(
        StreamingProvider,
        ToolRegistry::new(),
        AgentLoopConfig {
            max_turns: 1,
            model: "mock".to_string(),
        },
    );
    let mut observed_events = Vec::new();
    let mut observed_messages = Vec::new();
    let result = loop_runner
        .run_turn_cancellable_with_event_observer(
            &[AgentMessage::User {
                content: "stream".to_string(),
            }],
            &CancellationToken::new(),
            |event| observed_events.push(event.clone()),
            |message| observed_messages.push(message.clone()),
            |_| {},
        )
        .await
        .expect("turn succeeds");

    assert!(matches!(&observed_events[0], ProviderEvent::TextDelta { text } if text == "hel"));
    assert!(matches!(&observed_events[1], ProviderEvent::TextDelta { text } if text == "lo"));
    assert!(matches!(
        observed_events[2],
        ProviderEvent::FinalMessage { .. }
    ));
    assert_eq!(result.final_text, "hello");
    assert_eq!(result.new_messages, observed_messages);
}

#[tokio::test]
async fn agent_loop_records_model_on_assistant_responses() {
    #[derive(Clone, Debug)]
    struct FinalProvider;

    #[async_trait]
    impl Provider for FinalProvider {
        async fn complete(
            &self,
            _messages: &[AgentMessage],
            _tools: &[agent_rs::tools::ToolDefinition],
        ) -> Result<AssistantMessage, ProviderError> {
            Ok(AssistantMessage {
                content: "done".to_string(),
                tool_calls: Vec::new(),
                usage: None,
                model: None,
                metadata: Default::default(),
            })
        }
    }

    let loop_runner = AgentLoop::new(
        FinalProvider,
        ToolRegistry::new(),
        AgentLoopConfig {
            max_turns: 1,
            model: "openai:gpt-5.6-sol".to_string(),
        },
    );
    let result = loop_runner
        .run_turn(&[AgentMessage::User {
            content: "finish".to_string(),
        }])
        .await
        .expect("turn succeeds");

    let AgentMessage::Assistant(assistant) = &result.new_messages[0] else {
        panic!("expected assistant message");
    };
    assert_eq!(assistant.model.as_deref(), Some("openai:gpt-5.6-sol"));
}

#[tokio::test]
async fn agent_loop_sends_full_history_without_silent_trimming() {
    #[derive(Clone, Debug)]
    struct InspectingProvider;

    #[async_trait]
    impl Provider for InspectingProvider {
        async fn complete(
            &self,
            messages: &[AgentMessage],
            _tools: &[agent_rs::tools::ToolDefinition],
        ) -> Result<AssistantMessage, ProviderError> {
            assert!(messages.iter().any(|message| {
                matches!(message, AgentMessage::User { content } if content == "original ask")
            }));
            assert!(messages.iter().any(|message| {
                matches!(message, AgentMessage::User { content } if content == "latest ask")
            }));
            Ok(AssistantMessage {
                content: "done".to_string(),
                tool_calls: Vec::new(),
                usage: None,
                model: None,
                metadata: Default::default(),
            })
        }
    }

    let loop_runner = AgentLoop::new(
        InspectingProvider,
        ToolRegistry::new(),
        AgentLoopConfig {
            max_turns: 1,
            model: "mock".to_string(),
        },
    );
    let result = loop_runner
        .run_turn(&[
            AgentMessage::User {
                content: "original ask".to_string(),
            },
            AgentMessage::Assistant(AssistantMessage {
                content: "large response ".repeat(10_000),
                tool_calls: Vec::new(),
                usage: None,
                model: None,
                metadata: Default::default(),
            }),
            AgentMessage::User {
                content: "latest ask".to_string(),
            },
        ])
        .await
        .expect("turn succeeds with full history");

    assert_eq!(result.final_text, "done");
}

#[tokio::test]
async fn agent_loop_returns_context_limit_error_without_retrying() {
    #[derive(Clone, Debug)]
    struct FullContextProvider {
        attempts: std::sync::Arc<std::sync::atomic::AtomicUsize>,
    }

    #[async_trait]
    impl Provider for FullContextProvider {
        async fn complete(
            &self,
            _messages: &[AgentMessage],
            _tools: &[agent_rs::tools::ToolDefinition],
        ) -> Result<AssistantMessage, ProviderError> {
            self.attempts
                .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            Err(ProviderError::request(
                "maximum context length exceeded".to_string(),
            ))
        }
    }

    let attempts = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let loop_runner = AgentLoop::new(
        FullContextProvider {
            attempts: attempts.clone(),
        },
        ToolRegistry::new(),
        AgentLoopConfig {
            max_turns: 2,
            model: "mock".to_string(),
        },
    );
    let err = loop_runner
        .run_turn(&[AgentMessage::User {
            content: "continue".to_string(),
        }])
        .await
        .expect_err("context overflow should be returned to the CLI");

    assert!(matches!(err, ProviderError::ContextLengthExceeded(_)));
    assert_eq!(attempts.load(std::sync::atomic::Ordering::SeqCst), 1);
}

#[tokio::test]
async fn agent_loop_bounces_non_context_provider_request_error_back_to_agent() {
    #[derive(Clone, Debug)]
    struct RecoveringProvider {
        attempts: std::sync::Arc<std::sync::atomic::AtomicUsize>,
    }

    #[async_trait]
    impl Provider for RecoveringProvider {
        async fn complete(
            &self,
            messages: &[AgentMessage],
            _tools: &[agent_rs::tools::ToolDefinition],
        ) -> Result<AssistantMessage, ProviderError> {
            let attempt = self
                .attempts
                .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            if attempt == 0 {
                return Err(ProviderError::Request(
                    "400 malformed tool history".to_string(),
                ));
            }
            assert!(messages.iter().any(|message| {
                matches!(message, AgentMessage::System { content } if content.contains("400 malformed tool history"))
            }));
            Ok(AssistantMessage {
                content: "recovered".to_string(),
                tool_calls: Vec::new(),
                usage: None,
                model: None,
                metadata: Default::default(),
            })
        }
    }

    let loop_runner = AgentLoop::new(
        RecoveringProvider {
            attempts: Default::default(),
        },
        ToolRegistry::new(),
        AgentLoopConfig {
            max_turns: 2,
            model: "mock".to_string(),
        },
    );
    let mut observed = Vec::new();
    let result = loop_runner
        .run_turn_cancellable_with_observer(
            &[AgentMessage::User {
                content: "continue".to_string(),
            }],
            &CancellationToken::new(),
            |message| observed.push(message.clone()),
            |_| {},
        )
        .await
        .expect("turn recovers");

    assert_eq!(result.final_text, "recovered");
    assert!(matches!(observed[0], AgentMessage::System { .. }));
    assert_eq!(result.new_messages, observed);
}

#[tokio::test]
async fn agent_loop_default_allows_more_than_legacy_turn_limit() {
    #[derive(Clone, Debug)]
    struct ManyToolCallsProvider {
        requests: std::sync::Arc<std::sync::atomic::AtomicUsize>,
    }

    #[async_trait]
    impl Provider for ManyToolCallsProvider {
        async fn complete(
            &self,
            _messages: &[AgentMessage],
            _tools: &[agent_rs::tools::ToolDefinition],
        ) -> Result<AssistantMessage, ProviderError> {
            let request = self
                .requests
                .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            if request < 200 {
                return Ok(AssistantMessage {
                    content: String::new(),
                    tool_calls: vec![ToolCall {
                        id: format!("call_{request}"),
                        name: "communicate".to_string(),
                        arguments: json!({
                            "intent": "continue the long-running task",
                            "message": "working"
                        }),
                    }],
                    usage: None,
                    model: None,
                    metadata: Default::default(),
                });
            }

            Ok(AssistantMessage {
                content: "completed after a long run".to_string(),
                tool_calls: Vec::new(),
                usage: None,
                model: None,
                metadata: Default::default(),
            })
        }
    }

    let loop_runner = AgentLoop::new(
        ManyToolCallsProvider {
            requests: Default::default(),
        },
        ToolRegistry::new(),
        AgentLoopConfig::default(),
    );

    let result = loop_runner
        .run_turn(&[AgentMessage::User {
            content: "keep working".to_string(),
        }])
        .await
        .expect("default limit should not cut off a run after 200 tool turns");

    assert_eq!(result.final_text, "completed after a long run");
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
                model: None,
                metadata: Default::default(),
            })
        }
    }

    let loop_runner = AgentLoop::new(
        LoopingProvider,
        ToolRegistry::new(),
        AgentLoopConfig {
            max_turns: 1,
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
                model: None,
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
                model: None,
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

#[tokio::test]
async fn chat_provider_sends_images_as_openai_compatible_content_parts() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "choices": [{"message": {"role": "assistant", "content": "a cat"}}]
        })))
        .mount(&server)
        .await;
    let provider = OpenAiCompatibleProvider::new(ProviderConfig {
        provider: "ollama".to_string(),
        model: "llava".to_string(),
        base_url: server.uri(),
        api_key: "test-key".to_string(),
        flavor: ProviderFlavor::OpenAiChat,
    });

    provider
        .complete(
            &[AgentMessage::UserWithImages {
                content: "What is this?".to_string(),
                images: vec![agent_rs::agent::ImageAttachment {
                    media_type: "image/png".to_string(),
                    data: "aGVsbG8=".to_string(),
                }],
            }],
            &[],
        )
        .await
        .expect("provider response");

    let requests = server.received_requests().await.expect("requests");
    let body: serde_json::Value = serde_json::from_slice(&requests[0].body).expect("json body");
    assert_eq!(
        body["messages"][0]["content"][0],
        json!({
            "type": "text", "text": "What is this?"
        })
    );
    assert_eq!(
        body["messages"][0]["content"][1],
        json!({
            "type": "image_url",
            "image_url": {"url": "data:image/png;base64,aGVsbG8="}
        })
    );
}

#[tokio::test]
async fn responses_provider_sends_images_as_input_image_parts() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/responses"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({"output_text": "ok"})))
        .mount(&server)
        .await;
    let provider = OpenAiCompatibleProvider::new(ProviderConfig {
        provider: "openai".to_string(),
        model: "gpt-test".to_string(),
        base_url: server.uri(),
        api_key: "test-key".to_string(),
        flavor: ProviderFlavor::OpenAiResponses,
    });

    provider
        .complete(
            &[AgentMessage::UserWithImages {
                content: "Read this".to_string(),
                images: vec![agent_rs::agent::ImageAttachment {
                    media_type: "image/jpeg".to_string(),
                    data: "/9j/".to_string(),
                }],
            }],
            &[],
        )
        .await
        .expect("provider response");

    let requests = server.received_requests().await.expect("requests");
    let body: serde_json::Value = serde_json::from_slice(&requests[0].body).expect("json body");
    assert_eq!(
        body["input"][0]["content"][0],
        json!({
            "type": "input_text", "text": "Read this"
        })
    );
    assert_eq!(
        body["input"][0]["content"][1],
        json!({
            "type": "input_image", "image_url": "data:image/jpeg;base64,/9j/"
        })
    );
}
