use anyhow::{bail, Context, Result};
use serde_json::{json, Value};

use super::types::*;

/// Translate an Anthropic /v1/messages request into an OpenAI /v1/chat/completions request.
pub fn anthropic_to_openai_request(req: &AnthropicMessagesRequest, upstream_model: &str) -> Result<OpenAIChatRequest> {
    let mut messages: Vec<OpenAIMessage> = Vec::new();

    // 1) Anthropic top-level `system` (string OR content blocks) -> OpenAI system message.
    if let Some(sys) = &req.system {
        let sys_text = render_system(sys);
        if !sys_text.is_empty() {
            messages.push(OpenAIMessage {
                role: "system".into(),
                content: Some(Value::String(sys_text)),
                tool_calls: None,
                tool_call_id: None,
                name: None,
                reasoning_content: None,
            });
        }
    }

    // 2) Translate each Anthropic message.
    for m in &req.messages {
        let blocks = parse_message_blocks(&m.content);
        let role = m.role.as_str();
        match role {
            "user" => {
                // tool_result blocks become separate role=tool messages; text/image blocks become a single user message.
                let mut user_parts: Vec<Value> = Vec::new();
                let mut tool_results: Vec<(String, Value, Option<bool>)> = Vec::new();
                for b in blocks {
                    match b {
                        AnthropicContentBlock::Text { text, .. } => {
                            user_parts.push(json!({"type":"text","text":text}));
                        }
                        AnthropicContentBlock::Image { source } => {
                            user_parts.push(image_part_from_anthropic(&source));
                        }
                        AnthropicContentBlock::ToolResult { tool_use_id, content, is_error } => {
                            tool_results.push((tool_use_id, content, is_error));
                        }
                        AnthropicContentBlock::ToolUse { .. }
                        | AnthropicContentBlock::Thinking { .. }
                        | AnthropicContentBlock::Unknown => {}
                    }
                }
                // Tool results come *before* the new user turn in OpenAI semantics.
                for (id, content, is_error) in tool_results {
                    messages.push(OpenAIMessage {
                        role: "tool".into(),
                        content: Some(Value::String(render_tool_result_content(&content, is_error))),
                        tool_calls: None,
                        tool_call_id: Some(id),
                        name: None,
                        reasoning_content: None,
                    });
                }
                if !user_parts.is_empty() {
                    let content = if user_parts.len() == 1 && user_parts[0]["type"] == "text" {
                        Some(user_parts[0]["text"].clone())
                    } else {
                        Some(Value::Array(user_parts))
                    };
                    messages.push(OpenAIMessage {
                        role: "user".into(),
                        content,
                        tool_calls: None, tool_call_id: None, name: None, reasoning_content: None,
                    });
                }
            }
            "assistant" => {
                let mut text_acc = String::new();
                let mut tool_calls: Vec<OpenAIToolCall> = Vec::new();
                for b in blocks {
                    match b {
                        AnthropicContentBlock::Text { text, .. } => { text_acc.push_str(&text); }
                        AnthropicContentBlock::Thinking { .. } => { /* drop from input; vLLM regenerates */ }
                        AnthropicContentBlock::ToolUse { id, name, input } => {
                            tool_calls.push(OpenAIToolCall {
                                id,
                                kind: "function".into(),
                                function: OpenAIToolCallFn {
                                    name,
                                    arguments: serde_json::to_string(&input).unwrap_or_else(|_| "{}".into()),
                                },
                            });
                        }
                        _ => {}
                    }
                }
                messages.push(OpenAIMessage {
                    role: "assistant".into(),
                    content: if text_acc.is_empty() { None } else { Some(Value::String(text_acc)) },
                    tool_calls: if tool_calls.is_empty() { None } else { Some(tool_calls) },
                    tool_call_id: None, name: None, reasoning_content: None,
                });
            }
            other => {
                bail!("unsupported Anthropic message role: {other}");
            }
        }
    }

    // 3) Tools / tool_choice.
    let tools = req.tools.as_ref().map(|ts| {
        ts.iter().map(|t| OpenAITool {
            kind: "function".into(),
            function: OpenAIFunction {
                name: t.name.clone(),
                description: t.description.clone(),
                parameters: Some(t.input_schema.clone()),
            },
        }).collect()
    });
    let tool_choice = req.tool_choice.as_ref().map(|tc| match tc {
        AnthropicToolChoice::Auto => json!("auto"),
        AnthropicToolChoice::Any  => json!("required"),
        AnthropicToolChoice::None => json!("none"),
        AnthropicToolChoice::Tool { name } => json!({"type":"function","function":{"name": name}}),
    });

    let stop = req.stop_sequences.as_ref().map(|s| json!(s));

    // Carry through any unknown extras (e.g., chat_template_kwargs) into OpenAI extra.
    let extra = req.extra.clone();

    Ok(OpenAIChatRequest {
        model: upstream_model.to_string(),
        messages,
        stream: req.stream,
        stream_options: req.stream.unwrap_or(false).then(|| json!({"include_usage": true})),
        max_tokens: Some(req.max_tokens),
        max_completion_tokens: None,
        temperature: req.temperature,
        top_p: req.top_p,
        top_k: req.top_k,
        stop,
        tools,
        tool_choice,
        n: None,
        extra,
    })
}

fn render_system(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        Value::Array(blocks) => {
            let mut acc = String::new();
            for b in blocks {
                if b.get("type").and_then(|t| t.as_str()) == Some("text") {
                    if let Some(t) = b.get("text").and_then(|t| t.as_str()) {
                        if !acc.is_empty() { acc.push_str("\n\n"); }
                        acc.push_str(t);
                    }
                }
            }
            acc
        }
        _ => String::new(),
    }
}

fn parse_message_blocks(content: &Value) -> Vec<AnthropicContentBlock> {
    match content {
        Value::String(s) => vec![AnthropicContentBlock::Text { text: s.clone(), cache_control: None }],
        Value::Array(items) => items.iter().filter_map(|b| {
            serde_json::from_value::<AnthropicContentBlock>(b.clone()).ok()
        }).collect(),
        _ => Vec::new(),
    }
}

fn image_part_from_anthropic(source: &Value) -> Value {
    // Anthropic image source: {type:"base64", media_type:"image/png", data:"..."} OR {type:"url", url:"..."}
    let stype = source.get("type").and_then(|t| t.as_str()).unwrap_or("");
    if stype == "url" {
        if let Some(url) = source.get("url").and_then(|t| t.as_str()) {
            return json!({"type":"image_url","image_url":{"url": url}});
        }
    }
    if stype == "base64" {
        let mt = source.get("media_type").and_then(|t| t.as_str()).unwrap_or("image/png");
        let data = source.get("data").and_then(|t| t.as_str()).unwrap_or("");
        let url = format!("data:{};base64,{}", mt, data);
        return json!({"type":"image_url","image_url":{"url": url}});
    }
    json!({"type":"text","text":"[unsupported image source]"})
}

fn render_tool_result_content(content: &Value, is_error: Option<bool>) -> String {
    let body = match content {
        Value::String(s) => s.clone(),
        Value::Array(items) => {
            let mut acc = String::new();
            for it in items {
                if it.get("type").and_then(|t| t.as_str()) == Some("text") {
                    if let Some(t) = it.get("text").and_then(|t| t.as_str()) {
                        if !acc.is_empty() { acc.push('\n'); }
                        acc.push_str(t);
                    }
                }
            }
            acc
        }
        _ => content.to_string(),
    };
    if is_error.unwrap_or(false) {
        format!("[error] {body}")
    } else {
        body
    }
}

/// Translate a non-streaming OpenAI response into an Anthropic envelope.
pub fn openai_to_anthropic_response(resp: &OpenAIChatResponse, requested_model: &str) -> Result<AnthropicMessagesResponse> {
    let choice = resp.choices.first().context("upstream returned no choices")?;
    let mut blocks: Vec<AnthropicResponseBlock> = Vec::new();

    if let Some(rc) = &choice.message.reasoning_content {
        if !rc.is_empty() {
            blocks.push(AnthropicResponseBlock::Thinking { thinking: rc.clone() });
        }
    }
    if let Some(content) = &choice.message.content {
        if let Some(s) = content.as_str() {
            if !s.is_empty() {
                blocks.push(AnthropicResponseBlock::Text { text: s.to_string() });
            }
        } else if let Some(parts) = content.as_array() {
            let mut acc = String::new();
            for p in parts {
                if p.get("type").and_then(|t| t.as_str()) == Some("text") {
                    if let Some(t) = p.get("text").and_then(|t| t.as_str()) {
                        acc.push_str(t);
                    }
                }
            }
            if !acc.is_empty() {
                blocks.push(AnthropicResponseBlock::Text { text: acc });
            }
        }
    }
    if let Some(tcs) = &choice.message.tool_calls {
        for tc in tcs {
            let input: Value = if tc.function.arguments.is_empty() {
                json!({})
            } else {
                serde_json::from_str(&tc.function.arguments).unwrap_or_else(|_| json!({}))
            };
            blocks.push(AnthropicResponseBlock::ToolUse {
                id: tc.id.clone(),
                name: tc.function.name.clone(),
                input,
            });
        }
    }
    if blocks.is_empty() {
        // Always emit at least an empty text block so clients can render something.
        blocks.push(AnthropicResponseBlock::Text { text: String::new() });
    }

    let stop_reason = choice.finish_reason.as_deref().map(map_finish_reason).map(String::from);
    let usage = resp.usage.clone().unwrap_or_default();

    Ok(AnthropicMessagesResponse {
        id: format!("msg_{}", short_id(&resp.id)),
        kind: "message",
        role: "assistant",
        model: requested_model.to_string(),
        content: blocks,
        stop_reason,
        stop_sequence: None,
        usage: AnthropicUsage {
            input_tokens: usage.prompt_tokens,
            output_tokens: usage.completion_tokens,
        },
    })
}

pub fn map_finish_reason(r: &str) -> &'static str {
    match r {
        "stop" => "end_turn",
        "length" => "max_tokens",
        "tool_calls" => "tool_use",
        "content_filter" => "stop_sequence",
        _ => "end_turn",
    }
}

fn short_id(s: &str) -> String {
    // strip prefixes like "chatcmpl-" for cleanliness; cap length.
    let stripped = s.strip_prefix("chatcmpl-").unwrap_or(s);
    stripped.chars().take(24).collect()
}
