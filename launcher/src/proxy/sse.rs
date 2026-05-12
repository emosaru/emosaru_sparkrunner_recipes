// Anthropic streaming SSE state machine + upstream OpenAI SSE parser.
//
// Inbound: vLLM emits OpenAI-shaped SSE events:
//   data: {"id":..., "choices":[{"index":0,"delta":{...},"finish_reason":null}]}
//   ...
//   data: {"id":..., "choices":[{"index":0,"delta":{},"finish_reason":"stop"}], "usage": {...}}
//   data: [DONE]
//
// Outbound: we emit Anthropic events in this order:
//   message_start
//   for each content block produced:
//     content_block_start { type: text|thinking|tool_use, ... }
//     content_block_delta { delta: text_delta | thinking_delta | input_json_delta }
//     ...
//     content_block_stop
//   message_delta { stop_reason, usage }
//   message_stop
//
// Block ordering: we open blocks in the order their first delta arrives:
//   1. thinking (if reasoning_content seen first)
//   2. text (any content)
//   3+. tool_use blocks (one per upstream tool_calls[].index)

use std::collections::HashMap;
use std::convert::Infallible;

use anyhow::Result;
use bytes::Bytes;
use futures::stream::{Stream, StreamExt};
use serde_json::{json, Value};

use super::translate::map_finish_reason;
use super::types::*;

#[derive(Debug, Clone)]
pub enum SseEvent {
    Event { name: String, data: String },
}

/// Parse a raw byte stream of SSE data from an upstream OpenAI server.
/// Yields OpenAIStreamEvent values; the final `[DONE]` marker terminates the stream.
pub fn parse_openai_sse<S>(stream: S) -> impl Stream<Item = Result<OpenAIStreamEvent>>
where
    S: Stream<Item = std::result::Result<Bytes, reqwest::Error>> + Unpin,
{
    use async_stream::stream;
    let mut buf: Vec<u8> = Vec::with_capacity(8192);
    let mut s = Box::pin(stream);
    stream! {
        while let Some(chunk) = s.next().await {
            let chunk = match chunk { Ok(c) => c, Err(e) => { yield Err(anyhow::anyhow!(e)); return; } };
            buf.extend_from_slice(&chunk);
            // Pull complete frames (terminated by "\n\n") out of buf.
            loop {
                let Some(pos) = find_frame_end(&buf) else { break };
                let frame = buf.drain(..pos).collect::<Vec<u8>>();
                // Consume the "\n\n" or "\r\n\r\n" terminator.
                let _ = buf.drain(..frame_term_len(&buf));
                let text = String::from_utf8_lossy(&frame);
                let data = extract_data(&text);
                if data.is_empty() { continue; }
                if data.trim() == "[DONE]" { return; }
                match serde_json::from_str::<OpenAIStreamEvent>(&data) {
                    Ok(ev) => yield Ok(ev),
                    Err(_) => continue, // best-effort: skip malformed
                }
            }
        }
    }
}

fn find_frame_end(buf: &[u8]) -> Option<usize> {
    // Return offset (relative to buf start) at which a complete frame ends.
    for i in 0..buf.len().saturating_sub(1) {
        if buf[i] == b'\n' && buf[i+1] == b'\n' { return Some(i + 1); }
        if i + 3 < buf.len()
            && buf[i] == b'\r' && buf[i+1] == b'\n' && buf[i+2] == b'\r' && buf[i+3] == b'\n'
        {
            return Some(i + 1);
        }
    }
    None
}
fn frame_term_len(buf: &[u8]) -> usize {
    if buf.starts_with(b"\n") { 1 } else if buf.starts_with(b"\r\n\r\n") { 4 } else { 1 }
}

fn extract_data(frame: &str) -> String {
    // Concatenate all `data:` lines (per SSE spec, multiple data lines join with \n).
    let mut out = String::new();
    for line in frame.split('\n') {
        let line = line.trim_end_matches('\r');
        if let Some(rest) = line.strip_prefix("data:") {
            if !out.is_empty() { out.push('\n'); }
            out.push_str(rest.trim_start());
        }
    }
    out
}

// ---------------------------------------------- Anthropic emission state ----

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BlockKind { Text, Thinking, ToolUse }

#[derive(Debug)]
pub struct AnthropicEmitter {
    requested_model: String,
    message_id: String,
    next_block_index: u32,
    current_block: Option<(u32, BlockKind)>, // (index, kind)
    // OpenAI tool_call index -> our anthropic block index
    tool_block: HashMap<u32, u32>,
    // Whether we've sent message_start.
    started: bool,
    pub input_tokens: u32,
    pub output_tokens: u32,
    pub stop_reason: Option<String>,
}

impl AnthropicEmitter {
    pub fn new(requested_model: String) -> Self {
        Self {
            requested_model,
            message_id: format!("msg_{}", uuid::Uuid::new_v4().simple()),
            next_block_index: 0,
            current_block: None,
            tool_block: HashMap::new(),
            started: false,
            input_tokens: 0,
            output_tokens: 0,
            stop_reason: None,
        }
    }

    pub fn start_event(&mut self) -> SseEvent {
        self.started = true;
        let body = json!({
            "type": "message_start",
            "message": {
                "id": self.message_id,
                "type": "message",
                "role": "assistant",
                "model": self.requested_model,
                "content": [],
                "stop_reason": null,
                "stop_sequence": null,
                "usage": {"input_tokens": 0, "output_tokens": 0}
            }
        });
        SseEvent::Event { name: "message_start".into(), data: body.to_string() }
    }

    fn close_current(&mut self, out: &mut Vec<SseEvent>) {
        if let Some((idx, _)) = self.current_block.take() {
            out.push(SseEvent::Event {
                name: "content_block_stop".into(),
                data: json!({"type":"content_block_stop","index":idx}).to_string(),
            });
        }
    }

    fn open_block(&mut self, kind: BlockKind, body: Value, out: &mut Vec<SseEvent>) -> u32 {
        let idx = self.next_block_index;
        self.next_block_index += 1;
        self.current_block = Some((idx, kind));
        out.push(SseEvent::Event {
            name: "content_block_start".into(),
            data: json!({"type":"content_block_start","index":idx,"content_block":body}).to_string(),
        });
        idx
    }

    /// Apply one upstream OpenAI delta event; append zero or more Anthropic SSE events.
    pub fn apply(&mut self, ev: &OpenAIStreamEvent, out: &mut Vec<SseEvent>) {
        if !self.started {
            out.push(self.start_event());
        }
        for choice in &ev.choices {
            let delta = &choice.delta;

            if let Some(rc) = &delta.reasoning_content {
                if !rc.is_empty() {
                    self.ensure_block(BlockKind::Thinking, out);
                    let (idx, _) = self.current_block.unwrap();
                    out.push(SseEvent::Event {
                        name: "content_block_delta".into(),
                        data: json!({
                            "type":"content_block_delta","index":idx,
                            "delta":{"type":"thinking_delta","thinking":rc}
                        }).to_string(),
                    });
                }
            }
            if let Some(text) = &delta.content {
                if !text.is_empty() {
                    self.ensure_block(BlockKind::Text, out);
                    let (idx, _) = self.current_block.unwrap();
                    out.push(SseEvent::Event {
                        name: "content_block_delta".into(),
                        data: json!({
                            "type":"content_block_delta","index":idx,
                            "delta":{"type":"text_delta","text":text}
                        }).to_string(),
                    });
                }
            }
            if let Some(tcs) = &delta.tool_calls {
                for tc in tcs {
                    self.handle_tool_call(tc, out);
                }
            }
            if let Some(fr) = &choice.finish_reason {
                self.stop_reason = Some(map_finish_reason(fr).to_string());
            }
        }
        if let Some(usage) = &ev.usage {
            self.input_tokens = usage.prompt_tokens;
            self.output_tokens = usage.completion_tokens;
        }
    }

    fn ensure_block(&mut self, kind: BlockKind, out: &mut Vec<SseEvent>) {
        if let Some((_, k)) = self.current_block {
            if k == kind { return; }
            // Different kind — close the current one before opening new.
            self.close_current(out);
        }
        let body = match kind {
            BlockKind::Text => json!({"type":"text","text":""}),
            BlockKind::Thinking => json!({"type":"thinking","thinking":""}),
            BlockKind::ToolUse => unreachable!("use handle_tool_call for tool_use"),
        };
        self.open_block(kind, body, out);
    }

    fn handle_tool_call(&mut self, tc: &OpenAIStreamToolCall, out: &mut Vec<SseEvent>) {
        let oi = tc.index;
        let block_idx = if let Some(&bi) = self.tool_block.get(&oi) {
            bi
        } else {
            // New tool: close anything currently open (text/thinking), then open ToolUse block.
            if let Some((_, k)) = self.current_block {
                if k != BlockKind::ToolUse { self.close_current(out); }
            }
            let id = tc.id.clone().unwrap_or_else(|| format!("toolu_{}", uuid::Uuid::new_v4().simple()));
            let name = tc.function.as_ref().and_then(|f| f.name.clone()).unwrap_or_default();
            let body = json!({"type":"tool_use","id": id,"name": name,"input":{}});
            let idx = self.open_block(BlockKind::ToolUse, body, out);
            self.tool_block.insert(oi, idx);
            idx
        };
        // Forward argument chunks as input_json_delta.
        if let Some(f) = &tc.function {
            if let Some(args) = &f.arguments {
                if !args.is_empty() {
                    out.push(SseEvent::Event {
                        name: "content_block_delta".into(),
                        data: json!({
                            "type":"content_block_delta","index": block_idx,
                            "delta": {"type":"input_json_delta","partial_json": args}
                        }).to_string(),
                    });
                }
            }
        }
    }

    /// Final events. Must be called once after the upstream stream is exhausted.
    pub fn finish(&mut self, out: &mut Vec<SseEvent>) {
        self.close_current(out);
        let stop_reason = self.stop_reason.clone().unwrap_or_else(|| "end_turn".into());
        let delta = json!({
            "type":"message_delta",
            "delta":{"stop_reason": stop_reason, "stop_sequence": null},
            "usage":{"input_tokens": self.input_tokens, "output_tokens": self.output_tokens}
        });
        out.push(SseEvent::Event { name: "message_delta".into(), data: delta.to_string() });
        out.push(SseEvent::Event {
            name: "message_stop".into(),
            data: json!({"type":"message_stop"}).to_string(),
        });
    }
}

// ----- axum SSE conversion ----------------------------------------------------

pub fn sse_event_to_axum(ev: SseEvent) -> std::result::Result<axum::response::sse::Event, Infallible> {
    let SseEvent::Event { name, data } = ev;
    Ok(axum::response::sse::Event::default().event(name).data(data))
}
