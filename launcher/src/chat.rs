// In-TUI chat: POST to our own OpenAI port, parse the SSE stream, append text
// deltas to the latest assistant message in shared AppState. Runs as a
// detached tokio task so the TUI keeps rendering deltas as they arrive.

use std::time::{Duration, Instant};

use bytes::Bytes;
use futures::StreamExt;
use serde_json::json;

use crate::metrics::{ChatMessage, SharedState};

pub fn submit(state: SharedState, model: String, openai_base: String) {
    tokio::spawn(async move { run(state, model, openai_base).await; });
}

async fn run(state: SharedState, model: String, openai_base: String) {
    // Snapshot the conversation and start a new assistant turn.
    let payload_messages = {
        let mut s = state.write();
        let c = s.chats.entry(model.clone()).or_default();
        let input = std::mem::take(&mut c.input);
        if input.trim().is_empty() { c.in_flight = false; return; }
        let user_msg = ChatMessage { role: "user".into(), content: input, ..Default::default() };
        c.messages.push(user_msg);
        // Snapshot conversation we'll send (everything so far, no empty placeholder yet).
        let history = c.messages.clone();
        c.messages.push(ChatMessage { role: "assistant".into(), ..Default::default() });
        c.in_flight = true;
        c.error = None;
        history.into_iter()
            .map(|m| json!({"role": m.role, "content": m.content}))
            .collect::<Vec<_>>()
    };

    let req_body = json!({
        "model": model,
        "messages": payload_messages,
        "stream": true,
        "max_tokens": 4096,
    });
    let url = format!("{}/v1/chat/completions", openai_base.trim_end_matches('/'));
    let client = match reqwest::Client::builder().timeout(Duration::from_secs(600)).build() {
        Ok(c) => c,
        Err(e) => { finish_with_error(&state, &model, format!("client build: {e}")); return; }
    };

    let resp = match client.post(&url).json(&req_body).send().await {
        Ok(r) => r,
        Err(e) => { finish_with_error(&state, &model, format!("POST {url}: {e}")); return; }
    };
    if !resp.status().is_success() {
        let status = resp.status();
        let body = resp.text().await.unwrap_or_default();
        finish_with_error(&state, &model, format!("HTTP {status}: {body}"));
        return;
    }

    let mut byte_stream = Box::pin(resp.bytes_stream());
    let mut buf: Vec<u8> = Vec::with_capacity(4096);
    while let Some(chunk) = byte_stream.next().await {
        let chunk: Bytes = match chunk { Ok(c) => c, Err(e) => { finish_with_error(&state, &model, format!("stream: {e}")); return; } };
        buf.extend_from_slice(&chunk);
        loop {
            let Some(end) = find_frame_end(&buf) else { break };
            let frame = buf.drain(..end).collect::<Vec<u8>>();
            let _ = buf.drain(..frame_term_len(&buf));
            let text = String::from_utf8_lossy(&frame);
            let data = extract_data(&text);
            if data.is_empty() { continue; }
            if data.trim() == "[DONE]" {
                mark_done(&state, &model);
                return;
            }
            let Ok(ev) = serde_json::from_str::<serde_json::Value>(&data) else { continue };
            let choices = ev.get("choices").and_then(|c| c.as_array()).cloned().unwrap_or_default();
            for ch in choices {
                let delta = ch.get("delta").cloned().unwrap_or(json!({}));
                // vLLM emits the thinking trace under `reasoning_content` (older)
                // or `reasoning` (newer); accept either.
                let thinking_chunk = delta.get("reasoning_content")
                    .and_then(|v| v.as_str())
                    .or_else(|| delta.get("reasoning").and_then(|v| v.as_str()));
                if let Some(t) = thinking_chunk {
                    if !t.is_empty() { append_thinking(&state, &model, t); }
                }
                if let Some(content) = delta.get("content").and_then(|v| v.as_str()) {
                    if !content.is_empty() { append_content(&state, &model, content); }
                }
            }
        }
    }
    mark_done(&state, &model);
}

fn append_content(state: &SharedState, model: &str, text: &str) {
    let mut s = state.write();
    if let Some(c) = s.chats.get_mut(model) {
        if let Some(last) = c.messages.last_mut() {
            if last.role == "assistant" {
                // First content delta means thinking has ended.
                if !last.thinking_complete {
                    last.thinking_complete = true;
                    last.thinking_ended_at = Some(Instant::now());
                }
                last.content.push_str(text);
            }
        }
    }
}

fn append_thinking(state: &SharedState, model: &str, text: &str) {
    let mut s = state.write();
    if let Some(c) = s.chats.get_mut(model) {
        if let Some(last) = c.messages.last_mut() {
            if last.role == "assistant" {
                if last.thinking_started_at.is_none() {
                    last.thinking_started_at = Some(Instant::now());
                }
                last.thinking.push_str(text);
            }
        }
    }
}

fn mark_done(state: &SharedState, model: &str) {
    let mut s = state.write();
    if let Some(c) = s.chats.get_mut(model) {
        c.in_flight = false;
        if let Some(last) = c.messages.last_mut() {
            if last.role == "assistant" && !last.thinking_complete && !last.thinking.is_empty() {
                last.thinking_complete = true;
                last.thinking_ended_at = Some(Instant::now());
            }
        }
    }
}

fn finish_with_error(state: &SharedState, model: &str, msg: String) {
    let mut s = state.write();
    if let Some(c) = s.chats.get_mut(model) {
        c.in_flight = false;
        c.error = Some(msg);
        // Drop the empty assistant placeholder, if any.
        if let Some(last) = c.messages.last() {
            if last.role == "assistant" && last.content.is_empty() {
                c.messages.pop();
            }
        }
    }
}

fn find_frame_end(buf: &[u8]) -> Option<usize> {
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
