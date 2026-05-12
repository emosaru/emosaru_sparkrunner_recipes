// Anthropic-port endpoints: /v1/models (Anthropic shape) and /v1/messages.
//
// /v1/messages does Anthropic ↔ OpenAI translation. For non-streaming we wait
// for the upstream response and translate. For streaming we drive the SSE
// state machine in proxy::sse to emit Anthropic events as upstream OpenAI
// deltas arrive.

use std::time::Duration;

use axum::extract::State;
use axum::http::{HeaderMap, StatusCode};
use axum::response::sse::{KeepAlive, Sse};
use axum::response::{IntoResponse, Json, Response};
use futures::StreamExt;
use serde_json::json;

use super::router::{lookup, RouteMap};
use super::sse::{parse_openai_sse, sse_event_to_axum, AnthropicEmitter};
use super::stats::{record, SharedStats};
use super::translate::{anthropic_to_openai_request, openai_to_anthropic_response};
use super::types::*;

#[derive(Clone)]
pub struct AnthropicState {
    pub routes: RouteMap,
    pub stats: SharedStats,
    pub client: reqwest::Client,
}

pub async fn list_models(State(s): State<AnthropicState>) -> impl IntoResponse {
    // Claude Code's /model picker filters out IDs that don't start with
    // `claude-`/`anthropic`, AND it rejects IDs containing `/` (real Anthropic
    // IDs use only hyphens and dots). Use picker_id() to sanitize: prepend
    // `claude-` and replace `/` with `-`. The route map already contains both
    // the original and sanitized forms, so lookups via either still work.
    use std::collections::BTreeMap;
    let created_at = chrono::Utc::now().to_rfc3339();
    let mut by_picker: BTreeMap<String, String> = BTreeMap::new();
    for id in s.routes.keys() {
        let pid = super::router::picker_id(id);
        // Dedupe: keep one entry per picker_id, prefer a display_name that
        // doesn't already match the picker_id (i.e. show the prettiest form).
        by_picker.entry(pid).or_insert_with(|| id.clone());
    }
    let entries: Vec<AnthropicModelEntry> = by_picker.into_iter().map(|(pid, display)| AnthropicModelEntry {
        kind: "model",
        id: pid,
        display_name: display,
        created_at: created_at.clone(),
    }).collect();
    let first_id = entries.first().map(|e| e.id.clone());
    let last_id = entries.last().map(|e| e.id.clone());
    record(&s.stats, "anthropic:models", 200);
    Json(AnthropicModelsResponse { data: entries, has_more: false, first_id, last_id })
}

/// Resolve a model name from a /v1/messages request to a route map key.
/// Tries exact match first (handles both canonical names and picker-ids,
/// since both are registered in the route map). Falls back to stripping
/// a leading `claude-` prefix so a stripped-bare name still resolves.
fn resolve_model<'a>(routes: &'a super::router::RouteMap, requested: &str) -> Option<&'a super::router::UpstreamRoute> {
    if let Some(r) = lookup(routes, requested) { return Some(r); }
    if let Some(stripped) = requested.strip_prefix("claude-") {
        return lookup(routes, stripped);
    }
    None
}

pub async fn messages(
    State(s): State<AnthropicState>,
    headers: HeaderMap,
    Json(req): Json<AnthropicMessagesRequest>,
) -> Response {
    let route = match resolve_model(&s.routes, &req.model) {
        Some(r) => r.clone(),
        None => {
            record(&s.stats, "anthropic:messages", 404);
            super::log::line("anthropic:messages", 404, Some(&req.model), Some("reason=unknown_model"));
            return anthropic_error(StatusCode::NOT_FOUND, "not_found_error", &format!("unknown model: {}", req.model));
        }
    };
    let requested_model = req.model.clone();
    let stream_mode = req.stream.unwrap_or(false);

    let oai_req = match anthropic_to_openai_request(&req, &route.canonical) {
        Ok(r) => r,
        Err(e) => {
            record(&s.stats, "anthropic:messages", 400);
            return anthropic_error(StatusCode::BAD_REQUEST, "invalid_request_error", &e.to_string());
        }
    };

    let url = format!("{}/chat/completions", route.api_base);
    let mut rb = s.client.post(&url).json(&oai_req).timeout(Duration::from_secs(600));
    // Pass through auth if the client sent one; otherwise vLLM doesn't care.
    if let Some(auth) = headers.get(axum::http::header::AUTHORIZATION) {
        if let Ok(v) = auth.to_str() { rb = rb.header("authorization", v); }
    } else if let Some(key) = headers.get("x-api-key").and_then(|v| v.to_str().ok()) {
        rb = rb.bearer_auth(key);
    } else {
        rb = rb.bearer_auth("not-needed");
    }

    let upstream = match rb.send().await {
        Ok(r) => r,
        Err(e) => {
            record(&s.stats, "anthropic:messages", 502);
            return anthropic_error(StatusCode::BAD_GATEWAY, "api_error", &format!("vllm unreachable: {e}"));
        }
    };
    let status = upstream.status();
    if !status.is_success() {
        let code = status.as_u16();
        let body = upstream.text().await.unwrap_or_default();
        record(&s.stats, "anthropic:messages", code);
        let kind = match code { 400 => "invalid_request_error", 401 => "authentication_error", 404 => "not_found_error", 429 => "rate_limit_error", 500..=599 => "api_error", _ => "api_error" };
        return anthropic_error(StatusCode::from_u16(code).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR), kind, &body);
    }

    if !stream_mode {
        // Non-streaming: parse full OpenAI body, translate, return JSON.
        let oai: OpenAIChatResponse = match upstream.json().await {
            Ok(v) => v,
            Err(e) => {
                record(&s.stats, "anthropic:messages", 502);
                return anthropic_error(StatusCode::BAD_GATEWAY, "api_error", &format!("upstream parse: {e}"));
            }
        };
        match openai_to_anthropic_response(&oai, &requested_model) {
            Ok(resp) => {
                record(&s.stats, "anthropic:messages", 200);
                Json(resp).into_response()
            }
            Err(e) => {
                record(&s.stats, "anthropic:messages", 500);
                anthropic_error(StatusCode::INTERNAL_SERVER_ERROR, "api_error", &e.to_string())
            }
        }
    } else {
        // Streaming: drive the emitter from the upstream SSE stream.
        record(&s.stats, "anthropic:messages", 200);
        let bytes = upstream.bytes_stream();
        let mut emitter = AnthropicEmitter::new(requested_model);
        let upstream_events = parse_openai_sse(bytes);
        let stream = async_stream::stream! {
            let mut upstream = Box::pin(upstream_events);
            // Emit message_start eagerly so clients see the message id.
            let start = emitter.start_event();
            yield sse_event_to_axum(start);
            while let Some(item) = upstream.next().await {
                let ev = match item {
                    Ok(e) => e,
                    Err(_) => continue,
                };
                let mut out = Vec::new();
                emitter.apply(&ev, &mut out);
                for e in out { yield sse_event_to_axum(e); }
            }
            let mut tail = Vec::new();
            emitter.finish(&mut tail);
            for e in tail { yield sse_event_to_axum(e); }
        };
        Sse::new(stream)
            .keep_alive(KeepAlive::default().text("ping").interval(Duration::from_secs(15)))
            .into_response()
    }
}

pub fn anthropic_error(status: StatusCode, kind: &str, msg: &str) -> Response {
    let body = json!({"type":"error","error":{"type": kind, "message": msg}});
    let mut resp = (status, Json(body)).into_response();
    resp.headers_mut().insert(
        axum::http::header::CONTENT_TYPE,
        axum::http::HeaderValue::from_static("application/json"),
    );
    resp
}
