// OpenAI-port endpoints: /v1/models and /v1/chat/completions.
//
// /v1/chat/completions is a streaming-aware reverse proxy: we look up the
// model name in our route map, swap in the upstream's canonical model id,
// and forward to vLLM. For streaming responses we forward bytes as-is.

use std::time::Duration;

use axum::body::Body;
use axum::extract::State;
use axum::http::{HeaderMap, StatusCode};
use axum::response::{IntoResponse, Json, Response};
use futures::StreamExt;
use serde_json::json;

use super::router::{lookup, RouteMap};
use super::stats::{record, SharedStats};
use super::types::*;

#[derive(Clone)]
pub struct OpenAIState {
    pub routes: RouteMap,
    pub stats: SharedStats,
    pub client: reqwest::Client,
}

pub async fn list_models(State(s): State<OpenAIState>) -> impl IntoResponse {
    let created = chrono::Utc::now().timestamp() as u64;
    let entries: Vec<OpenAIModelEntry> = s.routes.keys().map(|id| OpenAIModelEntry {
        id: id.clone(), object: "model", created, owned_by: "stackctl".to_string(),
    }).collect();
    record(&s.stats, "openai:models", 200);
    Json(OpenAIModelsResponse { object: "list", data: entries })
}

pub async fn chat_completions(
    State(s): State<OpenAIState>,
    headers: HeaderMap,
    Json(mut req): Json<OpenAIChatRequest>,
) -> Response {
    let route = match lookup(&s.routes, &req.model) {
        Some(r) => r.clone(),
        None => {
            record(&s.stats, "openai:chat", 404);
            super::log::line("openai:chat", 404, Some(&req.model), Some("reason=unknown_model"));
            return error_response(StatusCode::NOT_FOUND, "model_not_found", &format!("unknown model: {}", req.model));
        }
    };
    req.model = route.canonical.clone();
    forward(&s, headers, "chat/completions", "openai:chat", &serde_json::to_value(&req).unwrap_or_default(), &route).await
}

pub async fn embeddings(
    State(s): State<OpenAIState>,
    headers: HeaderMap,
    Json(mut body): Json<serde_json::Value>,
) -> Response {
    proxy_json(&s, headers, &mut body, "embeddings", "openai:embeddings").await
}

pub async fn rerank(
    State(s): State<OpenAIState>,
    headers: HeaderMap,
    Json(mut body): Json<serde_json::Value>,
) -> Response {
    proxy_json(&s, headers, &mut body, "rerank", "openai:rerank").await
}

async fn proxy_json(
    s: &OpenAIState,
    headers: HeaderMap,
    body: &mut serde_json::Value,
    upstream_path: &str,
    route_label: &str,
) -> Response {
    let model = body.get("model").and_then(|v| v.as_str()).unwrap_or("").to_string();
    let route = match lookup(&s.routes, &model) {
        Some(r) => r.clone(),
        None => {
            record(&s.stats, route_label, 404);
            super::log::line(route_label, 404, Some(&model), Some("reason=unknown_model"));
            return error_response(StatusCode::NOT_FOUND, "model_not_found", &format!("unknown model: {model}"));
        }
    };
    // Substitute the upstream canonical model id before forwarding.
    body["model"] = serde_json::Value::String(route.canonical.clone());
    forward(s, headers, upstream_path, route_label, body, &route).await
}

async fn forward(
    s: &OpenAIState,
    headers: HeaderMap,
    upstream_path: &str,
    route_label: &str,
    body: &serde_json::Value,
    route: &super::router::UpstreamRoute,
) -> Response {
    let url = format!("{}/{}", route.api_base, upstream_path);
    let mut rb = s.client.post(&url).json(body).timeout(Duration::from_secs(600));
    if let Some(auth) = headers.get(axum::http::header::AUTHORIZATION) {
        if let Ok(v) = auth.to_str() { rb = rb.header("authorization", v); }
    } else {
        rb = rb.bearer_auth("not-needed");
    }

    let upstream = match rb.send().await {
        Ok(r) => r,
        Err(e) => {
            record(&s.stats, route_label, 502);
            super::log::line(route_label, 502, None, Some(&format!("reason=upstream_unreachable err={e}")));
            return error_response(StatusCode::BAD_GATEWAY, "upstream_unreachable", &format!("vllm unreachable: {e}"));
        }
    };
    let status = upstream.status();
    let status_axum = StatusCode::from_u16(status.as_u16()).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
    let content_type = upstream
        .headers()
        .get(axum::http::header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok()).unwrap_or("application/json").to_string();

    record(&s.stats, route_label, status.as_u16());

    let stream = upstream.bytes_stream().map(|res| res.map_err(std::io::Error::other));
    let body = Body::from_stream(stream);
    Response::builder()
        .status(status_axum)
        .header(axum::http::header::CONTENT_TYPE, content_type)
        .body(body)
        .unwrap()
}

pub fn error_response(status: StatusCode, kind: &str, msg: &str) -> Response {
    let body = json!({"error":{"type": kind, "message": msg}});
    (status, Json(body)).into_response()
}
