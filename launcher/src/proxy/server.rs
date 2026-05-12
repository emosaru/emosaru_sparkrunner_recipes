// Server orchestration. Bring up two axum apps (OpenAI port + Anthropic port)
// against a shared route map and stats handle. Exposes a ProxyHandle that
// stops both listeners on drop.

use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use axum::routing::{get, post};
use axum::Router;
use tokio::net::TcpListener;
use tokio::task::JoinHandle;

use super::anthropic::{self, AnthropicState};
use super::openai::{self, OpenAIState};
use super::router::{build, RouteMap};
use super::stats::{self, SharedStats};
use crate::recipe::Stack;

pub struct ProxyHandle {
    pub routes: RouteMap,
    pub stats: SharedStats,
    openai_task: JoinHandle<()>,
    anthropic_task: JoinHandle<()>,
}

impl ProxyHandle {
    pub fn abort(&self) {
        self.openai_task.abort();
        self.anthropic_task.abort();
    }
}

pub async fn start(stack: Arc<Stack>) -> Result<ProxyHandle> {
    let routes = build(&stack);
    let stats_handle = stats::new();
    stats_handle.write().started_at = Some(Instant::now());
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(600))
        .pool_idle_timeout(Duration::from_secs(60))
        .build()
        .context("build reqwest client")?;

    let oa_state = OpenAIState { routes: routes.clone(), stats: stats_handle.clone(), client: client.clone() };
    let an_state = AnthropicState { routes: routes.clone(), stats: stats_handle.clone(), client: client.clone() };

    let oa_app = Router::new()
        .route("/v1/models", get(openai::list_models))
        .route("/v1/chat/completions", post(openai::chat_completions))
        .route("/v1/embeddings", post(openai::embeddings))
        .route("/v1/rerank", post(openai::rerank))
        .fallback(unknown_path_openai)
        .with_state(oa_state);

    let an_app = Router::new()
        .route("/v1/models", get(anthropic::list_models))
        .route("/v1/messages", post(anthropic::messages))
        .fallback(unknown_path_anthropic)
        .with_state(an_state);

    let oa_addr = format!("{}:{}", stack.proxy.host, stack.proxy.openai_port);
    let an_addr = format!("{}:{}", stack.proxy.host, stack.proxy.anthropic_port);

    let oa_listener = TcpListener::bind(&oa_addr).await
        .with_context(|| format!("bind {oa_addr}"))?;
    let an_listener = TcpListener::bind(&an_addr).await
        .with_context(|| format!("bind {an_addr}"))?;

    let openai_task = tokio::spawn(async move {
        if let Err(e) = axum::serve(oa_listener, oa_app).await {
            eprintln!("OpenAI server exited: {e}");
        }
    });
    let anthropic_task = tokio::spawn(async move {
        if let Err(e) = axum::serve(an_listener, an_app).await {
            eprintln!("Anthropic server exited: {e}");
        }
    });

    Ok(ProxyHandle { routes, stats: stats_handle, openai_task, anthropic_task })
}

pub fn stop(handle: &ProxyHandle) {
    handle.abort();
}

async fn unknown_path_anthropic(req: axum::extract::Request) -> axum::response::Response {
    let method = req.method().to_string();
    let path = req.uri().path().to_string();
    super::log::line("anthropic:unknown_path", 404, None, Some(&format!("method={method} path={path}")));
    let body = serde_json::json!({"type":"error","error":{"type":"not_found_error","message":format!("unknown path: {method} {path}")}});
    (axum::http::StatusCode::NOT_FOUND, axum::Json(body)).into_response()
}

async fn unknown_path_openai(req: axum::extract::Request) -> axum::response::Response {
    let method = req.method().to_string();
    let path = req.uri().path().to_string();
    super::log::line("openai:unknown_path", 404, None, Some(&format!("method={method} path={path}")));
    let body = serde_json::json!({"error":{"type":"not_found","message":format!("unknown path: {method} {path}")}});
    (axum::http::StatusCode::NOT_FOUND, axum::Json(body)).into_response()
}
use axum::response::IntoResponse;
