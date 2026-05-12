// Minimal access logger for the proxy. Appends one line per request to
// /tmp/stackctl-proxy.log so we can post-hoc see exactly which model/path
// each 4xx/5xx response was for. Cheap, always-on, no rotation (size grows;
// truncate manually if needed).

use std::io::Write;
use std::sync::OnceLock;

use parking_lot::Mutex;

static LOG_FILE: OnceLock<Mutex<std::fs::File>> = OnceLock::new();

fn file() -> &'static Mutex<std::fs::File> {
    LOG_FILE.get_or_init(|| {
        let path = std::env::var("STACKCTL_PROXY_LOG").unwrap_or_else(|_| "/tmp/stackctl-proxy.log".into());
        let f = std::fs::OpenOptions::new()
            .create(true).append(true).open(&path)
            .unwrap_or_else(|e| panic!("open {path}: {e}"));
        Mutex::new(f)
    })
}

pub fn line(route: &str, status: u16, model: Option<&str>, extra: Option<&str>) {
    let ts = chrono::Utc::now().format("%Y-%m-%dT%H:%M:%S%.3fZ");
    let model = model.unwrap_or("-");
    let extra = extra.unwrap_or("");
    let _ = writeln!(file().lock(), "{ts} {route} {status} model={model} {extra}");
}
