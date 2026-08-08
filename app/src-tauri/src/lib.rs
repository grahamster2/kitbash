//! Kitbash desktop shell.
//!
//! Every call to the GPU server is proxied through Rust instead of being
//! fetched from the webview. The server is a plain FastAPI app with no CORS
//! middleware, and the webview's origin is `tauri://localhost` — a browser
//! fetch would be blocked before it left the process. Going through reqwest
//! also means the base URL can be any host on the tailnet without the webview
//! applying private-network or mixed-content rules to it.

use std::time::Duration;

use serde_json::Value;
use tauri::ipc::Response;

const JSON_TIMEOUT: Duration = Duration::from_secs(20);
// A finished mesh is a few hundred KB, but it may be crossing a tailnet from a
// machine that is also busy generating.
const MESH_TIMEOUT: Duration = Duration::from_secs(120);
// Assemble and export both do real mesh work on the server — loading every
// source part, welding, and for Roblox decimating each mesh to the triangle cap.
const WORK_TIMEOUT: Duration = Duration::from_secs(300);
// A candidate batch is four round-trips to a hosted image provider behind one
// request. Four seconds is typical; a rate-limited or retried provider is not,
// and failing that at the 20s JSON timeout would look like a server fault.
const IMAGE_TIMEOUT: Duration = Duration::from_secs(180);
// /decompose generates every reference image inline and builds every scripted
// part before it answers; only the meshes go onto the queue. That is seconds
// for a plan with one generation and minutes for one with eight, and the
// measured bonanza example held the connection for 22 s across ten parts.
const PLAN_TIMEOUT: Duration = Duration::from_secs(900);

fn client(timeout: Duration) -> Result<reqwest::Client, String> {
    reqwest::Client::builder()
        .timeout(timeout)
        .build()
        .map_err(|e| e.to_string())
}

fn url(base: &str, path: &str) -> String {
    format!("{}{}", base.trim_end_matches('/'), path)
}

/// The failure a user actually hits is "wrong host / server not up", so say
/// that rather than surfacing a bare transport error.
fn unreachable(base: &str, err: reqwest::Error) -> String {
    if err.is_timeout() {
        format!("{base} did not respond in time")
    } else if err.is_connect() {
        format!("could not reach the Kitbash server at {base}")
    } else {
        err.to_string()
    }
}

async fn get_json(base: &str, path: &str) -> Result<Value, String> {
    let res = client(JSON_TIMEOUT)?
        .get(url(base, path))
        .send()
        .await
        .map_err(|e| unreachable(base, e))?;
    let status = res.status();
    let body = res.text().await.map_err(|e| e.to_string())?;
    if !status.is_success() {
        return Err(format!("{path} -> {status}: {}", body.chars().take(300).collect::<String>()));
    }
    serde_json::from_str(&body).map_err(|e| e.to_string())
}

async fn post_json(base: &str, path: &str, body: &Value, timeout: Duration) -> Result<Value, String> {
    let res = client(timeout)?
        .post(url(base, path))
        .json(body)
        .send()
        .await
        .map_err(|e| unreachable(base, e))?;
    let status = res.status();
    let text = res.text().await.map_err(|e| e.to_string())?;
    if !status.is_success() {
        return Err(format!("POST {path} -> {status}: {}", text.chars().take(300).collect::<String>()));
    }
    serde_json::from_str(&text).map_err(|e| e.to_string())
}

async fn get_bytes(base: &str, path: &str, timeout: Duration) -> Result<Vec<u8>, String> {
    let res = client(timeout)?
        .get(url(base, path))
        .send()
        .await
        .map_err(|e| unreachable(base, e))?;
    let status = res.status();
    if !status.is_success() {
        let body = res.text().await.unwrap_or_default();
        return Err(format!("{path} -> {status}: {}", body.chars().take(300).collect::<String>()));
    }
    res.bytes().await.map(|b| b.to_vec()).map_err(|e| e.to_string())
}

/// Seeds the server field on first run. Same env var the MCP server uses, so a
/// machine that already knows where its GPU lives does not have to be told twice.
#[tauri::command]
fn default_base_url() -> String {
    std::env::var("KITBASH_SERVER_URL")
        .unwrap_or_else(|_| "http://127.0.0.1:8188".into())
        .trim_end_matches('/')
        .to_string()
}

#[tauri::command]
async fn health(base_url: String) -> Result<Value, String> {
    get_json(&base_url, "/health").await
}

#[tauri::command]
async fn list_jobs(base_url: String, limit: u32) -> Result<Value, String> {
    get_json(&base_url, &format!("/jobs?limit={limit}")).await
}

#[tauri::command]
async fn get_job(base_url: String, id: String) -> Result<Value, String> {
    get_json(&base_url, &format!("/jobs/{id}")).await
}

#[tauri::command]
async fn submit_job(base_url: String, body: Value) -> Result<Value, String> {
    post_json(&base_url, "/jobs", &body, JSON_TIMEOUT).await
}

/// The decision layer. Subject plus intent prose in, a strategy with its
/// evidence, its price and a runnable draft plan out — no LLM and no GPU, so
/// this answers in about a millisecond and the 20s JSON timeout is generous.
#[tauri::command]
async fn strategy(base_url: String, body: Value) -> Result<Value, String> {
    post_json(&base_url, "/strategy", &body, JSON_TIMEOUT).await
}

/// Runs a plan. Returns once every part is queued or built, with the
/// `/assemble` request already written — see `PLAN_TIMEOUT`.
#[tauri::command]
async fn decompose(base_url: String, body: Value) -> Result<Value, String> {
    post_json(&base_url, "/decompose", &body, PLAN_TIMEOUT).await
}

/// Real bounds for a finished part, so placement in the scene builder is
/// computed from measurements rather than guessed.
#[tauri::command]
async fn describe_job(base_url: String, id: String) -> Result<Value, String> {
    get_json(&base_url, &format!("/jobs/{id}/describe")).await
}

#[tauri::command]
async fn assemble(base_url: String, body: Value) -> Result<Value, String> {
    post_json(&base_url, "/assemble", &body, WORK_TIMEOUT).await
}

/// Give exactly one of `job_id` / `scene_id` in `body`; the server rejects both.
#[tauri::command]
async fn export(base_url: String, body: Value) -> Result<Value, String> {
    post_json(&base_url, "/export", &body, WORK_TIMEOUT).await
}

/// Returns raw GLB bytes. Tauri v2 hands `Response` to the webview as an
/// ArrayBuffer, so the mesh never gets base64'd on its way to three.js.
#[tauri::command]
async fn fetch_mesh(base_url: String, id: String) -> Result<Response, String> {
    get_bytes(&base_url, &format!("/jobs/{id}/mesh"), MESH_TIMEOUT)
        .await
        .map(Response::new)
}

/// Prompt -> several reference images to choose between, in one request.
#[tauri::command]
async fn create_candidates(base_url: String, body: Value) -> Result<Value, String> {
    post_json(&base_url, "/images/candidates", &body, IMAGE_TIMEOUT).await
}

/// Re-reads a batch. The POST returns the finished set, so this only earns its
/// keep against a server that fills a batch in behind an early response.
#[tauri::command]
async fn get_batch(base_url: String, id: String) -> Result<Value, String> {
    get_json(&base_url, &format!("/images/batches/{id}")).await
}

/// One image from one prompt — the pre-batch endpoint, kept as the fallback for
/// a server that has not been updated yet.
#[tauri::command]
async fn create_image(base_url: String, body: Value) -> Result<Value, String> {
    post_json(&base_url, "/images", &body, IMAGE_TIMEOUT).await
}

/// Returns raw PNG bytes for a candidate. Same `Response` treatment as the
/// meshes — see `fetch_mesh`; the JS side turns these into a blob URL.
#[tauri::command]
async fn fetch_image(base_url: String, id: String) -> Result<Response, String> {
    get_bytes(&base_url, &format!("/images/{id}"), MESH_TIMEOUT)
        .await
        .map(Response::new)
}

#[tauri::command]
async fn fetch_scene(base_url: String, id: String) -> Result<Response, String> {
    get_bytes(&base_url, &format!("/scenes/{id}/mesh"), MESH_TIMEOUT)
        .await
        .map(Response::new)
}

/// `path` is an absolute path *on the server* as returned by `/export`; the
/// bytes land at `dest` on this machine. Written here rather than handed to the
/// webview because an export is several files and the .obj runs to megabytes —
/// no reason to move any of it through the IPC bridge.
#[tauri::command]
async fn download_exported_file(base_url: String, path: String, dest: String) -> Result<String, String> {
    let encoded = urlencoding_component(&path);
    let bytes = get_bytes(&base_url, &format!("/export/file?path={encoded}"), MESH_TIMEOUT).await?;
    let dest = std::path::PathBuf::from(&dest);
    if let Some(parent) = dest.parent() {
        std::fs::create_dir_all(parent).map_err(|e| format!("{}: {e}", parent.display()))?;
    }
    std::fs::write(&dest, &bytes).map_err(|e| format!("{}: {e}", dest.display()))?;
    Ok(dest.to_string_lossy().into_owned())
}

/// Export paths are Windows paths — backslashes, drive colons and spaces all
/// have to survive the query string intact.
fn urlencoding_component(s: &str) -> String {
    s.bytes()
        .map(|b| match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => (b as char).to_string(),
            _ => format!("%{b:02X}"),
        })
        .collect()
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .invoke_handler(tauri::generate_handler![
            default_base_url,
            health,
            list_jobs,
            get_job,
            describe_job,
            submit_job,
            strategy,
            decompose,
            assemble,
            export,
            create_candidates,
            get_batch,
            create_image,
            fetch_image,
            fetch_mesh,
            fetch_scene,
            download_exported_file
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}

