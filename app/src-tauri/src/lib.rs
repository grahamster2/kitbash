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
    let res = client(JSON_TIMEOUT)?
        .post(url(&base_url, "/jobs"))
        .json(&body)
        .send()
        .await
        .map_err(|e| unreachable(&base_url, e))?;
    let status = res.status();
    let text = res.text().await.map_err(|e| e.to_string())?;
    if !status.is_success() {
        return Err(format!("POST /jobs -> {status}: {}", text.chars().take(300).collect::<String>()));
    }
    serde_json::from_str(&text).map_err(|e| e.to_string())
}

/// Returns raw GLB bytes. Tauri v2 hands `Response` to the webview as an
/// ArrayBuffer, so the mesh never gets base64'd on its way to three.js.
#[tauri::command]
async fn fetch_mesh(base_url: String, id: String) -> Result<Response, String> {
    let res = client(MESH_TIMEOUT)?
        .get(url(&base_url, &format!("/jobs/{id}/mesh")))
        .send()
        .await
        .map_err(|e| unreachable(&base_url, e))?;
    let status = res.status();
    if !status.is_success() {
        let body = res.text().await.unwrap_or_default();
        return Err(format!("mesh {id} -> {status}: {}", body.chars().take(300).collect::<String>()));
    }
    let bytes = res.bytes().await.map_err(|e| e.to_string())?;
    Ok(Response::new(bytes.to_vec()))
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .invoke_handler(tauri::generate_handler![
            default_base_url,
            health,
            list_jobs,
            get_job,
            submit_job,
            fetch_mesh
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
