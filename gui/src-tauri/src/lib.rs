use std::io::Write;
use std::sync::atomic::{AtomicI64, Ordering};
use std::sync::OnceLock;
use tauri::Manager;

static LAST_BEAT_MS: AtomicI64 = AtomicI64::new(0);
static LAST_OP: OnceLock<std::sync::Mutex<String>> = OnceLock::new();

fn log_path(app: &tauri::AppHandle) -> Option<std::path::PathBuf> {
    let dir = app.path().app_log_dir().ok()?;
    std::fs::create_dir_all(&dir).ok()?;
    Some(dir.join("tenryu-studio.log"))
}

fn append_line(app: &tauri::AppHandle, line: &str) {
    if let Some(p) = log_path(app) {
        // rotate at 5 MB
        if let Ok(meta) = std::fs::metadata(&p) {
            if meta.len() > 5_000_000 {
                let _ = std::fs::rename(&p, p.with_extension("log.1"));
            }
        }
        if let Ok(mut f) = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&p)
        {
            let now = chrono_free_timestamp();
            let _ = writeln!(f, "[{now}] {line}");
        }
    }
}

fn chrono_free_timestamp() -> String {
    // no chrono dependency: seconds since epoch + millis
    let d = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default();
    format!("{}.{:03}", d.as_secs(), d.subsec_millis())
}

#[tauri::command]
fn app_log(app: tauri::AppHandle, level: String, message: String) {
    if let Some(m) = LAST_OP.get() {
        if level == "op" {
            if let Ok(mut g) = m.lock() {
                *g = message.clone();
            }
        }
    }
    append_line(&app, &format!("[{level}] {message}"));
}

#[tauri::command]
fn app_heartbeat() {
    let d = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default();
    LAST_BEAT_MS.store(d.as_millis() as i64, Ordering::Relaxed);
}

#[tauri::command]
fn app_log_path(app: tauri::AppHandle) -> Option<String> {
    log_path(&app).map(|p| p.to_string_lossy().to_string())
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    let _ = LAST_OP.set(std::sync::Mutex::new(String::new()));
    tauri::Builder::default()
        .plugin(tauri_plugin_shell::init())
        .plugin(tauri_plugin_store::Builder::new().build())
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_fs::init())
        .invoke_handler(tauri::generate_handler![
            app_log,
            app_heartbeat,
            app_log_path
        ])
        .setup(|app| {
            let handle = app.handle().clone();
            append_line(&handle, "[lifecycle] app started");
            std::thread::spawn(move || {
                let mut reported = false;
                loop {
                    std::thread::sleep(std::time::Duration::from_secs(5));
                    let last_beat = LAST_BEAT_MS.load(Ordering::Relaxed);
                    if last_beat == 0 {
                        continue;
                    }
                    let now = std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .unwrap_or_default()
                        .as_millis() as i64;
                    if now - last_beat > 15_000 {
                        if !reported {
                            let last_op = LAST_OP
                                .get()
                                .and_then(|m| m.lock().ok().map(|g| g.clone()))
                                .unwrap_or_default();
                            append_line(
                                &handle,
                                &format!("[watchdog] UI unresponsive for >15s; last op: {last_op}"),
                            );
                            reported = true;
                        }
                    } else {
                        reported = false;
                    }
                }
            });
            Ok(())
        })
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
