use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use std::sync::atomic::{AtomicU64, Ordering};

// ─── Job Registry ─────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JobStatus {
    pub job_id: String,
    pub status: String, // "running" | "done" | "failed" | "timeout"
    pub exit_code: Option<i32>,
    pub stdout: String,
    pub stderr: String,
    pub started_at: u64,  // unix timestamp
    pub finished_at: Option<u64>,
}

struct JobRecord {
    status: JobStatus,
    handle: Option<std::thread::JoinHandle<()>>,
}

fn registry() -> &'static Mutex<HashMap<String, JobRecord>> {
    static REG: OnceLock<Mutex<HashMap<String, JobRecord>>> = OnceLock::new();
    REG.get_or_init(|| Mutex::new(HashMap::new()))
}

fn now_unix() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

// ─── Public API ───────────────────────────────────────────────────────────────

/// Spawn a shell command in a background thread. Returns a unique job_id.
pub fn spawn_job(
    command: String,
    cwd: Option<String>,
    timeout_secs: u64,
) -> Result<String> {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let job_id = format!(
        "job_{:x}_{}",
        now_unix(),
        COUNTER.fetch_add(1, Ordering::Relaxed)
    );
    let jid = job_id.clone();

    let started_at = now_unix();

    // Insert initial record
    {
        let mut reg = registry().lock().unwrap();
        reg.insert(
            job_id.clone(),
            JobRecord {
                status: JobStatus {
                    job_id: job_id.clone(),
                    status: "running".to_string(),
                    exit_code: None,
                    stdout: String::new(),
                    stderr: String::new(),
                    started_at,
                    finished_at: None,
                },
                handle: None,
            },
        );
    }

    // Spawn background thread
    let handle = std::thread::Builder::new()
        .name(format!("cortex-act-job-{}", job_id))
        .spawn(move || {
            let start = Instant::now();

            let mut cmd = std::process::Command::new("sh");
            cmd.arg("-c").arg(&command);

            if let Some(ref cwd_str) = cwd {
                cmd.current_dir(cwd_str);
            }

            // We collect output via wait_with_output
            let result = cmd
                .stdout(std::process::Stdio::piped())
                .stderr(std::process::Stdio::piped())
                .spawn()
                .and_then(|child| child.wait_with_output());

            let elapsed = start.elapsed();
            let finished_at = now_unix();

            let (final_status, exit_code, stdout, stderr) = match result {
                Ok(out) => {
                    let code = out.status.code().unwrap_or(-1);
                    let status = if elapsed > Duration::from_secs(timeout_secs) {
                        "timeout"
                    } else if out.status.success() {
                        "done"
                    } else {
                        "failed"
                    };
                    (
                        status.to_string(),
                        Some(code),
                        String::from_utf8_lossy(&out.stdout).into_owned(),
                        String::from_utf8_lossy(&out.stderr).into_owned(),
                    )
                }
                Err(e) => (
                    "failed".to_string(),
                    Some(-1),
                    String::new(),
                    format!("Process error: {}", e),
                ),
            };

            // Update registry with final state
            if let Ok(mut reg) = registry().lock() {
                if let Some(record) = reg.get_mut(&jid) {
                    record.status.status = final_status;
                    record.status.exit_code = exit_code;
                    record.status.stdout = stdout;
                    record.status.stderr = stderr;
                    record.status.finished_at = Some(finished_at);
                }
            }
        })
        .context("Failed to spawn background job thread")?;

    // Store handle
    {
        let mut reg = registry().lock().unwrap();
        if let Some(record) = reg.get_mut(&job_id) {
            record.handle = Some(handle);
        }
    }

    Ok(job_id)
}

/// Poll the status of a previously spawned job.
pub fn check_job(job_id: &str) -> Result<JobStatus> {
    let reg = registry().lock().unwrap();
    reg.get(job_id)
        .map(|r| r.status.clone())
        .with_context(|| format!("Job '{}' not found", job_id))
}
