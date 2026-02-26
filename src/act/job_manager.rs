//! # Async Job Manager v2 — CortexACT
//!
//! Spawns shell commands as background threads so the MCP tool call returns
//! immediately (avoiding client-side timeouts), then lets the agent poll for
//! completion via `cortex_check_job`.
//!
//! ## Storage layout
//!
//! ```text
//! ~/.cortexast/
//!   jobs/
//!     job_<hex>_<n>.log   ← combined stdout+stderr, appended line-by-line
//!   notifications.md       ← agent-friendly log of completed / failed jobs
//! ```
//!
//! ## Design decisions
//!
//! * **`std::thread` not `tokio::spawn`** — CortexAST runs inside a
//!   synchronous `BufRead` stdin loop; there is no ambient Tokio runtime.
//!   `std::thread::spawn` + `std::process::Command` is the correct primitive.
//!
//! * **File-based logs** — Long commands (e.g. `cargo build`, `pytest`) can
//!   emit megabytes of output. Buffering in-memory risks OOM; writing to a
//!   `.log` file is O(1) memory and lets `cortex_check_job` return a cheap
//!   tail without re-reading everything.
//!
//! * **Lazy 24 h cleanup** — Rather than a background timer thread, cleanup
//!   runs as a side-effect of every `spawn_job` call. This keeps the module
//!   single-threaded (no extra synchronisation) without leaving stale data
//!   forever.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::io::{BufRead, BufReader, Write};
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

// ─────────────────────────────────────────────────────────────────────────────
// Job state
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "state", content = "detail")]
pub enum JobState {
    Queued,
    Running,
    Done(i32),          // exit code
    Failed(String),     // error message (spawn error, kill, timeout, etc.)
}

impl JobState {
    pub fn label(&self) -> &'static str {
        match self {
            JobState::Queued  => "queued",
            JobState::Running => "running",
            JobState::Done(_) => "done",
            JobState::Failed(_) => "failed",
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Job record (stored in registry)
// ─────────────────────────────────────────────────────────────────────────────

struct Job {
    job_id:      String,
    command:     String,
    pid:         Option<u32>,
    state:       JobState,
    started_at:  u64,          // unix epoch seconds
    finished_at: Option<u64>,
    log_path:    PathBuf,
}

// ─────────────────────────────────────────────────────────────────────────────
// Response types (returned to the MCP caller)
// ─────────────────────────────────────────────────────────────────────────────

/// Returned by `spawn_job` on success.
#[derive(Debug, Serialize)]
pub struct SpawnResult {
    pub job_id:   String,
    pub pid:      Option<u32>,
    pub log_path: String,
    pub message:  String,
}

/// Returned by `check_job`.
#[derive(Debug, Serialize)]
pub struct CheckResult {
    pub job_id:       String,
    pub status:       String,
    pub pid:          Option<u32>,
    pub exit_code:    Option<i32>,
    pub duration_secs: u64,
    pub log_tail:     Vec<String>,   // last ≤20 lines from the log file
    pub log_path:     String,
}

// ─────────────────────────────────────────────────────────────────────────────
// Global registry
// ─────────────────────────────────────────────────────────────────────────────

fn registry() -> &'static Mutex<HashMap<String, Job>> {
    static REG: OnceLock<Mutex<HashMap<String, Job>>> = OnceLock::new();
    REG.get_or_init(|| Mutex::new(HashMap::new()))
}

static JOB_COUNTER: AtomicU64 = AtomicU64::new(0);

// ─────────────────────────────────────────────────────────────────────────────
// Path helpers
// ─────────────────────────────────────────────────────────────────────────────

fn cortexast_dir() -> PathBuf {
    dirs::home_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join(".cortexast")
}

fn jobs_dir() -> PathBuf {
    cortexast_dir().join("jobs")
}

fn notifications_path() -> PathBuf {
    cortexast_dir().join("notifications.md")
}

fn now_unix() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

fn format_unix_as_local(ts: u64) -> String {
    // Produce a simple UTC timestamp (no external time crate required).
    // Format: "2026-02-26 13:30:00 UTC"
    let secs = ts;
    let minutes = secs / 60;
    let hours   = minutes / 60;
    let days    = hours / 24;

    let s  = secs    % 60;
    let m  = minutes % 60;
    let h  = hours   % 24;

    // Approximate Gregorian date from days since epoch (good enough for logs)
    let mut y: u64 = 1970;
    let mut remaining_days = days;
    loop {
        let leap = (y % 4 == 0 && y % 100 != 0) || y % 400 == 0;
        let days_in_year: u64 = if leap { 366 } else { 365 };
        if remaining_days < days_in_year { break; }
        remaining_days -= days_in_year;
        y += 1;
    }
    let month_days: &[u64] = &[31,28,31,30,31,30,31,31,30,31,30,31];
    let mut month: u64 = 1;
    for &d in month_days {
        if remaining_days < d { break; }
        remaining_days -= d;
        month += 1;
    }
    let day = remaining_days + 1;
    format!("{y:04}-{month:02}-{day:02} {h:02}:{m:02}:{s:02} UTC")
}

// ─────────────────────────────────────────────────────────────────────────────
// Log tail helper
// ─────────────────────────────────────────────────────────────────────────────

/// Read the last `n` non-empty lines from a file. O(file) but files are
/// typically small (< 10 MB) and only read on explicit `check_job` calls.
fn tail_lines(path: &PathBuf, n: usize) -> Vec<String> {
    let Ok(f) = std::fs::File::open(path) else { return vec![] };
    let reader = BufReader::new(f);
    let mut lines: Vec<String> = reader
        .lines()
        .filter_map(|l| l.ok())
        .collect();
    if lines.len() > n {
        lines.drain(..lines.len() - n);
    }
    lines
}

// ─────────────────────────────────────────────────────────────────────────────
// Notification writer
// ─────────────────────────────────────────────────────────────────────────────

fn append_notification(job: &Job) {
    let path = notifications_path();
    if let Some(parent) = path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }

    let ts = format_unix_as_local(job.finished_at.unwrap_or_else(now_unix));
    let state_label = match &job.state {
        JobState::Done(code) => format!("DONE (exit {})", code),
        JobState::Failed(msg) => format!("FAILED — {}", msg),
        _ => "UNKNOWN".to_string(),
    };
    let duration = job
        .finished_at
        .map(|f| f.saturating_sub(job.started_at))
        .unwrap_or(0);

    let block = format!(
        "\n## [{state_label}] {job_id} — {ts}\n\
         \n\
         - **Command:** `{command}`\n\
         - **Duration:** {duration} s\n\
         - **Log:** `{log}`\n\
         \n\
         ---\n",
        job_id  = job.job_id,
        command = job.command,
        log     = job.log_path.display(),
    );

    // Append-only; create file if missing.
    if let Ok(mut file) = std::fs::OpenOptions::new()
        .create(true).append(true).open(&path)
    {
        let _ = file.write_all(block.as_bytes());
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Auto-cleanup: remove jobs (and logs) older than `max_age_secs`
// ─────────────────────────────────────────────────────────────────────────────

fn cleanup_old_jobs(max_age_secs: u64) {
    let now = now_unix();
    let mut reg = match registry().lock() {
        Ok(g)  => g,
        Err(_) => return,
    };
    let stale: Vec<String> = reg
        .iter()
        .filter(|(_, job)| now.saturating_sub(job.started_at) > max_age_secs)
        .map(|(id, _)| id.clone())
        .collect();

    for id in stale {
        if let Some(job) = reg.remove(&id) {
            // Best-effort log deletion — ignore errors.
            let _ = std::fs::remove_file(&job.log_path);
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Public API
// ─────────────────────────────────────────────────────────────────────────────

/// Spawn a shell command in a detached background thread.
///
/// The command runs even after this function returns.
/// `stdout` and `stderr` are both redirected to `.cortexast/jobs/{job_id}.log`.
pub fn spawn_job(
    command:      String,
    cwd:          Option<String>,
    timeout_secs: u64,
) -> Result<SpawnResult> {
    // Lazy cleanup of jobs older than 24 h (runs synchronously, very fast).
    cleanup_old_jobs(86_400);

    // Create jobs directory if absent.
    let dir = jobs_dir();
    std::fs::create_dir_all(&dir)
        .context("Failed to create ~/.cortexast/jobs/")?;

    // Generate a unique, human-readable job ID.
    let n      = JOB_COUNTER.fetch_add(1, Ordering::Relaxed);
    let job_id = format!("job_{:x}_{}", now_unix(), n);
    let log_path = dir.join(format!("{}.log", job_id));

    let started_at = now_unix();

    // Open the log file for writing before spawning (so the thread can use it).
    let log_file = std::fs::File::create(&log_path)
        .with_context(|| format!("Failed to create log file {}", log_path.display()))?;

    // --- Spawn the child process ---
    let mut cmd = Command::new("sh");
    cmd.arg("-c").arg(&command)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());

    if let Some(ref cwd_str) = cwd {
        cmd.current_dir(cwd_str);
    }

    let mut child: Child = cmd.spawn()
        .with_context(|| format!("Failed to spawn command: {}", command))?;

    let pid = child.id();

    // Extract piped handles before moving `child`.
    let child_stdout = child.stdout.take();
    let child_stderr = child.stderr.take();

    // Register the job immediately so `check_job` can find it.
    {
        let mut reg = registry().lock().unwrap();
        reg.insert(job_id.clone(), Job {
            job_id:      job_id.clone(),
            command:     command.clone(),
            pid:         Some(pid),
            state:       JobState::Running,
            started_at,
            finished_at: None,
            log_path:    log_path.clone(),
        });
    }

    // --- Background thread: drain stdout+stderr → log file, then wait ---
    let jid_thread  = job_id.clone();
    let log_path_t  = log_path.clone();

    std::thread::Builder::new()
        .name(format!("cortex-act-{}", job_id))
        .spawn(move || {
            let start = Instant::now();

            // Write log header.
            let mut log = log_file;
            let _ = writeln!(log, "[cortex-act] job_id={jid_thread}");
            let _ = writeln!(log, "[cortex-act] command={command}");
            let _ = writeln!(log, "[cortex-act] started={}", format_unix_as_local(started_at));
            let _ = writeln!(log, "[cortex-act] ---");

            // Drain stdout in a sub-thread so stderr doesn't get blocked.
            let log_stdout = log.try_clone().ok();
            if let (Some(out), Some(mut log_w)) = (child_stdout, log_stdout) {
                std::thread::spawn(move || {
                    for line in BufReader::new(out).lines().flatten() {
                        let _ = writeln!(log_w, "[stdout] {line}");
                    }
                });
            }

            // Drain stderr in this thread (simpler — one writer per handle).
            let log_stderr = match std::fs::OpenOptions::new()
                .create(true).append(true).open(&log_path_t)
            {
                Ok(f)  => Some(f),
                Err(_) => None,
            };
            if let (Some(err), Some(mut log_e)) = (child_stderr, log_stderr) {
                std::thread::spawn(move || {
                    for line in BufReader::new(err).lines().flatten() {
                        let _ = writeln!(log_e, "[stderr] {line}");
                    }
                });
            }

            // Wait for the child (or until timeout).
            let final_state;
            loop {
                match child.try_wait() {
                    Ok(Some(status)) => {
                        let code = status.code().unwrap_or(-1);
                        final_state = JobState::Done(code);
                        break;
                    }
                    Ok(None) => {
                        // Still running — check timeout.
                        if start.elapsed() > Duration::from_secs(timeout_secs) {
                            let _ = child.kill();
                            final_state = JobState::Failed(format!(
                                "timeout after {timeout_secs}s"
                            ));
                            break;
                        }
                        std::thread::sleep(Duration::from_millis(200));
                    }
                    Err(e) => {
                        final_state = JobState::Failed(format!("wait error: {e}"));
                        break;
                    }
                }
            }

            let finished_at = now_unix();
            let duration = finished_at.saturating_sub(started_at);

            // Write log footer.
            if let Ok(mut log_footer) = std::fs::OpenOptions::new()
                .create(true).append(true).open(&log_path_t)
            {
                let _ = writeln!(log_footer, "[cortex-act] ---");
                let _ = writeln!(log_footer, "[cortex-act] finished={}", format_unix_as_local(finished_at));
                let _ = writeln!(log_footer, "[cortex-act] duration={duration}s");
                let _ = writeln!(log_footer, "[cortex-act] status={}", final_state.label());
            }

            // Update registry.
            if let Ok(mut reg) = registry().lock() {
                if let Some(job) = reg.get_mut(&jid_thread) {
                    job.state       = final_state;
                    job.finished_at = Some(finished_at);

                    // Append notification (clone what we need).
                    let snapshot = Job {
                        job_id:      job.job_id.clone(),
                        command:     job.command.clone(),
                        pid:         job.pid,
                        state:       job.state.clone(),
                        started_at:  job.started_at,
                        finished_at: job.finished_at,
                        log_path:    job.log_path.clone(),
                    };
                    drop(reg); // release lock before I/O
                    append_notification(&snapshot);
                }
            }
        })
        .context("Failed to spawn background thread")?;

    Ok(SpawnResult {
        job_id:   job_id.clone(),
        pid:      Some(pid),
        log_path: log_path.display().to_string(),
        message:  format!(
            "Job {job_id} started (pid={pid}). Poll with cortex_check_job. Log: {}",
            log_path.display()
        ),
    })
}

/// Poll the status of a previously spawned job.
///
/// Returns the last 20 lines from the log file so the agent can see progress
/// without requesting the whole file.
pub fn check_job(job_id: &str) -> Result<CheckResult> {
    let reg = registry().lock().unwrap();
    let job = reg
        .get(job_id)
        .with_context(|| format!("Job '{}' not found. It may have been cleaned up (24 h TTL).", job_id))?;

    let now     = now_unix();
    let duration = job
        .finished_at
        .unwrap_or(now)
        .saturating_sub(job.started_at);

    let exit_code = match &job.state {
        JobState::Done(code) => Some(*code),
        _ => None,
    };

    let log_tail = tail_lines(&job.log_path, 20);

    Ok(CheckResult {
        job_id:       job.job_id.clone(),
        status:       job.state.label().to_string(),
        pid:          job.pid,
        exit_code,
        duration_secs: duration,
        log_tail,
        log_path:     job.log_path.display().to_string(),
    })
}

/// Send SIGTERM to a running job and mark it as Failed.
pub fn kill_job(job_id: &str) -> Result<String> {
    let pid = {
        let reg = registry().lock().unwrap();
        let job = reg
            .get(job_id)
            .with_context(|| format!("Job '{}' not found.", job_id))?;
        if job.state != JobState::Running {
            return Ok(format!(
                "Job {} is not running (state: {}). Nothing to kill.",
                job_id,
                job.state.label()
            ));
        }
        job.pid
    };

    if let Some(pid) = pid {
        #[cfg(unix)]
        libc_kill(pid as i32, 15 /* SIGTERM */);
        #[cfg(windows)]
        {
            // On Windows, use taskkill.
            let _ = Command::new("taskkill")
                .args(["/PID", &pid.to_string(), "/F"])
                .output();
        }
    }

    // Update registry.
    if let Ok(mut reg) = registry().lock() {
        if let Some(job) = reg.get_mut(job_id) {
            job.state       = JobState::Failed("killed by user".to_string());
            job.finished_at = Some(now_unix());
        }
    }

    Ok(format!("Sent SIGTERM to job {} (pid={:?}). Marked as failed.", job_id, pid))
}

/// POSIX signal shim — avoids adding `libc` as a dependency.
#[cfg(unix)]
fn libc_kill(pid: i32, sig: i32) {
    extern "C" { fn kill(pid: i32, sig: i32) -> i32; }
    unsafe { kill(pid, sig); }
}

/// Manually remove jobs older than `max_age_secs`. Also deletes their log files.
/// Called automatically inside `spawn_job` with a 24 h window.
pub fn cleanup_jobs(max_age_secs: u64) {
    cleanup_old_jobs(max_age_secs);
}
