use anyhow::Result;
use model2vec_rs::model::StaticModel;
use serde_json::json;
use std::io::{BufRead, Write};
use std::path::PathBuf;

use crate::chronos::{checkpoint_symbol, compare_symbol, list_checkpoints};
use crate::config::load_config;
use crate::inspector::{
    call_hierarchy, extract_symbols_from_source, find_implementations, find_usages,
    propagation_checklist, read_symbol_with_options, render_skeleton, repo_map_with_filter,
    run_diagnostics,
};
use crate::memory::{hybrid_search, MemoryStore};
use crate::rules::get_merged_rules;
use crate::scanner::{scan_workspace, ScanOptions};
use crate::slicer::{slice_paths_to_xml, slice_to_xml};
use crate::vector_store::{CodebaseIndex, IndexJob};
use rayon::prelude::*;

#[derive(Default)]
pub struct ServerState {
    /// Canonical workspace root. Populated from (highest priority first):
    ///   1. `repoPath` field in a tool call — per-call override.
    ///   2. MCP `initialize` params (`rootUri` / `rootPath` / `workspaceFolders`).
    ///   3. CLI `--root` / `CORTEXAST_ROOT` env var — startup bootstrap.
    ///   4. IDE-specific env vars (VSCODE_WORKSPACE_FOLDER, IDEA_INITIAL_DIRECTORY, …).
    ///   5. Find-up heuristic on tool args (`path` / `target_dir` / `target`).
    ///   6. `cwd` — last resort; refused if it equals $HOME or OS root.
    repo_root: Option<PathBuf>,
}

/// Returns `true` for "useless" roots that indicate the server started with the
/// wrong cwd (usually $HOME or filesystem root on any OS).
fn is_dead_root(p: &std::path::Path) -> bool {
    // `parent() == None` is the universal OS-root test across all platforms:
    //   `/`.parent()    → None  (Unix)
    //   `C:\`.parent()  → None  (Windows drive root — the old `count <= 1`
    //                            check missed this because C:\ has 2 components:
    //                            Prefix("C:") + RootDir)
    if p.parent().is_none() {
        return true;
    }
    // Bare single-component paths (e.g. ".") are also useless.
    if p.components().count() <= 1 {
        return true;
    }
    // Also catch $HOME specifically — no real project lives directly there.
    if let Ok(home) = std::env::var("HOME").or_else(|_| std::env::var("USERPROFILE")) {
        if p == std::path::Path::new(home.trim()) {
            return true;
        }
    }
    false
}

/// Parse a file URI (or plain path string) into an OS-native `PathBuf`.
///
/// Handles the cross-platform quirk that a simple `trim_start_matches("file://")`
/// gets wrong on Windows:
///
/// | URI input                        | Unix result             | Windows result       |
/// |----------------------------------|-------------------------|----------------------|
/// | `file:///Users/hero/project`     | `/Users/hero/project`   | (same — harmless)    |
/// | `file:///C:/Users/hero/project`  | `/C:/Users/hero/proj`   | `C:/Users/hero/proj` |
/// | plain `/Users/hero/project`      | `/Users/hero/project`   | (same)               |
///
/// On Windows, RFC 8089 file URIs encode the drive as `file:///C:/...`; after
/// stripping `file://` the leftover `/C:/...` must have its leading slash
/// removed to produce a valid Windows path.  We detect this with a byte-level
/// check (`bytes[1]` is ASCII alpha + `bytes[2]` == `:`), which cannot fire
/// for a legitimate Unix absolute path segment (e.g. `/Users/...`).
fn extract_path_from_uri(uri: &str) -> Option<PathBuf> {
    let rest = uri.strip_prefix("file://").unwrap_or(uri);

    // Windows drive-root artifact: strip the spurious leading `/` in `/C:/...`
    // so the result is a valid Windows path `C:/...`.
    let rest = if rest.starts_with('/')
        && rest.len() >= 3
        && rest.as_bytes()[1].is_ascii_alphabetic()
        && rest.as_bytes()[2] == b':'
    {
        &rest[1..]
    } else {
        rest
    };

    let s = rest.trim_end_matches('/');
    if s.is_empty() {
        None
    } else {
        Some(PathBuf::from(s))
    }
}

/// Helper to read the central codebase map (CortexSync config `codebases.json`)
fn get_network_map() -> Result<serde_json::Value, String> {
    let home = std::env::var("HOME")
        .or_else(|_| std::env::var("USERPROFILE"))
        .unwrap_or_default();
    
    // Path for CortexSync codebases config
    let config_path = PathBuf::from(home).join(".cortexast").join("codebases.json");

    if !config_path.exists() {
        return Err("Network map configuration not found at ~/.cortexast/codebases.json.".to_string());
    }

    match std::fs::read_to_string(&config_path) {
        Ok(contents) => {
            serde_json::from_str::<serde_json::Value>(&contents)
                .map_err(|e| format!("Failed to parse network map JSON: {}", e))
        }
        Err(e) => Err(format!("Failed to read network map: {}", e)),
    }
}

impl ServerState {
    /// Called once when the MCP `initialize` request is received.
    /// Extracts the workspace root from standard LSP/MCP protocol fields and
    /// writes it directly into `self.repo_root` — making the protocol signal
    /// the definitive canonical root. This is the only approach that works
    /// reliably across VS Code, Cursor, JetBrains, Zed, Neovim, and any other
    /// editor that correctly implements the MCP/LSP initialize spec.
    fn capture_init_root(&mut self, params: &serde_json::Value) {
        // Priority: workspaceFolders[0].uri → rootUri → rootPath
        // All three are standard MCP/LSP fields; strip file:// prefix and trailing slash.
        let raw_uri = params
            .get("workspaceFolders")
            .and_then(|f| f.as_array())
            .and_then(|a| a.first())
            .and_then(|f| f.get("uri").or_else(|| f.get("path")))
            .and_then(|v| v.as_str())
            .or_else(|| {
                params
                    .get("rootUri")
                    .or_else(|| params.get("rootPath"))
                    .and_then(|v| v.as_str())
            });

        // Use the cross-platform URI parser so Windows `file:///C:/...` URIs
        // are decoded correctly (simple trim_start_matches leaves `/C:/...`).
        let root = raw_uri.and_then(extract_path_from_uri);

        // The protocol root is authoritative — overwrite any earlier bootstrap
        // value (env vars / --root) so the editor's own answer always wins.
        if let Some(r) = root {
            self.repo_root = Some(r);
        }
    }

    fn repo_root_from_params(&mut self, params: &serde_json::Value) -> Result<PathBuf, String> {
        // ── Step 1: Explicit parameter (highest priority) ─────────────────────
        if let Some(path) = params.get("repoPath").and_then(|v| v.as_str()) {
            let pb = PathBuf::from(path);
            self.repo_root = Some(pb.clone());
            return Ok(pb);
        }

        // ── Step 2: Cached root (from MCP `initialize` or prior successful call)
        // This covers: --root CLI flag, CORTEXAST_ROOT, any IDE env var captured
        // at startup, and the MCP initialize protocol root (authoritative).
        if let Some(root) = &self.repo_root {
            return Ok(root.clone());
        }

        // ── Step 3: Cross-IDE environment variable cascade ────────────────────
        // Reached only when self.repo_root wasn't set at startup (e.g. the IDE
        // didn't export env vars into the MCP subprocess, AND no initialize was
        // received yet). Belt-and-suspenders: check the vars directly here too.
        let home = std::env::var("HOME")
            .or_else(|_| std::env::var("USERPROFILE"))
            .unwrap_or_default();
        // PWD and INIT_CWD must be filtered — they equal $HOME when the IDE
        // spawns the process in the wrong dir, which is a dead root.
        let env_root = std::env::var("CORTEXAST_ROOT")
            .ok()
            .or_else(|| std::env::var("VSCODE_WORKSPACE_FOLDER").ok())
            .or_else(|| std::env::var("IDEA_INITIAL_DIRECTORY").ok())
            .or_else(|| {
                std::env::var("INIT_CWD")
                    .ok()
                    .filter(|v| v.trim() != home.trim())
            })
            .or_else(|| {
                std::env::var("PWD")
                    .ok()
                    .filter(|v| v.trim() != home.trim())
            })
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .map(PathBuf::from);
        if let Some(pb) = env_root {
            self.repo_root = Some(pb.clone());
            return Ok(pb);
        }

        // ── Step 4: Find-up heuristic on the tool's path hint ─────────────────
        // Walk the hint's ancestor chain looking for a project root marker
        // (.git, Cargo.toml, package.json). This recovers cleanly even when the
        // hint is relative, as long as we can anchor it to an absolute base.
        let target_hint = params
            .get("target_dir")
            .or_else(|| params.get("path"))
            .or_else(|| params.get("target"))
            .and_then(|v| v.as_str());

        if let Some(hint) = target_hint {
            let hint_path = PathBuf::from(hint);
            let abs = if hint_path.is_absolute() {
                hint_path
            } else {
                std::env::current_dir()
                    .unwrap_or_else(|_| PathBuf::from("."))
                    .join(hint_path)
            };
            let mut current = abs;
            while let Some(parent) = current.parent() {
                if parent.join(".git").exists()
                    || parent.join("Cargo.toml").exists()
                    || parent.join("package.json").exists()
                {
                    let found = parent.to_path_buf();
                    self.repo_root = Some(found.clone());
                    return Ok(found);
                }
                current = parent.to_path_buf();
            }
        }

        // ── Step 5: CRITICAL safeguard — last resort is cwd ──────────────────
        let fallback = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
        if is_dead_root(&fallback) {
            return Err(format!(
                "CRITICAL: Workspace root resolved to '{}' (OS root or Home directory). \
                This would allow tools to destructively scan the entire filesystem. \
                Please provide the 'repoPath' parameter pointing to your project directory, \
                e.g. repoPath='/Users/you/projects/my-app'.",
                fallback.display()
            ));
        }

        self.repo_root = Some(fallback.clone());
        Ok(fallback)
    }

    /// Resolves the Omni-AST target project logic, enforcing strict whitelisting.
    /// Supports resolving from `codebases.json` via either ID or absolute path.
    fn resolve_target_project(&mut self, params: &serde_json::Value) -> Result<PathBuf, String> {
        // 1. Retrieve standard `repo_root` as fallback
        let base_root = self.repo_root_from_params(params)?;

        // 2. Check for Omni-AST `target_project` override
        if let Some(target_proj_str) = params.get("target_project").and_then(|v| v.as_str()).filter(|s| !s.is_empty()) {
            
            // 3. Load Whitelist
            let network_map = get_network_map()?;
            let codebases = network_map.as_array()
                .or_else(|| network_map.get("codebases").and_then(|v| v.as_array()))
                .ok_or_else(|| "Invalid network map format: missing codebase array.".to_string())?;

            // 4. Resolve by ID first, then fallback to match absolute path
            let mut resolved_path = None;
            for codebase in codebases {
                let id = codebase.get("id").and_then(|v| v.as_str()).unwrap_or_default();
                let path = codebase.get("path").and_then(|v| v.as_str()).unwrap_or_default();
                
                if target_proj_str == id || target_proj_str == path {
                    resolved_path = Some(PathBuf::from(path));
                    break;
                }
            }

            // 5. Enforce Security Constraints
            let override_path = match resolved_path {
                Some(p) => p,
                None => return Err(format!(
                    "CRITICAL: Omni-AST target_project '{}' is NOT in the approved network map whitelist. Access denied.",
                    target_proj_str
                )),
            };

            if !override_path.exists() {
                return Err(format!("CRITICAL: Omni-AST target_project path does not exist on disk: '{}'", override_path.display()));
            }

            return Ok(override_path);
        }

        // Default to the standard base_root
        Ok(base_root)
    }

    fn tool_list(&self, id: serde_json::Value) -> serde_json::Value {
        json!({
            "jsonrpc": "2.0",
            "id": id,
            "result": {
                "tools": [
                    {
                        "name": "cortex_code_explorer",
                        "description": "Codebase explorer. Use INSTEAD of ls/tree/find/cat. Two modes: `map_overview` (fast symbol map, near-zero tokens — run first on any repo) and `deep_slice` (token-budgeted XML with function bodies, vector-ranked by query). Use map_overview to orient; deep_slice to get code for editing.",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "action": {
                                    "type": "string",
                                    "enum": ["map_overview", "deep_slice"],
                                    "description": "map_overview: bird's-eye symbol map of a dir (requires target_dir='.'). deep_slice: token-budgeted XML with bodies (requires target file/dir; use single_file=true for a specific file, query for semantic ranking)."
                                },
                                "repoPath": { "type": "string", "description": "Abs path to repo root. Default: cwd." },
                                "target_project": { "type": "string", "description": "Cross-project: ID or abs path from network map. Overrides repoPath." },
                                "target_dir": { "type": "string", "description": "(map_overview) Dir to map. Use '.' for repo root." },
                                "search_filter": { "type": "string", "description": "(map_overview) Case-insensitive substring filter. OR via 'foo|bar'." },
                                "max_chars": { "type": "integer", "description": "Max output chars. Default 8000." },
                                "ignore_gitignore": { "type": "boolean", "description": "(map_overview) Include git-ignored files." },
                                "exclude": { "type": "array", "items": { "type": "string" }, "description": "Dir names to skip (e.g. ['node_modules','build'])." },
                                "target": { "type": "string", "description": "(deep_slice) Relative path to file or dir." },
                                "budget_tokens": { "type": "integer", "exclusiveMinimum": 0, "description": "(deep_slice) Token budget. Default 32000." },
                                "skeleton_only": { "type": "boolean", "description": "(deep_slice) Strip function bodies, return signatures only." },
                                "query": { "type": "string", "description": "(deep_slice) Semantic query for vector-ranked file selection." },
                                "query_limit": { "type": "integer", "description": "(deep_slice) Max files returned in query mode." },
                                "single_file": { "type": "boolean", "description": "(deep_slice) Skip vector search; return only the exact target file." },
                                "only_dir": { "type": "string", "description": "(deep_slice) Restrict semantic search to this subdir only." }
                            },
                            "required": ["action"]
                        }
                    },
                    {
                        "name": "cortex_symbol_analyzer",
                        "description": "AST symbol analysis. Use INSTEAD of grep/rg. Actions: read_source (extract exact source of a symbol from a file — do this before editing), find_usages (all call/type/field sites), find_implementations (structs implementing a trait), blast_radius (callers + callees — run before rename/delete), propagation_checklist (exhaustive update checklist for shared types).",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "action": {
                                    "type": "string",
                                    "enum": ["read_source", "find_usages", "find_implementations", "blast_radius", "propagation_checklist"],
                                    "description": "read_source: exact symbol body (needs path+symbol_name; use symbol_names[] for batch). find_usages: all call/type/field sites (needs symbol_name+target_dir). find_implementations: structs that impl a trait. blast_radius: full caller+callee hierarchy (run before rename/delete). propagation_checklist: Markdown checklist of all update sites for a shared type."
                                },
                                "repoPath": { "type": "string", "description": "Abs path to repo root." },
                                "target_project": { "type": "string", "description": "Cross-project: ID or abs path. Overrides repoPath." },
                                "symbol_name": { "type": "string", "description": "Target symbol name (exact, no regex)." },
                                "target_dir": { "type": "string", "description": "Scope dir ('.' = whole repo). Required for find_usages/blast_radius." },
                                "ignore_gitignore": { "type": "boolean", "description": "(propagation_checklist) Include git-ignored files." },
                                "max_chars": { "type": "integer", "description": "Max output chars. Default 8000." },
                                "only_dir": { "type": "string", "description": "(propagation_checklist) Restrict scan to this subdir." },
                                "aliases": { "type": "array", "items": { "type": "string" }, "description": "(propagation_checklist) Alternative names across language boundaries." },
                                "path": { "type": "string", "description": "(read_source) Source file. Required." },
                                "symbol_names": { "type": "array", "items": { "type": "string" }, "description": "(read_source) Batch: extract multiple symbols from path." },
                                "skeleton_only": { "type": "boolean", "description": "(read_source) Return signatures only, strip bodies." },
                                "instance_index": { "type": "integer", "description": "(read_source) 0-based index when symbol has multiple definitions in the file." },
                                "changed_path": { "type": "string", "description": "(propagation_checklist) Contract file path (e.g. .proto) — overrides symbol mode." },
                                "max_symbols": { "type": "integer", "description": "(propagation_checklist) Max extracted symbols. Default 20." }
                            },
                            "required": ["action"]
                        }
                    },
                    {
                        "name": "cortex_chronos",
                        "description": "AST snapshot tool for safe refactors. Workflow: save_checkpoint (before edit) → edit → compare_checkpoint (verify). Use instead of git diff — AST-level, ignores formatting noise. Actions: save_checkpoint, list_checkpoints, compare_checkpoint, delete_checkpoint.",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "action": {
                                    "type": "string",
                                    "enum": ["save_checkpoint", "list_checkpoints", "compare_checkpoint", "delete_checkpoint"],
                                    "description": "save_checkpoint: snapshot symbol before edit (needs path+symbol_name+tag). list_checkpoints: list all saved tags. compare_checkpoint: AST diff between two tags (needs symbol_name+tag_a+tag_b; tag_b='__live__' for on-disk state). delete_checkpoint: remove by namespace/symbol/tag."
                                },
                                "repoPath": { "type": "string", "description": "Abs path to repo root." },
                                "namespace": { "type": "string", "description": "Checkpoint group (default 'default'). delete_checkpoint with namespace only purges the whole group." },
                                "max_chars": { "type": "integer", "description": "Max output chars. Default 8000." },
                                "path": { "type": "string", "description": "Source file (required for save; optional for compare)." },
                                "symbol_name": { "type": "string", "description": "Target symbol name." },
                                "semantic_tag": { "type": "string", "description": "Tag name (e.g. 'pre-refactor')." },
                                "tag": { "type": "string", "description": "Alias for semantic_tag." },
                                "tag_a": { "type": "string", "description": "(compare) First tag." },
                                "tag_b": { "type": "string", "description": "(compare) Second tag. '__live__' = current file on disk." }
                            },
                            "required": ["action"]
                        }
                    },
                    {
                        "name": "run_diagnostics",
                        "description": "Run compiler diagnostics (cargo check / tsc / gcc). Call after any code edit to catch errors before proceeding. Returns file, line, code, message — structured for targeted fixes.",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "repoPath": { "type": "string" },
                                "target_project": { "type": "string", "description": "OMNI-AST: Optional ID or absolute path of another codebase in the network map. Overrides repoPath for cross-project exploration." },
                                "max_chars": { "type": "integer", "description": "Optional: Limit output length. Default 8000 (safe for VS Code Copilot inline)." }
                            },
                            "required": ["repoPath"]
                        }
                    },
                    {
                        "name": "cortex_memory_retriever",
                        "description": "Search past agent decisions in global memory (semantic + keyword hybrid). Call BEFORE any research or exploration — the answer may already be cached. Returns ranked entries: intent, decision, tags, files_touched.",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "query": { "type": "string", "description": "Natural-language search query." },
                                "top_k": { "type": "integer", "description": "Max results. Default 5.", "default": 5 },
                                "tags": { "type": "array", "items": { "type": "string" }, "description": "Filter by tags (case-insensitive)." },
                                "project_path": { "type": "string", "description": "Filter to entries matching this project path substring." },
                                "max_chars": { "type": "integer", "description": "Max output chars. Default 8000." }
                            },
                            "required": ["query"]
                        }
                    },
                    {
                        "name": "cortex_get_rules",
                        "description": "Fetch codebase AI rules for the current context. Returns merged rules filtered by file_path (frontend/backend/db context). Call before starting any task in a new project.",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "project_path": { "type": "string", "description": "Abs path to project workspace. Locates .cortexast.json / .cortex_rules.yml." },
                                "file_path": { "type": "string", "description": "Current file path for context filtering (frontend/backend/db). Rules apply to whole task scope." }
                            },
                            "required": ["project_path"]
                        }
                    },
                    {
                        "name": "cortex_remember",
                        "description": "Save task outcome to permanent global memory. Call at END of every task. intent+decision must be ≤200 chars each. For long artifacts write a file first and pass path via heavy_artifacts.",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "intent": { "type": "string", "description": "User intent summary. Max 200 chars." },
                                "decision": { "type": "string", "description": "Approach taken. Max 200 chars." },
                                "files_touched": { "type": "array", "items": { "type": "string" }, "description": "File paths modified or created." },
                                "tags": { "type": "array", "items": { "type": "string" }, "description": "Semantic labels (e.g. ['auth','refactor'])." },
                                "heavy_artifacts": {
                                    "type": "array",
                                    "description": "Pointers to long-form files (research, qa_log, architecture). Write file first, then pass path here.",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "artifact_type": { "type": "string", "enum": ["research", "qa_log", "architecture", "other"] },
                                            "file_path": { "type": "string" },
                                            "description": { "type": "string", "description": "≤50 char summary." }
                                        },
                                        "required": ["artifact_type", "file_path", "description"]
                                    }
                                }
                            },
                            "required": ["intent", "decision"]
                        }
                    },
                    {
                        "name": "cortex_list_network",
                        "description": "List all AI-tracked codebases (CortexSync network). Use to discover target_project IDs for cross-project operations.",
                        "inputSchema": {
                            "type": "object",
                            "properties": {}
                        }
                    },
                    {
                        "name": "cortex_manage_ast_languages",
                        "description": "Manage Wasm grammar parsers for non-core languages. Core (always active): rust, typescript, python. Call status to see active/available languages. Call add with languages[] to download and hot-reload parsers from GitHub tree-sitter releases. Available: go, php, cpp, c, c_sharp, java, ruby, dart.",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "action": {
                                    "type": "string",
                                    "description": "status: list active and downloadable languages. add: download and hot-reload parser(s).",
                                    "enum": ["status", "add"]
                                },
                                "languages": {
                                    "type": "array",
                                    "items": { "type": "string" },
                                    "description": "Language names to install (e.g. ['go','php','cpp']). Required for action=add."
                                }
                            },
                            "required": ["action"]
                        }
                    },
                    // ── Data Engine ───────────────────────────────────────────────────
                    {
                        "name": "cortex_data_explorer",
                        "description": "Explore and query tabular (CSV/TSV) or plain-text (log/env/txt/md) files without loading them into the AST pipeline. Ideal for quickly previewing schemas, filtering rows, or grepping log files.",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "path": {
                                    "type": "string",
                                    "description": "Absolute or repo-relative path to the data file."
                                },
                                "query": {
                                    "type": "string",
                                    "description": "Optional filter. For CSV/TSV: substring match against any field. For text files: substring match per line."
                                },
                                "max_rows": {
                                    "type": "integer",
                                    "description": "Max rows to show in overview (default 50).",
                                    "default": 50
                                },
                                "max_chars": {
                                    "type": "integer",
                                    "description": "Hard output cap in characters (default 8000).",
                                    "default": 8000
                                }
                            },
                            "required": ["path"]
                        }
                    },
                    {
                        "name": "cortex_get_capabilities",
                        "description": "List all file extensions supported by CortexAST, grouped by engine type (tree_sitter AST, data/CSV, markup/config via tree-sitter, raw text). Use this to quickly check whether a file type is supported before calling other tools.",
                        "inputSchema": {
                            "type": "object",
                            "properties": {},
                            "required": []
                        }
                    },
                ]  // ← end of tools array
            }
        })
    }

    fn tool_call(
        &mut self,
        id: serde_json::Value,
        params: &serde_json::Value,
    ) -> serde_json::Value {
        let name = params.get("name").and_then(|n| n.as_str()).unwrap_or("");
        let args = params.get("arguments").cloned().unwrap_or(json!({}));
        let max_chars = negotiated_max_chars(&args);

        let ok = |text: String| {
            let text = force_inline_truncate(text, max_chars);
            json!({
                "jsonrpc": "2.0",
                "id": id,
                "result": { "content": [{"type":"text","text": text }], "isError": false }
            })
        };

        let err = |msg: String| {
            let msg = force_inline_truncate(msg, max_chars);
            json!({
                "jsonrpc": "2.0",
                "id": id,
                "result": { "content": [{"type":"text","text": msg }], "isError": true }
            })
        };

        match name {
            // ── Megatools ────────────────────────────────────────────────
            "cortex_manage_ast_languages" => {
                let action = args
                    .get("action")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .trim();
                match action {
                    "status" => {
                        let active = crate::inspector::exported_language_config().read().unwrap().active_languages();
                        let available_to_download = vec!["go", "php", "ruby", "java", "c", "cpp", "c_sharp", "dart"];
                        ok(serde_json::to_string(&json!({
                            "active": active,
                            "available_to_download": available_to_download
                        })).unwrap_or_default())
                    }
                    "add" => {
                        let mut loaded_langs = Vec::new();
                        let mut failed_langs = Vec::new();
                        
                        let mut exts_to_invalidate = Vec::new();
                        
                        if let Some(arr) = args.get("languages").and_then(|v| v.as_array()) {
                            let mut cfg = crate::inspector::exported_language_config().write().unwrap();
                            for item in arr {
                                if let Some(lang) = item.as_str() {
                                    if cfg.active_languages().contains(&lang.to_string()) {
                                        loaded_langs.push(lang.to_string());
                                        continue;
                                    }
                                    match cfg.add_wasm_driver(lang) {
                                        Ok(_) => {
                                            loaded_langs.push(lang.to_string());
                                            exts_to_invalidate.extend(cfg.extensions_for_language(lang));
                                        }
                                        Err(e) => {
                                            eprintln!("Failed to add wasm driver for {}: {}", lang, e);
                                            failed_langs.push(lang.to_string());
                                        }
                                    }
                                }
                            }
                        } else {
                            return err("No languages provided for 'add' action".to_string());
                        }

                        let mut invalidated = 0;
                        if !exts_to_invalidate.is_empty() {
                            let repo_root = self.resolve_target_project(&args).unwrap_or_else(|_| std::env::current_dir().unwrap());
                            let cortex_dir = repo_root.join(".cortexast");
                            let db_dir = cortex_dir.join("db");
                            if db_dir.exists() {
                                if let Ok(mut index) = crate::vector_store::CodebaseIndex::open(&repo_root, &db_dir, "nomic-embed-text", 60) {
                                    let refs: Vec<&str> = exts_to_invalidate.iter().map(|s| s.as_str()).collect();
                                    invalidated = index.invalidate_extensions(&refs);
                                }
                            }
                        }

                        ok(serde_json::to_string(&json!({
                            "status": "success",
                            "message": format!(
                                "Successfully downloaded and hot-reloaded parsers: {:?}. Failed: {:?}. Retro-rescan invalidated {} cached records matching extensions: {:?}.", 
                                loaded_langs, failed_langs, invalidated, exts_to_invalidate
                            )
                        })).unwrap_or_default())
                    }
                    _ => err("Invalid action. Must be 'status' or 'add'.".to_string()),
                }
            }
            // ── CortexAct tools have been migrated to the standalone cortex-act binary ──
            "cortex_list_network" => {
                match get_network_map() {
                    Ok(json_data) => ok(serde_json::to_string(&json_data).unwrap_or_default()),
                    Err(e) => err(e),
                }
            }
            "cortex_code_explorer" => {
                let action = args
                    .get("action")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .trim();
                match action {
                    "map_overview" => {
                        let repo_root = match self.resolve_target_project(&args) { Ok(r) => r, Err(e) => return err(e) };
                        let Some(target_str) = args.get("target_dir").and_then(|v| v.as_str()) else {
                            return err(
                                "Error: action 'map_overview' requires the 'target_dir' parameter (e.g. '.' for the whole repo). \
                                Please call cortex_code_explorer again with action='map_overview' and target_dir='.'.".to_string()
                            );
                        };
                        let search_filter = args
                            .get("search_filter")
                            .and_then(|v| v.as_str())
                            .map(|s| s.trim())
                            .filter(|s| !s.is_empty());
                        let max_chars = Some(max_chars);
                        let ignore_gitignore = args.get("ignore_gitignore").and_then(|v| v.as_bool()).unwrap_or(false);
                        let exclude_dirs: Vec<String> = args
                            .get("exclude")
                            .and_then(|v| v.as_array())
                            .map(|arr| {
                                arr.iter()
                                    .filter_map(|x| x.as_str().map(|s| s.to_string()))
                                    .collect()
                            })
                            .unwrap_or_default();
                        let target_dir = resolve_path(&repo_root, target_str);

                        // Proactive guardrail: agents often hallucinate paths.
                        if !target_dir.exists() {
                            let mut entries: Vec<String> = Vec::new();
                            if let Ok(rd) = std::fs::read_dir(&repo_root) {
                                for e in rd.flatten() {
                                    if let Some(name) = e.file_name().to_str() {
                                        entries.push(name.to_string());
                                    }
                                }
                            }
                            entries.sort();
                            let shown: Vec<String> = entries.into_iter().take(30).collect();
                            return err(format!(
                                "Error: Path '{}' does not exist in repo root '{}'.\n\
Available top-level entries in this repo: [{}].\n\
Please correct your target_dir (or pass repoPath explicitly).",
                                target_str,
                                repo_root.display(),
                                shown
                                    .into_iter()
                                    .map(|s| format!("'{}'", s))
                                    .collect::<Vec<_>>()
                                    .join(", ")
                            ));
                        }

                        match repo_map_with_filter(&target_dir, search_filter, max_chars, ignore_gitignore, &exclude_dirs) {
                            Ok(s) => ok(s),
                            Err(e) => err(format!("repo_map failed: {e}")),
                        }
                    }
                    "deep_slice" => {
                        let repo_root = match self.resolve_target_project(&args) { Ok(r) => r, Err(e) => return err(e) };
                        let Some(target_str) = args.get("target").and_then(|v| v.as_str()) else {
                            return err(
                                "Error: action 'deep_slice' requires the 'target' parameter \
                                (relative path to a file or directory within the repo, e.g. 'src'). \
                                Please call cortex_code_explorer again with action='deep_slice' and target='<path>'.".to_string()
                            );
                        };
                        let target = PathBuf::from(target_str);

                        // Proactive path guard: give a "did you mean?" hint when the target
                        // doesn't exist (e.g. agent passes "orchestrator" instead of "orchestrator.rs").
                        {
                            let target_abs = if target.is_absolute() {
                                target.clone()
                            } else {
                                repo_root.join(&target)
                            };
                            if !target_abs.exists() {
                                let stem = target_abs
                                    .file_stem()
                                    .and_then(|s| s.to_str())
                                    .unwrap_or(target_str)
                                    .to_ascii_lowercase();
                                let parent = target_abs.parent().unwrap_or(&repo_root);
                                let search_root = if parent.exists() { parent } else { &repo_root };
                                let mut suggestions: Vec<String> = Vec::new();
                                if let Ok(rd) = std::fs::read_dir(search_root) {
                                    for e in rd.flatten() {
                                        let fname = e.file_name();
                                        let fname_str = fname.to_string_lossy();
                                        if fname_str.to_ascii_lowercase().contains(&stem) {
                                            if let Ok(rel) = e.path().strip_prefix(&repo_root) {
                                                suggestions.push(
                                                    rel.to_string_lossy().replace('\\', "/")
                                                );
                                            }
                                        }
                                    }
                                }
                                suggestions.sort();
                                suggestions.truncate(5);
                                let hint = if suggestions.is_empty() {
                                    String::new()
                                } else {
                                    format!(
                                        "\nDid you mean one of: {}",
                                        suggestions
                                            .iter()
                                            .map(|s| format!("'{s}'"))
                                            .collect::<Vec<_>>()
                                            .join(", ")
                                    )
                                };
                                return err(format!(
                                    "Error: Target '{}' does not exist in repo root '{}'.{hint}\n\
                                    Tip: Use cortex_code_explorer(action='map_overview', target_dir='.') \
                                    to browse the repo structure first.",
                                    target_str,
                                    repo_root.display(),
                                ));
                            }
                        }

                        let budget_tokens = args.get("budget_tokens").and_then(|v| v.as_u64()).unwrap_or(32_000) as usize;
                        let skeleton_only = args.get("skeleton_only").and_then(|v| v.as_bool()).unwrap_or(false);
                        let mut cfg = load_config(&repo_root);

                        // Merge per-call exclude dirs into config so build_scan_options picks them up.
                        if let Some(arr) = args.get("exclude").and_then(|v| v.as_array()) {
                            let extra: Vec<String> = arr
                                .iter()
                                .filter_map(|x| x.as_str().map(|s| s.to_string()))
                                .collect();
                            cfg.scan.exclude_dir_names.extend(extra);
                        }

                        // `single_file=true` bypasses all vector search — returns exactly the
                        // target file/dir without any semantic cross-file expansion.
                        let single_file = args.get("single_file").and_then(|v| v.as_bool()).unwrap_or(false);

                        // `only_dir` scopes vector-search candidates to a subdirectory (poly-repo
                        // support). When combined with `query=`, prevents cross-module spill.
                        let only_dir_path: Option<PathBuf> = args
                            .get("only_dir")
                            .and_then(|v| v.as_str())
                            .filter(|s| !s.is_empty())
                            .map(|s| resolve_path(&repo_root, s));

                        // Optional vector search query (skipped when single_file=true).
                        if !single_file {
                            if let Some(q) = args.get("query").and_then(|v| v.as_str()).filter(|s| !s.is_empty()) {
                                let query_limit = args.get("query_limit").and_then(|v| v.as_u64()).map(|n| n as usize);
                                match self.run_query_slice(&repo_root, &target, only_dir_path.as_deref(), q, query_limit, budget_tokens, skeleton_only, &cfg) {
                                    Ok(xml) => return ok(xml),
                                    Err(e) => return err(format!("query slice failed: {e}")),
                                }
                            }
                        }

                        match slice_to_xml(&repo_root, &target, budget_tokens, &cfg, skeleton_only) {
                            Ok((xml, _meta)) => ok(xml),
                            Err(e) => err(format!("slice failed: {e}")),
                        }
                    }
                    _ => err(format!(
                        "Error: Invalid or missing 'action' for cortex_code_explorer: received '{action}'. \
                        Choose one of: 'map_overview' (repo structure map) or 'deep_slice' (token-budgeted content slice). \
                        Example: cortex_code_explorer with action='map_overview' and target_dir='.'"
                    )),
                }
            }
            "cortex_symbol_analyzer" => {
                let action = args
                    .get("action")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .trim();
                match action {
                    "read_source" => {
                        let repo_root = match self.resolve_target_project(&args) { Ok(r) => r, Err(e) => return err(e) };
                        let Some(p) = args.get("path").and_then(|v| v.as_str()) else {
                            return err(
                                "Error: action 'read_source' requires both 'path' (source file containing the symbol) \
                                and 'symbol_name'. You omitted 'path'. \
                                Please call cortex_symbol_analyzer again with action='read_source', path='<file>', and symbol_name='<name>'. \
                                Tip: use cortex_code_explorer(action=map_overview) first if you are unsure of the file path.".to_string()
                            );
                        };
                        let abs = resolve_path(&repo_root, p);
                        let skeleton_only = args.get("skeleton_only").and_then(|v| v.as_bool()).unwrap_or(false);

                        // Multi-symbol batching: symbol_names: ["A", "B", ...]
                        if let Some(arr) = args.get("symbol_names").and_then(|v| v.as_array()) {
                            let mut out_parts: Vec<String> = Vec::new();
                            for v in arr {
                                let Some(sym) = v.as_str().filter(|s| !s.trim().is_empty()) else { continue };
                                match read_symbol_with_options(&abs, sym, skeleton_only, None) {
                                    Ok(s) => out_parts.push(s),
                                    Err(e) => out_parts.push(format!("// ERROR reading `{sym}`: {e}")),
                                }
                            }
                            if out_parts.is_empty() {
                                return err(
                                    "Error: action 'read_source' with 'symbol_names' requires a non-empty array of symbol name strings. \
                                    You provided an empty array or all entries were blank. \
                                    Example: symbol_names=['process_request', 'handle_error']".to_string()
                                );
                            }
                            return ok(out_parts.join("\n\n"));
                        }

                        let Some(sym) = args.get("symbol_name").and_then(|v| v.as_str()) else {
                            return err(
                                "Error: action 'read_source' requires both 'path' and 'symbol_name'. You omitted 'symbol_name'. \
                                Please call cortex_symbol_analyzer again with action='read_source', path='<file>', and symbol_name='<name>'. \
                                For batch extraction of multiple symbols from the same file, use symbol_names=['A','B'] instead.".to_string()
                            );
                        };
                        let instance_index = args.get("instance_index").and_then(|v| v.as_u64()).map(|n| n as usize);
                        match read_symbol_with_options(&abs, sym, skeleton_only, instance_index) {
                            Ok(s) => ok(s),
                            Err(e) => err(format!("read_symbol failed: {e}")),
                        }
                    }
                    "find_usages" => {
                        let repo_root = match self.resolve_target_project(&args) { Ok(r) => r, Err(e) => return err(e) };
                        let Some(target_str) = args.get("target_dir").and_then(|v| v.as_str()) else {
                            return err(
                                "Error: action 'find_usages' requires both 'symbol_name' and 'target_dir'. You omitted 'target_dir'. \
                                Use '.' to search the entire repo. \
                                Please call cortex_symbol_analyzer again with action='find_usages', symbol_name='<name>', and target_dir='.'.".to_string()
                            );
                        };
                        let Some(sym) = args.get("symbol_name").and_then(|v| v.as_str()) else {
                            return err(
                                "Error: action 'find_usages' requires both 'symbol_name' and 'target_dir'. You omitted 'symbol_name'. \
                                Please call cortex_symbol_analyzer again with action='find_usages', symbol_name='<name>', and target_dir='.'.".to_string()
                            );
                        };
                        let target_dir = resolve_path(&repo_root, target_str);
                        match find_usages(&target_dir, sym) {
                            Ok(s) => ok(s),
                            Err(e) => err(format!("find_usages failed: {e}")),
                        }
                    }
                    "find_implementations" => {
                        let repo_root = match self.resolve_target_project(&args) { Ok(r) => r, Err(e) => return err(e) };
                        let Some(target_str) = args.get("target_dir").and_then(|v| v.as_str()) else {
                            return err(
                                "Error: action 'find_implementations' requires both 'symbol_name' and 'target_dir'. You omitted 'target_dir'. \
                                Use '.' to search the entire repo. \
                                Please call cortex_symbol_analyzer again with action='find_implementations', symbol_name='<name>', and target_dir='.'.".to_string()
                            );
                        };
                        let Some(sym) = args.get("symbol_name").and_then(|v| v.as_str()) else {
                            return err(
                                "Error: action 'find_implementations' requires both 'symbol_name' and 'target_dir'. You omitted 'symbol_name'. \
                                Please call cortex_symbol_analyzer again with action='find_implementations', symbol_name='<name>', and target_dir='.'.".to_string()
                            );
                        };
                        let target_dir = resolve_path(&repo_root, target_str);
                        match find_implementations(&target_dir, sym) {
                            Ok(s) => ok(s),
                            Err(e) => err(format!("find_implementations failed: {e}")),
                        }
                    }
                    "blast_radius" => {
                        let repo_root = match self.resolve_target_project(&args) { Ok(r) => r, Err(e) => return err(e) };
                        let Some(target_str) = args.get("target_dir").and_then(|v| v.as_str()) else {
                            return err(
                                "Error: action 'blast_radius' requires both 'symbol_name' and 'target_dir'. You omitted 'target_dir'. \
                                Use '.' to search the entire repo. \
                                Please call cortex_symbol_analyzer again with action='blast_radius', symbol_name='<name>', and target_dir='.'.".to_string()
                            );
                        };
                        let Some(sym) = args.get("symbol_name").and_then(|v| v.as_str()) else {
                            return err(
                                "Error: action 'blast_radius' requires both 'symbol_name' and 'target_dir'. You omitted 'symbol_name'. \
                                Please call cortex_symbol_analyzer again with action='blast_radius', symbol_name='<name>', and target_dir='.'.".to_string()
                            );
                        };
                        let target_dir = resolve_path(&repo_root, target_str);
                        match call_hierarchy(&target_dir, sym) {
                            Ok(s) => ok(s),
                            Err(e) => err(format!("call_hierarchy failed: {e}")),
                        }
                    }
                    "propagation_checklist" => {
                        let repo_root = match self.resolve_target_project(&args) { Ok(r) => r, Err(e) => return err(e) };
                        // Legacy mode: changed_path checklist (if provided).
                        if let Some(changed_path) = args.get("changed_path").and_then(|v| v.as_str()).map(|s| s.trim()).filter(|s| !s.is_empty()) {
                            let abs = resolve_path(&repo_root, changed_path);
                            let max_symbols = args.get("max_symbols").and_then(|v| v.as_u64()).unwrap_or(20) as usize;

                            let mut out = String::new();
                            out.push_str("Propagation checklist\n");
                            out.push_str(&format!("Changed: {}\n\n", abs.display()));

                            let ext = abs.extension().and_then(|e| e.to_str()).unwrap_or("").to_ascii_lowercase();
                            if ext == "proto" {
                                let raw = std::fs::read_to_string(&abs);
                                if let Ok(text) = raw {
                                    let syms = extract_symbols_from_source(&abs, &text);
                                    if !syms.is_empty() {
                                        out.push_str("Detected contract symbols (sample):\n");
                                        for s in syms.into_iter().take(max_symbols) {
                                            out.push_str(&format!("- [{}] {}\n", s.kind, s.name));
                                        }
                                        out.push('\n');
                                    }
                                }

                                out.push_str("Checklist (Proto → generated clients):\n");
                                out.push_str("- Regenerate Rust stubs (prost/tonic build, buf, or your codegen pipeline)\n");
                                out.push_str("- Regenerate TypeScript/JS clients (grpc-web/connect/buf generate, etc.)\n");
                                out.push_str("- Update server handlers for any renamed RPCs/messages/enums\n");
                                out.push_str("- Run `run_diagnostics` and service-level tests\n\n");
                                out.push_str("Suggested CortexAST probes (fast, AST-accurate):\n");
                                out.push_str("- `cortex_code_explorer` action=map_overview with `search_filter` set to the service/message name\n");
                                out.push_str("- `cortex_symbol_analyzer` action=find_usages for each renamed message/service to find all consumers\n");
                            } else {
                                out.push_str("Checklist (API change propagation):\n");
                                out.push_str("- `cortex_symbol_analyzer` action=find_usages on the changed symbol(s) to locate all call sites\n");
                                out.push_str("- `cortex_symbol_analyzer` action=blast_radius to understand blast radius before refactoring\n");
                                out.push_str("- Update dependent modules/services and re-run `run_diagnostics`\n");
                            }

                            return ok(out);
                        }

                        // New mode: symbol-based cross-boundary checklist.
                        let Some(sym) = args
                            .get("symbol_name")
                            .and_then(|v| v.as_str())
                            .map(|s| s.trim())
                            .filter(|s| !s.is_empty())
                        else {
                            return err(
                                "Error: action 'propagation_checklist' requires 'symbol_name' (the shared type/struct/interface to trace). \
                                You omitted 'symbol_name'. \
                                Please call cortex_symbol_analyzer again with action='propagation_checklist' and symbol_name='<name>'. \
                                Alternatively, pass 'changed_path' (path to a .proto or contract file) for legacy file-based mode.".to_string()
                            );
                        };
                        let target_str = args.get("target_dir").and_then(|v| v.as_str()).unwrap_or(".");
                        let target_dir = resolve_path(&repo_root, target_str);
                        let ignore_gitignore = args.get("ignore_gitignore").and_then(|v| v.as_bool()).unwrap_or(false);

                        // `only_dir` overrides `target_dir` — scopes scan to a single microservice
                        // directory in poly-repo setups without changing the default API surface.
                        let scan_dir = if let Some(od) = args
                            .get("only_dir")
                            .and_then(|v| v.as_str())
                            .filter(|s| !s.is_empty())
                        {
                            resolve_path(&repo_root, od)
                        } else {
                            target_dir
                        };

                        let aliases: Vec<String> = args
                            .get("aliases")
                            .and_then(|v| v.as_array())
                            .map(|arr| {
                                arr.iter()
                                    .filter_map(|v| v.as_str())
                                    .map(|s| s.trim())
                                    .filter(|s| !s.is_empty())
                                    .map(|s| s.to_string())
                                    .collect::<Vec<_>>()
                            })
                            .unwrap_or_default();

                        match propagation_checklist(&scan_dir, sym, &aliases, ignore_gitignore) {
                            Ok(s) => ok(s),
                            Err(e) => err(format!("propagation_checklist failed: {e}")),
                        }
                    }
                    _ => err(format!(
                        "Error: Invalid or missing 'action' for cortex_symbol_analyzer: received '{action}'. \
                        Choose one of: 'read_source' (extract symbol AST), 'find_usages' (trace all call sites), 'find_implementations' (find implementors of a trait/interface), \
                        'blast_radius' (call hierarchy before rename/delete), or 'propagation_checklist' (cross-module update checklist). \
                        Example: cortex_symbol_analyzer with action='find_usages', symbol_name='my_fn', and target_dir='.'"
                    )),
                }
            }
            "cortex_chronos" => {
                let action = args
                    .get("action")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .trim();
                match action {
                    "save_checkpoint" => {
                        let repo_root = match self.repo_root_from_params(&args) { Ok(r) => r, Err(e) => return err(e) };
                        let cfg = load_config(&repo_root);
                        let Some(p) = args.get("path").and_then(|v| v.as_str()) else {
                            return err(
                                "Error: action 'save_checkpoint' requires 'path' (source file), 'symbol_name', and 'semantic_tag'. \
                                You omitted 'path'. \
                                Please call cortex_chronos again with action='save_checkpoint', path='<file>', \
                                symbol_name='<name>', and semantic_tag='pre-refactor' (or any descriptive tag).".to_string()
                            );
                        };
                        let Some(sym) = args.get("symbol_name").and_then(|v| v.as_str()) else {
                            return err(
                                "Error: action 'save_checkpoint' requires 'path', 'symbol_name', and 'semantic_tag'. \
                                You omitted 'symbol_name'. \
                                Please call cortex_chronos again with action='save_checkpoint', path='<file>', \
                                symbol_name='<name>', and semantic_tag='pre-refactor'.".to_string()
                            );
                        };
                        let tag = args
                            .get("semantic_tag")
                            .and_then(|v| v.as_str())
                            .or_else(|| args.get("tag").and_then(|v| v.as_str()))
                            .unwrap_or("");
                        let namespace = args.get("namespace").and_then(|v| v.as_str());
                        match checkpoint_symbol(&repo_root, &cfg, p, sym, tag, namespace) {
                            Ok(s) => ok(s),
                            Err(e) => err(format!("checkpoint_symbol failed: {e}")),
                        }
                    }
                    "list_checkpoints" => {
                        let repo_root = match self.repo_root_from_params(&args) { Ok(r) => r, Err(e) => return err(e) };
                        let cfg = load_config(&repo_root);
                        let namespace = args.get("namespace").and_then(|v| v.as_str());
                        match list_checkpoints(&repo_root, &cfg, namespace) {
                            Ok(s) => ok(s),
                            Err(e) => err(format!("list_checkpoints failed: {e}")),
                        }
                    }
                    "compare_checkpoint" => {
                        let repo_root = match self.repo_root_from_params(&args) { Ok(r) => r, Err(e) => return err(e) };
                        let cfg = load_config(&repo_root);
                        let Some(sym) = args.get("symbol_name").and_then(|v| v.as_str()) else {
                            return err(
                                "Error: action 'compare_checkpoint' requires 'symbol_name', 'tag_a', and 'tag_b'. \
                                You omitted 'symbol_name'. \
                                Please call cortex_chronos again with action='compare_checkpoint', \
                                symbol_name='<name>', tag_a='<before-tag>', and tag_b='<after-tag>'. \
                                Tip: call cortex_chronos(action=list_checkpoints) first to see all available tags.".to_string()
                            );
                        };
                        let Some(tag_a) = args.get("tag_a").and_then(|v| v.as_str()) else {
                            return err(
                                "Error: action 'compare_checkpoint' requires 'symbol_name', 'tag_a', and 'tag_b'. \
                                You omitted 'tag_a' (the 'before' snapshot tag). \
                                Please call cortex_chronos again with action='compare_checkpoint', \
                                symbol_name='<name>', tag_a='<before-tag>', and tag_b='<after-tag>'. \
                                Tip: call cortex_chronos(action=list_checkpoints) to see all available tags.".to_string()
                            );
                        };
                        let Some(tag_b) = args.get("tag_b").and_then(|v| v.as_str()) else {
                            return err(
                                "Error: action 'compare_checkpoint' requires 'symbol_name', 'tag_a', and 'tag_b'. \
                                You omitted 'tag_b' (the 'after' snapshot tag). \
                                Please call cortex_chronos again with action='compare_checkpoint', \
                                symbol_name='<name>', tag_a='<before-tag>', and tag_b='<after-tag>'.".to_string()
                            );
                        };
                        let path = args.get("path").and_then(|v| v.as_str());
                        let namespace = args.get("namespace").and_then(|v| v.as_str());
                        if tag_b.trim() == "__live__" && path.is_none() {
                            return err(
                                "Error: tag_b='__live__' requires 'path' (the source file containing the symbol). \
Please call cortex_chronos again with action='compare_checkpoint', symbol_name='<name>', tag_a='<snapshot-tag>', tag_b='__live__', and path='<file>'.".to_string()
                            );
                        }
                        match compare_symbol(&repo_root, &cfg, sym, tag_a, tag_b, path, namespace) {
                            Ok(s) => ok(s),
                            Err(e) => {
                                let msg = e.to_string();
                                if msg.contains("No checkpoint found") || msg.contains("No checkpoints found") {
                                    err(format!(
                                        "compare_symbol failed: {msg}\n\n\
Tip: run cortex_chronos(action=list_checkpoints) to see valid tag+symbol combinations, then retry.\n\
Common cause: you saved a checkpoint for a different symbol or under a different tag."
                                    ))
                                } else {
                                    err(format!("compare_symbol failed: {msg}"))
                                }
                            }
                        }
                    }
                    "delete_checkpoint" => {
                        let repo_root = match self.repo_root_from_params(&args) { Ok(r) => r, Err(e) => return err(e) };
                        let cfg = load_config(&repo_root);

                        let symbol_name = args
                            .get("symbol_name")
                            .and_then(|v| v.as_str())
                            .map(|s| s.trim())
                            .filter(|s| !s.is_empty());
                        let semantic_tag = args
                            .get("semantic_tag")
                            .and_then(|v| v.as_str())
                            .or_else(|| args.get("tag").and_then(|v| v.as_str()))
                            .map(|s| s.trim())
                            .filter(|s| !s.is_empty());
                        let path = args.get("path").and_then(|v| v.as_str());
                        let namespace = args.get("namespace").and_then(|v| v.as_str());

                        // Allow namespace-only purge (omit symbol_name + semantic_tag to wipe
                        // an entire namespace in one call, e.g. cleaning up a QC run).
                        // Only reject if ALL of: no namespace context AND no filters.
                        let has_namespace = namespace.map(|s| !s.trim().is_empty()).unwrap_or(false);
                        if symbol_name.is_none() && semantic_tag.is_none() && path.is_none() && !has_namespace {
                            return err(
                                "Error: action 'delete_checkpoint' requires at least one filter: 'symbol_name', 'semantic_tag'/'tag', or 'namespace'. \
Provide 'namespace' alone to purge an entire namespace (e.g. namespace='qa-run-1'). \
Call cortex_chronos with action='list_checkpoints' first to see what exists.".to_string(),
                            );
                        }

                        match crate::chronos::delete_checkpoints(&repo_root, &cfg, symbol_name, semantic_tag, path, namespace) {
                            Ok(s) => ok(s),
                            Err(e) => err(format!("delete_checkpoints failed: {e}")),
                        }
                    }
                    _ => err(format!(
                        "Error: Invalid or missing 'action' for cortex_chronos: received '{action}'. \
                        Choose one of: 'save_checkpoint' (snapshot before edit), 'list_checkpoints' (show all snapshots), \
                        'compare_checkpoint' (AST diff after edit), or 'delete_checkpoint' (remove saved checkpoints). \
                        Example: cortex_chronos with action='save_checkpoint', path='src/main.rs', symbol_name='my_fn', and semantic_tag='pre-refactor'"
                    )),
                }
            }

            // Standalone tool
            "run_diagnostics" => {
                let repo_root = match self.repo_root_from_params(&args) {
                    Ok(r) => r,
                    Err(e) => return err(e),
                };
                match run_diagnostics(&repo_root) {
                    Ok(s) => ok(s),
                    Err(e) => err(format!("diagnostics failed: {e}")),
                }
            }

            "cortex_memory_retriever" => {
                let query = match args.get("query").and_then(|v| v.as_str()) {
                    Some(q) if !q.trim().is_empty() => q.trim().to_string(),
                    _ => return err("cortex_memory_retriever requires a non-empty 'query' parameter.".to_string()),
                };
                let top_k = args.get("top_k").and_then(|v| v.as_u64()).map(|n| n as usize).unwrap_or(5).max(1);
                let tag_filter: Vec<String> = args
                    .get("tags")
                    .and_then(|v| v.as_array())
                    .map(|arr| arr.iter().filter_map(|x| x.as_str().map(String::from)).collect())
                    .unwrap_or_default();

                // Load the memory store from the default journal path.
                let store = MemoryStore::from_default();
                if store.entries().is_empty() {
                    return ok(format!(
                        "Memory journal is empty or does not exist yet.\n\
                         Expected location: {}\n\n\
                         Run CortexSync at least once to populate the journal.",
                        crate::memory::default_journal_path().display()
                    ));
                }

                // Embed the query. Load model lazily; graceful fallback to keyword-only on failure.
                let query_vec: Option<Vec<f32>> = StaticModel::from_pretrained(
                    "minishlab/potion-retrieval-32M",
                    None,
                    None,
                    None,
                )
                .ok()
                .map(|m| m.encode_single(&format!("query: {}", query)));

                // Tokenise the raw query for keyword scoring.
                let tokens_owned: Vec<String> = query
                    .split_whitespace()
                    .filter(|t| t.len() >= 2)
                    .map(|t| t.to_lowercase())
                    .collect();
                let tokens: Vec<&str> = tokens_owned.iter().map(String::as_str).collect();

                let project_path_filter = args
                    .get("project_path")
                    .and_then(|v| v.as_str())
                    .filter(|s| !s.trim().is_empty())
                    .map(String::from);

                let results = hybrid_search(
                    &store,
                    query_vec.as_deref(),
                    &tokens,
                    top_k,
                    &tag_filter,
                    project_path_filter.as_deref(),
                );

                if results.is_empty() {
                    return ok("No relevant memory entries found for the given query/tags.".to_string());
                }

                // Serialise results — omit the `vector` field to keep output token-efficient.
                let mut out = format!(
                    "## Memory Search Results\n**Query:** {query}\n**Matches:** {}/{} entries\n\n",
                    results.len(),
                    store.entries().len()
                );
                for (rank, r) in results.iter().enumerate() {
                    let e = &r.entry;
                    out.push_str(&format!(
                        "### #{rank} — score {:.3}\n\
                         - **id**: {}\n\
                         - **timestamp**: {}\n\
                         - **source_ide**: {}\n\
                         - **project**: {}\n\
                         - **intent**: {}\n\
                         - **decision**: {}\n\
                         - **tags**: {}\n\
                         - **tool_calls**: {}\n\
                         - **files_touched**: {}\n\n",
                        r.score,
                        e.id,
                        e.timestamp,
                        e.source_ide,
                        e.project_path,
                        e.intent,
                        e.decision,
                        e.tags.join(", "),
                        e.tool_calls.join(", "),
                        e.files_touched.join(", "),
                        rank = rank + 1,
                    ));
                }
                ok(out)
            }

            "cortex_get_rules" => {
                let project_path = match args.get("project_path").and_then(|v| v.as_str()) {
                    Some(p) if !p.trim().is_empty() => p.trim().to_string(),
                    _ => return err("cortex_get_rules requires a non-empty 'project_path' parameter.".to_string()),
                };
                let file_path_context = args.get("file_path").and_then(|v| v.as_str());

                match get_merged_rules(&project_path, file_path_context) {
                    Ok(merged) => {
                        // Pretty-print as JSON for readability.
                        let json_pretty = serde_json::to_string_pretty(&merged)
                            .unwrap_or_else(|_| merged.to_string());
                        let tiers_desc = format!(
                            "## Merged Rules for `{project_path}`\n\
                             **Tier resolution:** Global → Team → Project (project wins)\n\n\
                             ```json\n{json_pretty}\n```\n"
                        );
                        ok(tiers_desc)
                    }
                    Err(e) => err(format!("cortex_get_rules error: {e}")),
                }
            }

            // ── Compatibility shims (not exposed in tool_list) ───────────
            // Keep these aliases so existing clients don't instantly break.
            "map_repo" => {
                let mut new_args = args.clone();
                if new_args.get("action").is_none() {
                    new_args["action"] = json!("map_overview");
                }
                self.tool_call(
                    id,
                    &json!({ "name": "cortex_code_explorer", "arguments": new_args }),
                )
            }
            "get_context_slice" => {
                let mut new_args = args.clone();
                if new_args.get("action").is_none() {
                    new_args["action"] = json!("deep_slice");
                }
                self.tool_call(
                    id,
                    &json!({ "name": "cortex_code_explorer", "arguments": new_args }),
                )
            }
            "read_symbol" => {
                let mut new_args = args.clone();
                if new_args.get("action").is_none() {
                    new_args["action"] = json!("read_source");
                }
                self.tool_call(
                    id,
                    &json!({ "name": "cortex_symbol_analyzer", "arguments": new_args }),
                )
            }
            "find_usages" => {
                let mut new_args = args.clone();
                if new_args.get("action").is_none() {
                    new_args["action"] = json!("find_usages");
                }
                self.tool_call(
                    id,
                    &json!({ "name": "cortex_symbol_analyzer", "arguments": new_args }),
                )
            }
            "call_hierarchy" => {
                let mut new_args = args.clone();
                if new_args.get("action").is_none() {
                    new_args["action"] = json!("blast_radius");
                }
                self.tool_call(
                    id,
                    &json!({ "name": "cortex_symbol_analyzer", "arguments": new_args }),
                )
            }
            "propagation_checklist" => {
                let mut new_args = args.clone();
                if new_args.get("action").is_none() {
                    new_args["action"] = json!("propagation_checklist");
                }
                self.tool_call(
                    id,
                    &json!({ "name": "cortex_symbol_analyzer", "arguments": new_args }),
                )
            }
            "save_checkpoint" => {
                let mut new_args = args.clone();
                if new_args.get("action").is_none() {
                    new_args["action"] = json!("save_checkpoint");
                }
                self.tool_call(
                    id,
                    &json!({ "name": "cortex_chronos", "arguments": new_args }),
                )
            }
            "list_checkpoints" => {
                let mut new_args = args.clone();
                if new_args.get("action").is_none() {
                    new_args["action"] = json!("list_checkpoints");
                }
                self.tool_call(
                    id,
                    &json!({ "name": "cortex_chronos", "arguments": new_args }),
                )
            }
            "compare_checkpoint" => {
                let mut new_args = args.clone();
                if new_args.get("action").is_none() {
                    new_args["action"] = json!("compare_checkpoint");
                }
                self.tool_call(
                    id,
                    &json!({ "name": "cortex_chronos", "arguments": new_args }),
                )
            }

            // Deprecated (kept for now): skeleton reader
            "read_file_skeleton" => {
                let repo_root = match self.repo_root_from_params(&args) {
                    Ok(r) => r,
                    Err(e) => return err(e),
                };
                let Some(p) = args.get("path").and_then(|v| v.as_str()) else {
                    return err("Missing path".to_string());
                };
                let abs = resolve_path(&repo_root, p);
                match render_skeleton(&abs) {
                    Ok(s) => ok(s),
                    Err(e) => err(format!("skeleton failed: {e}")),
                }
            }

            // ── cortex_remember ─────────────────────────────────────────────
            // Commits a compressed memory entry to CortexSync via POST /api/remember.
            // Must be called at the end of every task.
            "cortex_remember" => {
                let intent = args
                    .get("intent")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                let decision = args
                    .get("decision")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();

                if intent.trim().is_empty() || decision.trim().is_empty() {
                    return err(
                        "cortex_remember: 'intent' and 'decision' are required and must be non-empty."
                            .to_string(),
                    );
                }

                let files_touched: Vec<String> = args
                    .get("files_touched")
                    .and_then(|v| v.as_array())
                    .map(|a| {
                        a.iter()
                            .filter_map(|x| x.as_str().map(String::from))
                            .collect()
                    })
                    .unwrap_or_default();

                let tags: Vec<String> = args
                    .get("tags")
                    .and_then(|v| v.as_array())
                    .map(|a| {
                        a.iter()
                            .filter_map(|x| x.as_str().map(String::from))
                            .collect()
                    })
                    .unwrap_or_default();

                let project_path = self
                    .repo_root
                    .as_ref()
                    .map(|p| p.display().to_string())
                    .unwrap_or_default();

                // Forward heavy_artifacts as-is (already a JSON array from the LLM).
                let heavy_artifacts = args
                    .get("heavy_artifacts")
                    .cloned()
                    .unwrap_or(serde_json::Value::Array(vec![]));

                let payload = serde_json::json!({
                    "intent": intent,
                    "decision": decision,
                    "project_path": project_path,
                    "files_touched": files_touched,
                    "tags": tags,
                    "heavy_artifacts": heavy_artifacts
                });

                match ureq::post("http://127.0.0.1:14333/api/remember").send_json(payload) {
                    Ok(_) => ok(
                        "Memory successfully vectorized and committed to the global ledger."
                            .to_string(),
                    ),
                    // CortexSync is offline — do NOT fail the MCP call. The LLM
                    // has already completed its task; failing here would cause
                    // confusing error noise. Return a warning so the agent knows
                    // memory was not persisted, but the task outcome is unaffected.
                    Err(_) => ok(
                        "[WARNING] CortexSync background daemon is offline. \
                         Memory could not be saved to the vector ledger, \
                         but your task is complete."
                            .to_string(),
                    ),
                }
            }

            // ── Data Engine ──────────────────────────────────────────────────────────
            "cortex_data_explorer" => {
                let path_str = match args.get("path").and_then(|v| v.as_str()) {
                    Some(p) => p.to_string(),
                    None => return err("Missing required parameter: path".to_string()),
                };
                let repo_root = self.repo_root.clone().unwrap_or_else(|| std::path::PathBuf::from("."));
                let abs_path = {
                    let p = std::path::PathBuf::from(&path_str);
                    if p.is_absolute() { p } else { repo_root.join(p) }
                };
                let query_filter = args.get("query").and_then(|v| v.as_str()).map(|s| s.to_string());
                let max_rows = args.get("max_rows").and_then(|v| v.as_u64()).unwrap_or(50) as usize;

                let reg = crate::data_engine::registry();
                match reg.engine_for(&abs_path) {
                    None => err(format!(
                        "No data engine supports this file type: {}",
                        abs_path.extension().and_then(|e| e.to_str()).unwrap_or("(none)")
                    )),
                    Some(engine) => {
                        let result = if query_filter.is_some() {
                            engine.read_target(&abs_path, query_filter.as_deref(), max_chars)
                        } else {
                            engine.get_overview(&abs_path, max_rows)
                        };
                        match result {
                            Ok(text) => ok(text),
                            Err(e) => err(format!("cortex_data_explorer error: {e:#}")),
                        }
                    }
                }
            }

            "cortex_get_capabilities" => {
                use crate::inspector::exported_language_config;
                let cfg = exported_language_config().read().unwrap();
                let ast_langs = cfg.active_languages();
                let mut ast_exts: Vec<String> = ast_langs
                    .iter()
                    .flat_map(|lang| cfg.extensions_for_language(lang))
                    .collect();
                ast_exts.sort();
                ast_exts.dedup();

                let reg = crate::data_engine::registry();
                let mut data_exts: Vec<String> = Vec::new();
                let mut markup_exts: Vec<String> = Vec::new();
                let mut text_exts: Vec<String> = Vec::new();
                for engine in reg.engines() {
                    let exts: Vec<String> = engine.supported_extensions().iter().map(|s| s.to_string()).collect();
                    match engine.name() {
                        "csv"          => data_exts.extend(exts),
                        "tree_sitter"  => markup_exts.extend(exts),
                        _              => text_exts.extend(exts),
                    }
                }

                let caps = serde_json::json!({
                    "tree_sitter": ast_exts,
                    "data": data_exts,
                    "markup": markup_exts,
                    "text": text_exts,
                });
                ok(serde_json::to_string_pretty(&caps).unwrap_or_default())
            }

            _ => err(format!("Tool not found: {name}")),
        }
    }

    /// Run vector-search-based slicing (query mode) from the MCP server.
    #[allow(clippy::too_many_arguments)]
    fn run_query_slice(
        &mut self,
        repo_root: &std::path::Path,
        target: &std::path::Path,
        only_dir: Option<&std::path::Path>,
        query: &str,
        query_limit: Option<usize>,
        budget_tokens: usize,
        skeleton_only: bool,
        cfg: &crate::config::Config,
    ) -> anyhow::Result<String> {
        let mut exclude_dir_names = vec![
            ".git".into(),
            "node_modules".into(),
            "dist".into(),
            "target".into(),
            cfg.output_dir.to_string_lossy().to_string(),
        ];
        exclude_dir_names.extend(cfg.scan.exclude_dir_names.iter().cloned());

        let opts = ScanOptions {
            repo_root: repo_root.to_path_buf(),
            target: target.to_path_buf(),
            max_file_bytes: cfg.token_estimator.max_file_bytes,
            exclude_dir_names,
        };
        let entries = scan_workspace(&opts)?;

        let db_dir = repo_root.join(&cfg.output_dir).join("db");
        let model_id = cfg.vector_search.model.as_str();
        let chunk_lines = cfg.vector_search.chunk_lines;
        let mut index = CodebaseIndex::open(repo_root, &db_dir, model_id, chunk_lines)?;

        let limit = query_limit.unwrap_or_else(|| {
            let budget_based = (budget_tokens / 1_500).clamp(8, 60);
            budget_based
                .min(cfg.vector_search.default_query_limit)
                .max(1)
        });
        let max_candidates = (limit * 12).clamp(80, 400);
        let terms: Vec<String> = query
            .split_whitespace()
            .map(|s| s.trim().to_ascii_lowercase())
            .filter(|s| s.len() >= 2)
            .collect();

        let mut scored: Vec<(i32, usize)> = entries
            .iter()
            .enumerate()
            .map(|(i, e)| {
                let rel = e.rel_path.to_string_lossy().replace('\\', "/");
                (score_path(&rel, &terms), i)
            })
            .collect();
        scored.sort_by(|(sa, ia), (sb, ib)| {
            sb.cmp(sa)
                .then_with(|| entries[*ia].bytes.cmp(&entries[*ib].bytes))
        });

        let mut to_index: Vec<(String, PathBuf)> = Vec::new();
        for (_score, idx) in scored.iter().take(max_candidates) {
            let e = &entries[*idx];
            let rel = e.rel_path.to_string_lossy().replace('\\', "/");
            if matches!(index.needs_reindex_path(&rel, &e.abs_path), Ok(true)) {
                to_index.push((rel, e.abs_path.clone()));
            }
        }

        let jobs: Vec<IndexJob> = to_index
            .par_iter()
            .filter_map(|(rel, abs)| {
                let bytes = std::fs::read(abs).ok()?;
                let content = String::from_utf8(bytes)
                    .unwrap_or_else(|e| String::from_utf8_lossy(e.as_bytes()).to_string());
                Some(IndexJob {
                    rel_path: rel.clone(),
                    abs_path: abs.clone(),
                    content,
                })
            })
            .collect();

        let rt = tokio::runtime::Runtime::new()?;
        let q_owned = query.to_string();
        let mut rel_paths: Vec<String> = rt.block_on(async move {
            let _ = index.index_jobs(&jobs, || {}).await;
            index.search(&q_owned, limit).await.unwrap_or_default()
        });

        // Scope results to `only_dir` when provided, or auto-scope to the target's
        // directory when target is a file — prevents cross-module semantic spill in
        // poly-repo / microservice codebases ("goes wide" issue).
        let scope_prefix: Option<String> = only_dir
            .map(|p| {
                let rel = p.strip_prefix(repo_root).unwrap_or(p);
                let s = rel.to_string_lossy().replace('\\', "/");
                if s.is_empty() { None } else { Some(s) }
            })
            .unwrap_or_else(|| {
                // Auto-scope to target directory when target is a specific file.
                if target.is_file() {
                    let parent = target.parent().unwrap_or(target);
                    let rel = parent.strip_prefix(repo_root).unwrap_or(parent);
                    let s = rel.to_string_lossy().replace('\\', "/");
                    if s.is_empty() { None } else { Some(s) }
                } else {
                    let rel = target.strip_prefix(repo_root).unwrap_or(target);
                    let s = rel.to_string_lossy().replace('\\', "/");
                    if s.is_empty() { None } else { Some(s) }
                }
            });

        if let Some(ref prefix) = scope_prefix {
            rel_paths.retain(|p| p.starts_with(prefix.as_str()));
        }

        let (xml, _meta) = if rel_paths.is_empty() {
            slice_to_xml(repo_root, target, budget_tokens, cfg, skeleton_only)?
        } else {
            slice_paths_to_xml(repo_root, &rel_paths, budget_tokens, cfg, skeleton_only)?
        };
        Ok(xml)
    }
}

/// Resolve a path parameter: if absolute, use as-is; otherwise join to repo_root.
fn resolve_path(repo_root: &std::path::Path, p: &str) -> PathBuf {
    let pb = PathBuf::from(p);
    if pb.is_absolute() {
        pb
    } else {
        repo_root.join(p)
    }
}

fn score_path(rel_path: &str, terms: &[String]) -> i32 {
    let p = rel_path.to_ascii_lowercase();
    let filename = p.rsplit('/').next().unwrap_or(&p);
    let mut score = 0i32;
    for t in terms {
        if filename.contains(t.as_str()) {
            score += 30;
        } else if p.contains(t.as_str()) {
            score += 10;
        }
    }
    score
}

pub fn run_stdio_server(startup_root: Option<PathBuf>) -> Result<()> {
    let stdin = std::io::stdin();
    let mut stdout = std::io::stdout();

    let mut state = ServerState::default();
    // ── Bootstrap repo_root before the first tool call arrives ──────────────
    // Priority (first non-None wins; the MCP initialize handler may overwrite
    // this later with the editor's authoritative root):
    //
    //   1. --root <PATH>  / CORTEXAST_ROOT     — explicit config (always wins)
    //   2. VSCODE_WORKSPACE_FOLDER             — VS Code / Cursor / Windsurf
    //   3. VSCODE_CWD                          — VS Code secondary
    //   4. IDEA_INITIAL_DIRECTORY              — JetBrains IDEs
    //   5. PWD / INIT_CWD (≠ $HOME)            — Zed, Neovim, npm runners
    //
    // This is a best-effort bootstrap only. The MCP `initialize` request
    // (capture_init_root) is the canonical, protocol-level source and will
    // overwrite this value when the editor sends it.
    let home = std::env::var("HOME")
        .or_else(|_| std::env::var("USERPROFILE"))
        .unwrap_or_default();
    let env_root = std::env::var("CORTEXAST_ROOT")
        .ok()
        .or_else(|| std::env::var("VSCODE_WORKSPACE_FOLDER").ok())
        .or_else(|| std::env::var("VSCODE_CWD").ok())
        .or_else(|| std::env::var("IDEA_INITIAL_DIRECTORY").ok())
        .or_else(|| {
            std::env::var("PWD")
                .ok()
                .filter(|v| v.trim() != home.trim())
        })
        .or_else(|| {
            std::env::var("INIT_CWD")
                .ok()
                .filter(|v| v.trim() != home.trim())
        })
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .map(PathBuf::from);
    if let Some(r) = startup_root.or(env_root) {
        state.repo_root = Some(r);
    }

    for line in stdin.lock().lines() {
        let Ok(line) = line else { continue };
        if line.trim().is_empty() {
            continue;
        }

        let msg: serde_json::Value = match serde_json::from_str(&line) {
            Ok(v) => v,
            Err(_) => continue,
        };

        // JSON-RPC notifications have no "id" field — don't respond.
        let has_id = msg.get("id").is_some();
        if !has_id {
            // Side-effect-only notifications (initialize ack, cancel, log, etc.) — ignore.
            continue;
        }

        let id = msg.get("id").cloned().unwrap_or(json!(null));
        let method = msg.get("method").and_then(|m| m.as_str()).unwrap_or("");

        let reply = match method {
            "initialize" => {
                // Capture workspace root from VS Code's initialize params so subsequent
                // tool calls without repoPath resolve to the correct directory.
                if let Some(p) = msg.get("params") {
                    state.capture_init_root(p);
                }
                json!({
                    "jsonrpc": "2.0",
                    "id": id,
                    "result": {
                        "protocolVersion": msg.get("params").and_then(|p| p.get("protocolVersion")).cloned().unwrap_or(json!("2024-11-05")),
                        "capabilities": { "tools": { "listChanged": true } },
                        "serverInfo": { "name": "cortexast", "version": env!("CARGO_PKG_VERSION") }
                    }
                })
            }
            "ping" => json!({
                "jsonrpc": "2.0",
                "id": id,
                "result": {}
            }),
            "tools/list" => state.tool_list(id),
            "tools/call" => {
                let params = msg.get("params").cloned().unwrap_or(json!({}));
                state.tool_call(id, &params)
            }
            // Return empty lists for resources/prompts — we don't implement them.
            "resources/list" => json!({
                "jsonrpc": "2.0",
                "id": id,
                "result": { "resources": [] }
            }),
            "prompts/list" => json!({
                "jsonrpc": "2.0",
                "id": id,
                "result": { "prompts": [] }
            }),
            _ => json!({
                "jsonrpc": "2.0",
                "id": id,
                "error": { "code": -32601, "message": format!("Method not found: {method}") }
            }),
        };

        writeln!(stdout, "{}", reply)?;
        stdout.flush()?;
    }

    Ok(())
}

const DEFAULT_MAX_CHARS: usize = 8_000;

fn negotiated_max_chars(args: &serde_json::Value) -> usize {
    args.get("max_chars")
        .and_then(|v| v.as_u64())
        .map(|n| n as usize)
        .filter(|n| *n > 0)
        .unwrap_or(DEFAULT_MAX_CHARS)
}

/// Hard inline cap: always truncates in the response body — never writes to disk.
/// Safe for any MCP client; the truncation marker makes partial output obvious.
fn force_inline_truncate(mut content: String, max_chars: usize) -> String {
    if content.len() <= max_chars {
        return content;
    }
    let total_len = content.len();
    let mut cut = max_chars.min(content.len());
    while cut > 0 && !content.is_char_boundary(cut) {
        cut -= 1;
    }
    content.truncate(cut);
    content.push_str(&format!(
        "\n\n... ✂️ [TRUNCATED: {max_chars}/{total_len} chars to prevent IDE spill]"
    ));
    content
}
