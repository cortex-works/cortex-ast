# Changelog

All notable changes to **CortexAST** are documented here.  
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).  
Versioning follows [Semantic Versioning](https://semver.org/).

---

## [2.1.0] — 2026-02-26

### Added — CortexAct: Source Code Editing & Auto-Healing Engine

#### Module 1 — `cortex_act_edit_ast` (AST Semantic Patcher)
- **Two-Phase Commit** with in-memory Virtual Dry-Run before any disk write.
- **Bottom-up byte sorting**: multiple edits are sorted by `start_byte` descending before application, guaranteeing earlier replacements never corrupt the byte offsets of later target nodes.
- **Write Permission Guard** (`check_write_permission`): verifies both `metadata().readonly()` and a live `OpenOptions::write` before touching the file; catches Unix ACL denials missed by permission bits. Error message names the expected user (`zelda`) and the `chmod` command to fix it.
- **Tree-sitter Validator**: after patching the in-memory buffer, parse with the shared `RwLock<LanguageConfig>` / `WasmStore`. Any `ERROR` or `MISSING` node triggers the Auto-Healer instead of a disk write.
- **`collect_ts_errors`**: walks the full AST cursor and produces a human-readable numbered list of error positions (row:col + snippet) to supply as context to the local LLM.

#### Module 2 — Auto-Healer (`auto_healer.rs`)
- Bridges to a local LLM endpoint (default: `http://127.0.0.1:1234/v1/chat/completions`; override via `llm_url`).
- **10-second hard timeout** via `ureq::AgentBuilder.timeout()` — prevents MCP Timebomb.
- **Context-aware prompt**: injects the numbered Tree-sitter error list so small models (e.g. `lfm2-2.6b`) know exactly which line/token to fix.
- **`sanitize_llm_code`**: strips residual ` ``` ` markdown fences and language tags from LLM responses before passing repaired code to the second Tree-sitter validation pass.
- Strict system prompt: `"Output ONLY raw code -- no markdown, no backticks, no explanations."`

#### Module 3 — `cortex_act_edit_config`
- Surgically modify a single key in `.json`, `.yaml`, or `.toml` using dot-path notation (e.g. `dependencies.express`).
- Supports `set` and `delete` actions without rewriting the whole file.

#### Module 4 — `cortex_act_edit_docs`
- Replace any `## Section` in a Markdown file, identified by heading level + heading text.
- Preserves all surrounding sections; configurable `heading_level` (default: `##`).

#### Module 5 — `cortex_act_run_async` + `cortex_check_job`
- Spawn shell commands as background threads. Returns `job_id` immediately (no MCP timeout).
- Poll via `cortex_check_job { "job_id": "..." }` for status, exit code, stdout, and stderr.
- Background threads auto-detect timeout via `Instant::elapsed()`.

#### Inspector (`inspector.rs`)
- `Symbol` struct gains `start_byte` and `end_byte` fields — exposed from `run_query` via `def_node` so callers gain byte-accurate ranges for patching.
- `driver_for_path` made `pub` to allow `act::editor` to reuse the shared Wasm engine.

#### Unit Tests Added (`src/act/`)
- `editor::tests::bottom_up_sort_preserves_byte_offsets` — proves bottom-up sorting correctness.
- `editor::tests::top_down_order_corrupts_offsets` — demonstrates the failure mode prevented.
- `editor::tests::ts_error_collection_on_broken_rust` — validates AST error walker output.
- `editor::tests::permission_guard_catches_readonly` — verifies `chmod 444` detection.
- `editor::tests::permission_guard_passes_for_writable` — happy path.
- `auto_healer::tests::sanitize_*` — 5 sanitizer tests (fence stripping, multi-block, passthrough, in-code ``` preservation, numbered error format).

---

## [2.0.4] — 2026-02-25

### Added — Self-Evolving AST Manager (`cortex_manage_ast_languages`)
- `status` action: reports active and downloadable languages.
- `add` action: downloads `.wasm` + `[lang]_prune.scm` files from CDN.
- **Hot-reload**: dynamically instantiates downloaded parsers into `RwLock<LanguageConfig>` without server restart.
- **Retroactive rescan**: calls `CodebaseIndex::invalidate_extensions` to purge stale vector cache entries for newly added language extensions.
- `WasmDriver` gains `make_parser()` method to correctly attach `WasmStore` per parser instance.

### Changed
- `LanguageConfig` refactored to `RwLock<LanguageConfig>` for concurrent read / exclusive write.
- Startup now scans `~/.cortex-works/grammars/` to load previously downloaded Wasm parsers.

---

## [2.0.0] — 2026-02-23

### Added — Wasm Plugin Architecture
- Modular backend: languages loaded as WebAssembly grammars for small binary size.
- `grammar_manager.rs`: CDN fetcher for `.wasm` and `.scm` grammar files.
- Universal regex fallback for unsupported extensions.
- Static languages (Rust, TypeScript, Python) always available.
- `WasmDriver` trait implementation for dynamic language loading.

---

## [1.x] — Earlier

Initial implementation of:
- `cortex_code_explorer` (map_overview + deep_slice)
- `cortex_symbol_analyzer` (read_source, find_usages, blast_radius, propagation_checklist)
- `cortex_chronos` (AST snapshot save/compare/rollback)
- `cortex_memory_retriever` + `cortex_remember` (global memory journal)
- `vector_store.rs` (model2vec embeddings, cosine search, cache invalidation)
- `cortex_diagnostics` (compiler error mapping)
- `cortex_list_network` (cross-project network map)
