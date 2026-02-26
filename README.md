# CortexAST 🧠⚡

> **The AI-Native Code Intelligence Backend for LLM Agents**
> Pure Rust · MCP Server · Semantic Code Navigation · AST Time Machine · Self-Evolving Wasm Parsers

[![Rust](https://img.shields.io/badge/rust-1.80%2B-orange?logo=rust)](https://www.rust-lang.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)
[![Version](https://img.shields.io/badge/version-2.1.0-blue)](./CHANGELOG.md)

---

## What is CortexAST?

CortexAST is a **production-grade MCP (Model Context Protocol) server** that gives AI coding agents (Claude, Gemini, GPT-4o, etc.) the ability to:

- **Navigate codebases semantically** — find symbols, blast-radius analysis, cross-file propagation checklists
- **Edit code surgically** — byte-accurate AST patching with a built-in Tree-sitter validator and Auto-Healer  
- **Evolve itself** — download and hot-reload WebAssembly language parsers at runtime (Go, PHP, Ruby, Java, …)
- **Run async jobs** — spawn background shell commands and poll results without MCP timeout

---

## Feature Modules

### 🔭 cortex_code_explorer
Bird's-eye symbol map (`map_overview`) and token-budgeted XML slice (`deep_slice`) of any codebase.

### 🎯 cortex_symbol_analyzer
AST-accurate `read_source`, `find_usages`, `blast_radius`, and `propagation_checklist` — no grep false positives.

### ⏳ cortex_chronos
Save/compare/rollback named AST snapshots. Detects semantic regressions that `git diff` hides.

### 🧠 cortex_memory_retriever / cortex_remember
Global memory journal (`~/.cortexast/global_memory.jsonl`) with vector-semantic recall.

### 🌐 cortex_manage_ast_languages — Self-Evolving Agent
Download and **hot-reload Wasm parsers** at runtime. No restart required.
```json
{ "action": "add", "languages": ["go", "java"] }
```

### ✏️ cortex_act_edit_ast — Semantic Patcher (v2.1.0)
Three-phase commit: **dry-run → validate (Tree-sitter) → commit to disk**.
- Targets semantic nodes by name (`function:login`, `class:Auth`)
- **Bottom-up byte sorting** — multiple edits never corrupt each other's offsets
- **Auto-Healer** bridge to local LM Studio/Ollama (10 s hard timeout, markdown sanitization, TS error context injection)
- **Write-permission guard** before any disk operation

### ⚙️ cortex_act_edit_config
Dot-path key-value surgery on `.json`, `.yaml`, `.toml` files.  
`"path": "dependencies.express"`, `"value": "^5.0.0"` — no full-file rewrite.

### 📄 cortex_act_edit_docs
Replace any `## Section` in a Markdown file without touching the rest of the document.

### ⏳ cortex_act_run_async / cortex_check_job
Spawn shell commands as background jobs. Poll by `job_id`. Avoids MCP timeouts on `cargo build`, `npm install`, etc.

---

## Quick Start

### Prerequisites
- Rust 1.80+
- Ollama or [LM Studio](https://lmstudio.ai) running locally (optional, for Auto-Healer)

### Build & Run
```bash
git clone https://github.com/DevsHero/CortexAST
cd CortexAST
cargo build --release

# Run as MCP server (stdio)
./target/release/cortexast
```

### MCP Config (`~/.cursor/mcp.json` or Claude Desktop)
```json
{
  "mcpServers": {
    "cortexast": {
      "command": "/path/to/cortexast",
      "args": []
    }
  }
}
```

---

## CortexAct: Patching Examples

### AST Edit — Replace a Function Body
```json
{
  "name": "cortex_act_edit_ast",
  "arguments": {
    "file": "/src/auth.rs",
    "edits": [
      {
        "target": "function:login",
        "action": "replace",
        "code": "pub fn login(user: &str, pass: &str) -> Result<Token> { Ok(Token::new()) }"
      }
    ]
  }
}
```

### Config Patch — Update a Dependency
```json
{
  "name": "cortex_act_edit_config",
  "arguments": {
    "file": "package.json",
    "action": "set",
    "path": "dependencies.express",
    "value": "^5.0.0"
  }
}
```

### Async Job — Run Cargo Test
```json
{
  "name": "cortex_act_run_async",
  "arguments": { "command": "cargo test --workspace", "cwd": "/src/myproject" }
}
```
Then poll: `{ "name": "cortex_check_job", "arguments": { "job_id": "job_abc123" } }`

---

## Auto-Healer Architecture

```
AST Edit Request
      │
      ▼
 Permission Guard ──✗──► Clear Error (chmod advice)
      │ ✓
      ▼
 In-Memory Patch (bottom-up bytes)
      │
      ▼
 Tree-sitter Validator
      │ ✓                   ✗
      ▼                     ▼
 Write to Disk        collect_ts_errors()
      │                     │
      ▼                Numbered error list
   Done              + broken code block
                            │
                            ▼
                    Local LLM (LM Studio)
                    10s hard timeout
                    Strict system prompt
                            │
                      sanitize_llm_code()
                            │
                    Tree-sitter Re-Validate
                     ✓            ✗
                     │             │
                Write to Disk   Bail (safe abort)
```

---

## Self-Evolving Wasm Language Support

| Always Available | Downloadable on Demand |
|---|---|
| Rust, TypeScript/JS, Python | Go, PHP, Ruby, Java, C, C++, C#, Dart |

```bash
# Agent calls this automatically when it detects a new language:
cortex_manage_ast_languages { "action": "add", "languages": ["go", "dart"] }
```

---

## Development

```bash
# Run all unit tests
cargo test

# Check (no link)
cargo check

# Run tests for act/ modules specifically
cargo test act::
```

### Test Coverage (act/ modules)

| Test | What it proves |
|---|---|
| `bottom_up_sort_preserves_byte_offsets` | Multiple replacements never corrupt offsets |
| `top_down_order_corrupts_offsets` | Demonstrates the failure mode we prevent |
| `ts_error_collection_on_broken_rust` | AST walker collects meaningful error messages |
| `permission_guard_catches_readonly` | Fails fast on `chmod 444` before any edit |
| `permission_guard_passes_for_writable` | Normal files pass through |
| `sanitize_strips_rust_fence` | Strips ```rust ... ``` from LLM response |
| `sanitize_multiple_blocks_joined` | Multi-block LLM responses preserved |
| `error_context_format_test` | Numbered error list format for LLM |

---

## Architecture

```
cortexast (binary)
└── src/
    ├── server.rs         # MCP stdio server — all tool schemas + handlers
    ├── inspector.rs      # LanguageConfig, LanguageDriver, Symbol, run_query
    ├── act/
    │   ├── editor.rs     # cortex_act_edit_ast — Two-Phase Commit + Auto-Healer
    │   ├── auto_healer.rs # LM Studio bridge, sanitize_llm_code
    │   ├── config_patcher.rs # JSON/YAML/TOML dot-path patcher
    │   ├── docs_patcher.rs   # Markdown section editor
    │   └── job_manager.rs    # Async job spawner + polling
    ├── grammar_manager.rs    # Wasm download + hot-reload
    ├── vector_store.rs       # model2vec embeddings + cache invalidation
    ├── chronos.rs            # AST snapshot time machine
    └── memory.rs             # global_memory.jsonl journal
```

---

## License

MIT — See [LICENSE](./LICENSE)

---

*Built with ❤️ in Rust · Semantic precision for the AI age*