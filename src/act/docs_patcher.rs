use anyhow::{Context, Result};

/// Replace the body of a named Markdown section (identified by heading level + text).
/// Finds `## Section Name` and replaces everything up to the next heading of same or higher level.
pub fn patch_docs(
    file: &str,
    section: &str,
    new_content: &str,
    heading_level: usize,
) -> Result<String> {
    let raw = std::fs::read_to_string(file).context("Reading Markdown file")?;
    let prefix = "#".repeat(heading_level.clamp(1, 6));
    let target_heading = format!("{} {}", prefix, section);

    let lines: Vec<&str> = raw.lines().collect();

    // Find the start line (the heading line itself)
    let start_idx = lines
        .iter()
        .position(|l| l.trim() == target_heading.trim())
        .with_context(|| {
            format!(
                "Section '{}' (level {}) not found in {}",
                section, heading_level, file
            )
        })?;

    // Find the end: next heading of same or higher level (fewer or equal # chars)
    let end_idx = lines
        .iter()
        .enumerate()
        .skip(start_idx + 1)
        .find(|(_, l)| {
            let trimmed = l.trim_start_matches('#');
            let hashes = l.len() - trimmed.len();
            hashes > 0 && hashes <= heading_level && l.starts_with('#')
        })
        .map(|(i, _)| i)
        .unwrap_or(lines.len());

    // Reassemble: heading + new content + rest
    let mut out_lines: Vec<&str> = Vec::new();
    out_lines.extend_from_slice(&lines[..=start_idx]);
    // Add blank line after heading if new_content doesn't start with one
    if !new_content.starts_with('\n') {
        out_lines.push("");
    }
    // We push the new content lines
    let content_lines: Vec<&str> = new_content.lines().collect();
    out_lines.extend_from_slice(&content_lines);
    // Blank line before next section
    if end_idx < lines.len() {
        out_lines.push("");
        out_lines.extend_from_slice(&lines[end_idx..]);
    }

    let result = out_lines.join("\n");
    std::fs::write(file, &result).context("Writing Markdown file")?;
    Ok(format!(
        "✅ Replaced section '{}' in '{}' ({} lines → {} lines)",
        section,
        file,
        end_idx - start_idx - 1,
        content_lines.len()
    ))
}
