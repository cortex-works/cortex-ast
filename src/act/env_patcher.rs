use anyhow::{Context, Result};

/// Extremely lightweight key-value patcher for .env files.
/// Preserves surrounding comments and whitespace via regex replacement.
pub fn patch_env(file: &str, action: &str, key: &str, value: Option<&str>) -> Result<String> {
    let raw = std::fs::read_to_string(file).context("Reading .env file")?;
    let mut lines: Vec<String> = raw.lines().map(|s| s.to_string()).collect();

    // Fast regex for .env matching
    let env_regex = regex::Regex::new(&format!(r#"^\s*{}\s*="#, regex::escape(key)))
        .context("Building regex")?;

    match action {
        "set" => {
            let v = value.context("'value' required for 'set' action")?;
            let new_line = format!("{}={}", key, v);

            let mut replaced = false;
            for line in lines.iter_mut() {
                if env_regex.is_match(line) {
                    *line = new_line.clone();
                    replaced = true;
                    break;
                }
            }
            if !replaced {
                if !lines.is_empty() && !lines.last().unwrap().is_empty() {
                    // Add newline if file doesn't end with empty line already implicitly
                }
                lines.push(new_line);
            }
        }
        "delete" => {
            lines.retain(|l| !env_regex.is_match(l));
        }
        other => anyhow::bail!("Unknown action: {}", other),
    }

    let result = lines.join("\n") + "\n";
    std::fs::write(file, &result).context("Writing .env file")?;
    
    Ok(format!("✅ Patched .env '{}' for key '{}'", file, key))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::NamedTempFile;

    #[test]
    fn test_patch_env() {
        let file = NamedTempFile::new().unwrap();
        let path = file.path().to_str().unwrap();
        fs::write(path, "FOO=bar\n  BAZ=123  \n# comment").unwrap();

        patch_env(path, "set", "FOO", Some("baz")).unwrap();
        assert!(fs::read_to_string(path).unwrap().contains("FOO=baz"));

        patch_env(path, "delete", "BAZ", None).unwrap();
        assert!(!fs::read_to_string(path).unwrap().contains("BAZ="));

        patch_env(path, "set", "NEW_KEY", Some("hello")).unwrap();
        assert!(fs::read_to_string(path).unwrap().contains("NEW_KEY=hello"));
    }
}
