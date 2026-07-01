use std::fs;
use std::net::TcpListener;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::OnceLock;
use std::time::{Duration, Instant};

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use tokio::process::{Child, Command};
use tokio::sync::Mutex;
use tokio::time;
use uuid::Uuid;

static BROWSER_SESSION: OnceLock<Mutex<Option<BrowserSession>>> = OnceLock::new();

#[derive(Clone, Debug, Deserialize, JsonSchema, Serialize)]
pub struct BrowserControlArgs {
    /// JavaScript to run inside an async function with `browser`, `context`, and `page` available.
    /// Use normal Playwright APIs, e.g. `await page.goto("https://example.com"); return await page.title();`.
    #[serde(default)]
    pub javascript: String,
    /// Optional URL to open before running the JavaScript when the copied-profile browser starts on about:blank.
    /// Omit this on follow-up calls to keep the browser exactly where the previous call left it.
    #[serde(default)]
    pub url: Option<String>,
    /// Maximum time in seconds for the Playwright script.
    #[serde(default = "default_timeout")]
    pub timeout: u64,
    /// Close the persistent browser session after this script runs. Set this to true once the task is complete.
    #[serde(default)]
    pub close: bool,
    /// Discard any existing persistent browser session and start from a fresh copy of the Chrome Default profile.
    #[serde(default)]
    pub reset: bool,
    /// Show the browser window. Leave false unless the user directly asks to see it.
    #[serde(default)]
    pub visible: bool,
}

pub async fn browser_control(args: BrowserControlArgs) -> Result<String, String> {
    let playwright = find_playwright().ok_or_else(|| {
        "playwright is not installed or not found in PATH; browser_control cannot work. Install Playwright globally (for example: npm install -g playwright) and retry.".to_string()
    })?;
    let node = find_executable("node").ok_or_else(|| {
        "node is not installed or not found in PATH; browser_control requires Node.js to run Playwright.".to_string()
    })?;

    let session_mutex = BROWSER_SESSION.get_or_init(|| Mutex::new(None));
    let mut session_guard = session_mutex.lock().await;

    if args.reset
        && let Some(session) = session_guard.take()
    {
        close_session(session).await;
    }

    let needs_start = match session_guard.as_mut() {
        Some(session) => {
            session.visible != args.visible
                || session
                    .child
                    .try_wait()
                    .map_err(|err| format!("failed to inspect Chrome process: {err}"))?
                    .is_some()
        }
        None => true,
    };

    if needs_start {
        if let Some(session) = session_guard.take() {
            close_session(session).await;
        }
        *session_guard = Some(start_session(args.url.as_deref(), args.visible).await?);
    }

    let Some(session) = session_guard.as_ref() else {
        return Err("failed to create browser session".to_string());
    };

    let script_path = session
        .temp
        .path
        .join(format!("browser-control-{}.js", Uuid::new_v4()));
    let script = playwright_script(
        &playwright,
        session.port,
        args.url.as_deref(),
        &args.javascript,
        args.close,
    );
    fs::write(&script_path, script)
        .map_err(|err| format!("failed to write Playwright control script: {err}"))?;

    let output = match time::timeout(
        Duration::from_secs(args.timeout),
        Command::new(node)
            .arg(&script_path)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .output(),
    )
    .await
    {
        Ok(result) => result.map_err(|err| format!("failed to run Playwright script: {err}"))?,
        Err(_) => {
            if args.close
                && let Some(session) = session_guard.take()
            {
                close_session(session).await;
            }
            return Err(format!(
                "Playwright browser_control timed out after {}s",
                args.timeout
            ));
        }
    };

    let _ = fs::remove_file(&script_path);

    let mut combined = String::new();
    combined.push_str(&String::from_utf8_lossy(&output.stdout));
    combined.push_str(&String::from_utf8_lossy(&output.stderr));

    if args.close
        && let Some(session) = session_guard.take()
    {
        close_session(session).await;
    }

    if output.status.success() {
        Ok(if combined.trim().is_empty() {
            if args.close {
                "Playwright script completed successfully with no output. Browser session closed."
                    .to_string()
            } else {
                "Playwright script completed successfully with no output. Browser session remains open; call browser_control with close=true when done.".to_string()
            }
        } else {
            append_session_status(combined, args.close)
        })
    } else {
        let code = output.status.code().unwrap_or(1);
        if !combined.is_empty() && !combined.ends_with('\n') {
            combined.push('\n');
        }
        combined.push_str(&format!("(exit code: {code})"));
        Err(combined)
    }
}

async fn start_session(url: Option<&str>, visible: bool) -> Result<BrowserSession, String> {
    let chrome = chrome_path()?;
    let profile = chrome_profile_dir()?;

    let temp = TempDirGuard::new()?;
    let user_data_dir = temp.path.join("chrome-user-data");
    let copied_default = user_data_dir.join("Default");
    fs::create_dir_all(&user_data_dir)
        .map_err(|err| format!("failed to create temporary Chrome profile: {err}"))?;
    copy_dir_recursive(&profile, &copied_default)
        .map_err(|err| format!("failed to copy Chrome Default profile: {err}"))?;
    copy_local_state(&profile, &user_data_dir)?;

    let port = available_port()?;
    let mut child = launch_chrome(&chrome, &user_data_dir, port, url, visible)?;

    if let Err(err) = wait_for_cdp(port, &mut child).await {
        let _ = child.kill().await;
        return Err(err);
    }

    Ok(BrowserSession {
        port,
        child,
        temp,
        visible,
    })
}

async fn close_session(mut session: BrowserSession) {
    let _ = session.child.kill().await;
}

fn append_session_status(mut output: String, closed: bool) -> String {
    if !output.ends_with('\n') {
        output.push('\n');
    }
    if closed {
        output.push_str("[browser session closed]");
    } else {
        output.push_str(
            "[browser session remains open; call browser_control with close=true when done]",
        );
    }
    output
}

fn launch_chrome(
    chrome: &Path,
    user_data_dir: &Path,
    port: u16,
    url: Option<&str>,
    visible: bool,
) -> Result<Child, String> {
    let mut command = Command::new(chrome);
    command
        .args(chrome_launch_args(user_data_dir, port, url, visible))
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .kill_on_drop(true);
    command
        .spawn()
        .map_err(|err| format!("failed to launch Chrome: {err}"))
}

fn chrome_launch_args(
    user_data_dir: &Path,
    port: u16,
    url: Option<&str>,
    visible: bool,
) -> Vec<String> {
    let mut args = vec![
        format!("--user-data-dir={}", user_data_dir.display()),
        "--profile-directory=Default".to_string(),
        format!("--remote-debugging-port={port}"),
        "--remote-debugging-address=127.0.0.1".to_string(),
        "--no-first-run".to_string(),
        "--no-default-browser-check".to_string(),
    ];
    if !visible {
        args.push("--headless=new".to_string());
        args.push("--disable-gpu".to_string());
    }
    if let Some(url) = url {
        args.push(url.to_string());
    }
    args
}

async fn wait_for_cdp(port: u16, child: &mut Child) -> Result<(), String> {
    let url = format!("http://127.0.0.1:{port}/json/version");
    let deadline = Instant::now() + Duration::from_secs(15);
    while Instant::now() < deadline {
        if let Some(status) = child
            .try_wait()
            .map_err(|err| format!("failed to inspect Chrome process: {err}"))?
        {
            return Err(format!(
                "Chrome exited before DevTools became available (status: {status})"
            ));
        }
        if let Ok(response) = reqwest::get(&url).await
            && response.status().is_success()
        {
            return Ok(());
        }
        time::sleep(Duration::from_millis(200)).await;
    }
    Err("Chrome DevTools endpoint did not become available within 15s".to_string())
}

fn playwright_script(
    playwright: &Path,
    port: u16,
    url: Option<&str>,
    javascript: &str,
    close: bool,
) -> String {
    let package_dir = playwright_package_dir(playwright);
    let package_dir_json =
        serde_json::to_string(&package_dir.map(|path| path.display().to_string()))
            .unwrap_or_else(|_| "null".to_string());
    let endpoint_json = serde_json::to_string(&format!("http://127.0.0.1:{port}"))
        .unwrap_or_else(|_| "null".to_string());
    let url_json = serde_json::to_string(&url).unwrap_or_else(|_| "null".to_string());
    let close_json = serde_json::to_string(&close).unwrap_or_else(|_| "false".to_string());

    format!(
        r#"const packageDir = {package_dir_json};
const endpoint = {endpoint_json};
const initialUrl = {url_json};
const closeBrowser = {close_json};
function loadPlaywright() {{
  try {{ return require('playwright'); }} catch (firstError) {{
    if (packageDir) {{ return require(packageDir); }}
    throw firstError;
  }}
}}
(async () => {{
  const playwright = loadPlaywright();
  const browser = await playwright.chromium.connectOverCDP(endpoint);
  const context = browser.contexts()[0] || await browser.newContext();
  const pages = context.pages();
  const page = pages[0] || await context.newPage();
  if (initialUrl && page.url() === 'about:blank') {{
    await page.goto(initialUrl, {{ waitUntil: 'domcontentloaded' }});
  }}
  const result = await (async ({{ browser, context, page, playwright }}) => {{
{javascript}
  }})({{ browser, context, page, playwright }});
  if (result !== undefined) {{
    if (typeof result === 'string') console.log(result);
    else console.log(JSON.stringify(result, null, 2));
  }}
  if (closeBrowser) {{
    await browser.close();
  }}
}})().catch(error => {{
  console.error(error && error.stack ? error.stack : String(error));
  process.exit(1);
}});
"#
    )
}

fn playwright_package_dir(playwright: &Path) -> Option<PathBuf> {
    let mut current = fs::canonicalize(playwright).ok()?;
    if current.is_file() {
        current.pop();
    }
    loop {
        if current.file_name().is_some_and(|name| name == "playwright")
            && current.join("package.json").is_file()
        {
            return Some(current);
        }
        if !current.pop() {
            return None;
        }
    }
}

fn chrome_path() -> Result<PathBuf, String> {
    if let Ok(path) = std::env::var("AGENT_BROWSER_CHROME_PATH") {
        let path = PathBuf::from(path);
        if path.is_file() {
            return Ok(path);
        }
        return Err(format!(
            "AGENT_BROWSER_CHROME_PATH does not point to a file: {}",
            path.display()
        ));
    }

    chrome_candidates()
        .into_iter()
        .find(|path| path.is_file())
        .or_else(|| find_executable("google-chrome"))
        .or_else(|| find_executable("chrome"))
        .or_else(|| find_executable("chromium"))
        .or_else(|| find_executable("chromium-browser"))
        .ok_or_else(|| "could not find Google Chrome; set AGENT_BROWSER_CHROME_PATH".to_string())
}

fn chrome_candidates() -> Vec<PathBuf> {
    let mut candidates = Vec::new();
    #[cfg(target_os = "macos")]
    {
        candidates.push(PathBuf::from(
            "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
        ));
        if let Some(home) = dirs::home_dir() {
            candidates
                .push(home.join("Applications/Google Chrome.app/Contents/MacOS/Google Chrome"));
        }
    }
    #[cfg(target_os = "windows")]
    {
        if let Ok(program_files) = std::env::var("ProgramFiles") {
            candidates
                .push(PathBuf::from(program_files).join("Google/Chrome/Application/chrome.exe"));
        }
        if let Ok(program_files_x86) = std::env::var("ProgramFiles(x86)") {
            candidates.push(
                PathBuf::from(program_files_x86).join("Google/Chrome/Application/chrome.exe"),
            );
        }
    }
    candidates
}

fn chrome_profile_dir() -> Result<PathBuf, String> {
    if let Ok(path) = std::env::var("AGENT_BROWSER_CHROME_PROFILE_DIR") {
        let path = PathBuf::from(path);
        if path.is_dir() {
            return Ok(path);
        }
        return Err(format!(
            "AGENT_BROWSER_CHROME_PROFILE_DIR does not point to a directory: {}",
            path.display()
        ));
    }

    let home = dirs::home_dir().ok_or_else(|| "could not determine home directory".to_string())?;
    let path = if cfg!(target_os = "macos") {
        home.join("Library/Application Support/Google/Chrome/Default")
    } else if cfg!(target_os = "windows") {
        let local_app_data = std::env::var("LOCALAPPDATA")
            .map(PathBuf::from)
            .map_err(|_| {
                "LOCALAPPDATA is not set; set AGENT_BROWSER_CHROME_PROFILE_DIR".to_string()
            })?;
        local_app_data.join("Google/Chrome/User Data/Default")
    } else {
        home.join(".config/google-chrome/Default")
    };

    if path.is_dir() {
        Ok(path)
    } else {
        Err(format!(
            "could not find Chrome Default profile at {}; set AGENT_BROWSER_CHROME_PROFILE_DIR",
            path.display()
        ))
    }
}

fn copy_local_state(profile: &Path, user_data_dir: &Path) -> Result<(), String> {
    let Some(parent) = profile.parent() else {
        return Ok(());
    };
    let source = parent.join("Local State");
    if source.is_file() {
        fs::copy(&source, user_data_dir.join("Local State"))
            .map_err(|err| format!("failed to copy Chrome Local State: {err}"))?;
    }
    Ok(())
}

fn copy_dir_recursive(source: &Path, destination: &Path) -> Result<(), std::io::Error> {
    fs::create_dir_all(destination)?;
    for entry in fs::read_dir(source)? {
        let entry = entry?;
        let source_path = entry.path();
        let destination_path = destination.join(entry.file_name());
        let metadata = entry.file_type()?;
        if metadata.is_dir() {
            copy_dir_recursive(&source_path, &destination_path)?;
        } else {
            fs::copy(&source_path, &destination_path)?;
        }
    }
    Ok(())
}

fn available_port() -> Result<u16, String> {
    let listener = TcpListener::bind(("127.0.0.1", 0))
        .map_err(|err| format!("failed to allocate local debugging port: {err}"))?;
    listener
        .local_addr()
        .map(|addr| addr.port())
        .map_err(|err| format!("failed to inspect local debugging port: {err}"))
}

fn find_playwright() -> Option<PathBuf> {
    let path = std::env::var_os("PATH")?;
    std::env::split_paths(&path)
        .map(|dir| dir.join("playwright"))
        .find(|candidate| candidate.is_file())
}

fn find_executable(name: &str) -> Option<PathBuf> {
    let candidate = PathBuf::from(name);
    if candidate.components().count() > 1 && candidate.is_file() {
        return Some(candidate);
    }

    let path = std::env::var_os("PATH")?;
    std::env::split_paths(&path)
        .map(|dir| dir.join(name))
        .find(|candidate| candidate.is_file())
}

fn default_timeout() -> u64 {
    60
}

struct BrowserSession {
    port: u16,
    child: Child,
    temp: TempDirGuard,
    visible: bool,
}

struct TempDirGuard {
    path: PathBuf,
}

impl TempDirGuard {
    fn new() -> Result<Self, String> {
        let path = std::env::temp_dir().join(format!("agent-browser-{}", Uuid::new_v4()));
        fs::create_dir_all(&path)
            .map_err(|err| format!("failed to create temporary browser directory: {err}"))?;
        Ok(Self { path })
    }
}

impl Drop for TempDirGuard {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.path);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chrome_launch_args_default_to_headless() {
        let args = chrome_launch_args(Path::new("/tmp/profile"), 9222, None, false);

        assert!(args.contains(&"--headless=new".to_string()));
        assert!(args.contains(&"--disable-gpu".to_string()));
    }

    #[test]
    fn chrome_launch_args_can_show_browser_explicitly() {
        let args = chrome_launch_args(
            Path::new("/tmp/profile"),
            9222,
            Some("https://example.com"),
            true,
        );

        assert!(!args.contains(&"--headless=new".to_string()));
        assert!(args.contains(&"https://example.com".to_string()));
    }

    #[test]
    fn generated_playwright_script_keeps_browser_open_by_default() {
        let script = playwright_script(
            Path::new("/usr/local/bin/playwright"),
            9222,
            Some("https://example.com"),
            "return await page.title();",
            false,
        );

        assert!(script.contains("const closeBrowser = false;"));
        assert!(script.contains("if (closeBrowser)"));
        assert!(!script.contains("await browser.close();\n}})().catch"));
    }

    #[test]
    fn generated_playwright_script_can_close_browser_explicitly() {
        let script = playwright_script(
            Path::new("/usr/local/bin/playwright"),
            9222,
            None,
            "return 'done';",
            true,
        );

        assert!(script.contains("const closeBrowser = true;"));
        assert!(script.contains("await browser.close();"));
    }
}
