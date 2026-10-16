use agent_rs::tools::browser::{PlaywrightBrowserSessionArgs, playwright_browser_session};

#[tokio::test]
#[ignore = "manual live browser smoke; requires Chrome, node, and playwright"]
async fn playwright_browser_session_live_headless_manual() {
    let output = playwright_browser_session(PlaywrightBrowserSessionArgs {
        javascript: "return await page.title();".to_string(),
        url: Some("data:text/html,<title>manual-headless-smoke</title>".to_string()),
        timeout: 15,
        close: true,
        reset: true,
        visible: false,
    })
    .await
    .expect("headless playwright_browser_session smoke should succeed");

    assert!(output.contains("manual-headless-smoke"), "{output}");
    assert!(output.contains("[browser session closed]"), "{output}");
}
