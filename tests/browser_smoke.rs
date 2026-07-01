use agent_rs::tools::browser::{BrowserControlArgs, browser_control};

#[tokio::test]
#[ignore = "manual live browser smoke; requires Chrome profile, node, and playwright"]
async fn browser_control_live_headless_manual() {
    let output = browser_control(BrowserControlArgs {
        javascript: "return await page.title();".to_string(),
        url: Some("data:text/html,<title>manual-headless-smoke</title>".to_string()),
        timeout: 15,
        close: true,
        reset: true,
        visible: false,
    })
    .await
    .expect("headless browser_control smoke should succeed");

    assert!(output.contains("manual-headless-smoke"), "{output}");
    assert!(output.contains("[browser session closed]"), "{output}");
}
