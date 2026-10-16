use agent_rs::cli;
use agent_rs::tools::browser::shutdown_browser_session;
use agent_rs::tools::chrome_tab::shutdown_chrome_tab_session;

#[tokio::main]
async fn main() -> std::process::ExitCode {
    let result = cli::run().await;
    shutdown_browser_session().await;
    shutdown_chrome_tab_session().await;
    match result {
        Ok(()) => std::process::ExitCode::SUCCESS,
        Err(err) => {
            eprintln!("{err}");
            std::process::ExitCode::from(1)
        }
    }
}
