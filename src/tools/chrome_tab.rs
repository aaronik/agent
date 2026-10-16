use std::sync::{Arc, OnceLock};
use std::time::Duration;

use futures_util::{SinkExt, StreamExt};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::{Mutex, mpsc, oneshot};
use tokio::time;
use tokio_tungstenite::tungstenite::{
    Message,
    handshake::server::{Request, Response},
};
use uuid::Uuid;

const PORT: u16 = 17777;
const ADDRESS: &str = "ws://127.0.0.1:17777/agent";
// Chrome retries on a 30s alarm after its initial bounded reconnect burst.
const FIRST_CONNECTION_WAIT: Duration = Duration::from_secs(35);
static BRIDGE: OnceLock<Result<(), String>> = OnceLock::new();
static LEASE: OnceLock<Arc<Mutex<Option<Lease>>>> = OnceLock::new();
type PendingRequest = (Value, oneshot::Sender<Value>);
type Connection = Arc<Mutex<ConnectionState>>;

#[derive(Default)]
struct ConnectionState {
    sender: Option<mpsc::Sender<PendingRequest>>,
    enabled: bool,
}

struct Lease {
    socket: tokio_tungstenite::WebSocketStream<tokio_tungstenite::MaybeTlsStream<TcpStream>>,
}

#[derive(Clone, Debug, Deserialize, JsonSchema, Serialize)]
pub struct ChromeTabArgs {
    /// One of: status, open, navigate, snapshot, click, type, scroll, close.
    pub action: String,
    /// Tab ID returned by open. Required for all tab actions except status and open.
    #[serde(default)]
    pub tab_id: Option<i64>,
    /// HTTPS or HTTP URL for open or navigate.
    #[serde(default)]
    pub url: Option<String>,
    /// CSS selector for click or type.
    #[serde(default)]
    pub selector: Option<String>,
    /// Text for type.
    #[serde(default)]
    pub text: Option<String>,
    /// Vertical scroll distance in pixels (default 600).
    #[serde(default)]
    pub pixels: Option<i64>,
}

pub async fn chrome_tab(args: ChromeTabArgs) -> Result<String, String> {
    validate(&args)?;
    if let Err(err) = BRIDGE.get_or_init(|| start_bridge().map_err(|e| e.to_string())) {
        return Err(err.clone());
    }
    let reply = request_host(serde_json::to_value(args).map_err(|e| e.to_string())?).await?;
    if let Some(error) = reply.get("error").and_then(Value::as_str) {
        Err(error.to_string())
    } else {
        Ok(reply
            .get("result")
            .map_or_else(|| reply.to_string(), Value::to_string))
    }
}

fn validate(args: &ChromeTabArgs) -> Result<(), String> {
    match args.action.as_str() {
        "status" | "open" | "navigate" | "snapshot" | "click" | "type" | "scroll" | "close" => {}
        _ => return Err("unknown browser action".into()),
    }
    if !matches!(args.action.as_str(), "status" | "open") && args.tab_id.is_none() {
        return Err("tab_id is required for this action".into());
    }
    if matches!(args.action.as_str(), "open" | "navigate") {
        let url = args.url.as_deref().ok_or("url is required")?;
        let parsed = url::Url::parse(url).map_err(|_| "invalid URL")?;
        if !matches!(parsed.scheme(), "http" | "https") {
            return Err("only HTTP(S) URLs are allowed".into());
        }
    }
    if matches!(args.action.as_str(), "click" | "type") && args.selector.is_none() {
        return Err("selector is required".into());
    }
    if args.action == "type" && args.text.is_none() {
        return Err("text is required".into());
    }
    Ok(())
}

fn start_bridge() -> std::io::Result<()> {
    match std::net::TcpListener::bind((std::net::Ipv4Addr::LOCALHOST, PORT)) {
        Ok(listener) => {
            listener.set_nonblocking(true)?;
            let listener = TcpListener::from_std(listener)?;
            let connection: Connection = Default::default();
            let shared = connection.clone();
            tokio::spawn(async move {
                while let Ok((stream, _)) = listener.accept().await {
                    let connection = shared.clone();
                    tokio::spawn(async move { serve_peer(stream, connection).await });
                }
            });
            Ok(())
        }
        Err(error) if error.kind() == std::io::ErrorKind::AddrInUse => Ok(()),
        Err(error) => Err(error),
    }
}

fn status(connected: bool, enabled: bool) -> Value {
    json!({"result": if !connected {
        "Chrome bridge is waiting for the extension. Load or reload chrome-extension in chrome://extensions."
    } else if !enabled {
        "Chrome extension is installed and connected, but its access switch is switched off. Turn it on in the extension popup to allow browser actions."
    } else {
        "Chrome extension connected and switched on. Opened tabs are controlled in your normal Chrome profile."
    }})
}

async fn dispatch(connection: &Connection, request: Value) -> Result<Value, String> {
    // The first call may start the bridge before the extension reconnects.
    // Wait through the extension's 30-second alarm retry on first use.
    let is_status = request["action"] == "status";
    let is_cleanup = request["action"] == "close_owner";
    let deadline = time::Instant::now()
        + if is_cleanup {
            Duration::from_secs(2)
        } else {
            FIRST_CONNECTION_WAIT
        };
    let sender = loop {
        let state = connection.lock().await;
        if is_status && state.sender.is_some() {
            return Ok(status(true, state.enabled));
        }
        if let Some(sender) = state.sender.clone() {
            if !state.enabled && !is_cleanup {
                return Err("Chrome extension access switch is switched off. Turn it on in the popup to allow browser actions.".into());
            }
            break sender;
        }
        drop(state);
        if time::Instant::now() >= deadline {
            if is_status {
                return Ok(status(false, false));
            }
            return Err("Chrome extension not connected. Check the extension popup and retry; the bridge may have just started.".into());
        }
        time::sleep(Duration::from_millis(100)).await;
    };
    let (tx, rx) = oneshot::channel();
    sender
        .send((request, tx))
        .await
        .map_err(|_| "Chrome extension disconnected".to_string())?;
    time::timeout(Duration::from_secs(25), rx)
        .await
        .map_err(|_| "Chrome extension did not respond within 25 seconds".to_string())?
        .map_err(|_| "Chrome extension disconnected".to_string())
}

async fn connect_lease(address: &str) -> Result<Lease, String> {
    let (mut socket, _) = tokio_tungstenite::connect_async(address)
        .await
        .map_err(|_| {
            "Cannot reach Chrome bridge on 127.0.0.1:17777; retry after the other agent exits."
                .to_string()
        })?;
    let owner = Uuid::new_v4().to_string();
    socket
        .send(Message::Text(
            json!({"owner":owner,"protocol":2}).to_string().into(),
        ))
        .await
        .map_err(|e| e.to_string())?;
    let reply = time::timeout(Duration::from_secs(2), socket.next()).await;
    if !matches!(reply, Ok(Some(Ok(Message::Text(ref text)))) if serde_json::from_str::<Value>(text).ok().is_some_and(|value| value["protocol"] == 2))
    {
        return Err("incompatible Chrome bridge on 127.0.0.1:17777. Exit the older agent process holding port 17777, then restart the agent; reloading the extension alone does not update the bridge.".into());
    }
    Ok(Lease { socket })
}

async fn request_host(request: Value) -> Result<Value, String> {
    let lease = LEASE.get_or_init(|| Arc::new(Mutex::new(None)));
    let mut lease = lease.lock().await;
    if lease.is_none() {
        *lease = Some(connect_lease(ADDRESS).await?);
    }
    let current = lease.as_mut().expect("lease initialized");
    let result = async {
        current
            .socket
            .send(Message::Text(request.to_string().into()))
            .await
            .map_err(|e| e.to_string())?;
        let reply = time::timeout(Duration::from_secs(26), current.socket.next())
            .await
            .map_err(|_| "Chrome bridge did not respond".to_string())?
            .ok_or("Chrome bridge disconnected")?
            .map_err(|e| e.to_string())?;
        serde_json::from_str(reply.to_text().map_err(|e| e.to_string())?).map_err(|e| e.to_string())
    }
    .await;
    if result.is_err() {
        *lease = None;
    }
    result
}

pub async fn shutdown_chrome_tab_session() {
    if let Some(lease) = LEASE.get()
        && let Some(mut lease) = lease.lock().await.take()
    {
        let _ = time::timeout(Duration::from_secs(3), async {
            lease
                .socket
                .send(Message::Text(json!({"release":true}).to_string().into()))
                .await?;
            // The bridge acknowledges only after the extension removes this owner's tabs.
            let _ = lease.socket.next().await;
            lease.socket.close(None).await
        })
        .await;
    }
}

#[allow(clippy::result_large_err)] // tungstenite requires a large ErrorResponse from this callback.
async fn serve_peer(stream: TcpStream, connection: Connection) {
    let (role_tx, role_rx) = oneshot::channel();
    let mut role_tx = Some(role_tx);
    let ws = tokio_tungstenite::accept_hdr_async(stream, |req: &Request, response: Response| {
        let role = match req.uri().path() {
            "/agent" => "agent",
            _ => "extension",
        };
        if let Some(tx) = role_tx.take() {
            let _ = tx.send(role);
        }
        check_origin(req, response)
    })
    .await;
    let Ok(ws) = ws else { return };
    let Ok(role) = role_rx.await else { return };
    if role == "extension" {
        serve_extension(ws, connection).await;
    } else {
        serve_agent(ws, connection).await;
    }
}

async fn serve_agent(
    mut ws: tokio_tungstenite::WebSocketStream<TcpStream>,
    connection: Connection,
) {
    let Ok(Some(Ok(Message::Text(first)))) = time::timeout(Duration::from_secs(5), ws.next()).await
    else {
        return;
    };
    let Ok(hello) = serde_json::from_str::<Value>(&first) else {
        return;
    };
    let Some(owner) = hello
        .get("owner")
        .and_then(Value::as_str)
        .filter(|id| Uuid::parse_str(id).is_ok())
    else {
        return;
    };
    let owner = owner.to_string();
    if hello["protocol"] != 2 {
        return;
    }
    if ws
        .send(Message::Text(json!({"protocol":2}).to_string().into()))
        .await
        .is_err()
    {
        return;
    }
    while let Some(Ok(Message::Text(text))) = ws.next().await {
        let Ok(mut request) = serde_json::from_str::<Value>(&text) else {
            break;
        };
        if request["release"] == true {
            break;
        }
        if request["action"] == "close_owner" {
            let _ = ws
                .send(Message::Text(
                    json!({"error":"internal-only action"}).to_string().into(),
                ))
                .await;
            continue;
        }
        if request["action"] == "open" {
            request["owner"] = Value::String(owner.clone());
        }
        let reply = match dispatch(&connection, request).await {
            Ok(reply) => reply,
            Err(error) => json!({"error":error}),
        };
        if ws
            .send(Message::Text(reply.to_string().into()))
            .await
            .is_err()
        {
            break;
        }
    }
    let _ = time::timeout(
        Duration::from_secs(2),
        dispatch(&connection, json!({"action":"close_owner", "owner":owner})),
    )
    .await;
    let _ = ws
        .send(Message::Text(
            json!({"cleanup_done":true}).to_string().into(),
        ))
        .await;
}

async fn serve_extension(
    mut ws: tokio_tungstenite::WebSocketStream<TcpStream>,
    connection: Connection,
) {
    let (tx, mut rx) = mpsc::channel::<PendingRequest>(8);
    if ws
        .send(Message::Text(json!({"connected":true}).to_string().into()))
        .await
        .is_err()
    {
        return;
    }
    {
        let mut state = connection.lock().await;
        state.sender = Some(tx.clone());
        state.enabled = false;
    }
    loop {
        tokio::select! {
            item = rx.recv() => {
                let Some((request, reply)) = item else { break };
                if ws.send(Message::Text(request.to_string().into())).await.is_err() { break; }
                let received = time::timeout(Duration::from_secs(24), async {
                    loop {
                        match ws.next().await {
                            Some(Ok(Message::Text(text))) => {
                                if let Ok(value) = serde_json::from_str::<Value>(&text) {
                                    if value["event"] == "status" {
                                        connection.lock().await.enabled = value["enabled"] == true;
                                        continue;
                                    }
                                    break Some(value);
                                }
                                break None;
                            },
                            Some(Ok(Message::Ping(payload))) => { let _ = ws.send(Message::Pong(payload)).await; }
                            Some(Ok(_)) => {},
                            _ => break None,
                        }
                    }
                }).await;
                let Some(value) = received.ok().flatten() else { break };
                let _ = reply.send(value);
            }
            event = ws.next() => {
                match event {
                    Some(Ok(Message::Ping(payload))) => { if ws.send(Message::Pong(payload)).await.is_err() { break; } }
                    Some(Ok(Message::Text(text))) => {
                        if let Ok(value) = serde_json::from_str::<Value>(&text)
                            && value["event"] == "status" {
                                connection.lock().await.enabled = value["enabled"] == true;
                        }
                    },
                    Some(Ok(_)) => {},
                    _ => break,
                }
            }
        }
    }
    let mut guard = connection.lock().await;
    if guard
        .sender
        .as_ref()
        .is_some_and(|current| current.same_channel(&tx))
    {
        guard.sender = None;
        guard.enabled = false;
    }
}

#[allow(clippy::result_large_err)] // Signature required by tungstenite's Callback trait.
fn check_origin(
    req: &Request,
    response: Response,
) -> Result<Response, tokio_tungstenite::tungstenite::handshake::server::ErrorResponse> {
    let origin = req
        .headers()
        .get("origin")
        .and_then(|h| h.to_str().ok())
        .unwrap_or("");
    let extension_origin = origin.starts_with("chrome-extension://")
        && origin["chrome-extension://".len()..]
            .chars()
            .all(|c| ('a'..='p').contains(&c))
        && origin.len() == "chrome-extension://".len() + 32;
    if (req.uri().path() == "/" && extension_origin)
        || (req.uri().path() == "/agent" && origin.is_empty())
    {
        Ok(response)
    } else {
        Err(tokio_tungstenite::tungstenite::http::Response::builder()
            .status(403)
            .body(Some("invalid bridge origin".to_string()))
            .unwrap())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn bridge_accepts_extension_without_pairing_and_forwards_agent_request() {
        use tokio_tungstenite::tungstenite::client::IntoClientRequest;
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = format!("ws://127.0.0.1:{}", listener.local_addr().unwrap().port());
        let connection: Connection = Default::default();
        let host = tokio::spawn({
            let connection = connection.clone();
            async move {
                for _ in 0..2 {
                    let (stream, _) = listener.accept().await.unwrap();
                    let connection = connection.clone();
                    tokio::spawn(async move { serve_peer(stream, connection).await });
                }
            }
        });
        let mut request = address.as_str().into_client_request().unwrap();
        request.headers_mut().insert(
            "origin",
            format!("chrome-extension://{}", "a".repeat(32))
                .parse()
                .unwrap(),
        );
        let (mut extension, _) = tokio_tungstenite::connect_async(request).await.unwrap();
        assert!(
            extension
                .next()
                .await
                .unwrap()
                .unwrap()
                .to_text()
                .unwrap()
                .contains("connected")
        );
        extension
            .send(Message::Text(
                json!({"event":"status","enabled":false}).to_string().into(),
            ))
            .await
            .unwrap();
        time::sleep(Duration::from_millis(10)).await;
        assert!(
            dispatch(&connection, json!({"action":"status"}))
                .await
                .unwrap()["result"]
                .as_str()
                .unwrap()
                .contains("switched off")
        );
        assert!(
            dispatch(&connection, json!({"action":"open"}))
                .await
                .unwrap_err()
                .contains("switched off")
        );
        extension
            .send(Message::Text(
                json!({"event":"status","enabled":true}).to_string().into(),
            ))
            .await
            .unwrap();
        time::sleep(Duration::from_millis(10)).await;
        let responder = tokio::spawn(async move {
            let request = extension.next().await.unwrap().unwrap();
            assert_eq!(
                serde_json::from_str::<Value>(request.to_text().unwrap()).unwrap()["action"],
                "open"
            );
            extension
                .send(Message::Text(
                    json!({"result":{"tab_id":42}}).to_string().into(),
                ))
                .await
                .unwrap();
        });
        let (mut client, _) = tokio_tungstenite::connect_async(format!("{address}/agent"))
            .await
            .unwrap();
        client
            .send(Message::Text(
                json!({"owner":Uuid::new_v4().to_string(),"protocol":2})
                    .to_string()
                    .into(),
            ))
            .await
            .unwrap();
        let hello = time::timeout(Duration::from_secs(2), client.next())
            .await
            .unwrap()
            .unwrap()
            .unwrap();
        assert_eq!(
            serde_json::from_str::<Value>(hello.to_text().unwrap()).unwrap()["protocol"],
            2
        );
        client
            .send(Message::Text(
                json!({"action":"open", "url":"https://example.org"})
                    .to_string()
                    .into(),
            ))
            .await
            .unwrap();
        assert!(
            client
                .next()
                .await
                .unwrap()
                .unwrap()
                .to_text()
                .unwrap()
                .contains("42")
        );
        responder.await.unwrap();
        host.await.unwrap();
    }

    #[tokio::test]
    async fn lease_rejects_older_bridge_protocol_before_sending_action() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = format!("ws://127.0.0.1:{}", listener.local_addr().unwrap().port());
        let server = tokio::spawn(async move {
            let (stream, _) = listener.accept().await.unwrap();
            let mut ws = tokio_tungstenite::accept_async(stream).await.unwrap();
            let hello = ws.next().await.unwrap().unwrap();
            assert!(hello.to_text().unwrap().contains("owner"));
            // An old bridge treats the owner frame as a command and returns an error.
            ws.send(Message::Text(
                json!({"error":"Tab not opened by Agent; access denied"})
                    .to_string()
                    .into(),
            ))
            .await
            .unwrap();
            if let Ok(Some(Ok(Message::Text(text)))) =
                time::timeout(Duration::from_millis(100), ws.next()).await
            {
                panic!("sent action to incompatible bridge: {text}");
            }
        });
        let err = connect_lease(&address)
            .await
            .err()
            .expect("old bridge must be refused");
        assert!(err.contains("incompatible Chrome bridge"), "{err}");
        server.await.unwrap();
    }

    #[tokio::test]
    async fn agent_disconnect_notifies_extension_to_close_only_its_tabs() {
        use tokio_tungstenite::tungstenite::client::IntoClientRequest;
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = format!("ws://127.0.0.1:{}", listener.local_addr().unwrap().port());
        let connection: Connection = Default::default();
        let server = tokio::spawn({
            let connection = connection.clone();
            async move {
                for _ in 0..2 {
                    let (stream, _) = listener.accept().await.unwrap();
                    let state = connection.clone();
                    tokio::spawn(async move { serve_peer(stream, state).await });
                }
            }
        });
        let mut request = address.as_str().into_client_request().unwrap();
        request.headers_mut().insert(
            "origin",
            format!("chrome-extension://{}", "a".repeat(32))
                .parse()
                .unwrap(),
        );
        let (mut extension, _) = tokio_tungstenite::connect_async(request).await.unwrap();
        extension.next().await.unwrap().unwrap();
        extension
            .send(Message::Text(
                json!({"event":"status","enabled":true}).to_string().into(),
            ))
            .await
            .unwrap();
        let (mut agent, _) = tokio_tungstenite::connect_async(format!("{address}/agent"))
            .await
            .unwrap();
        agent
            .send(Message::Text(
                json!({"owner":Uuid::new_v4().to_string(),"protocol":2})
                    .to_string()
                    .into(),
            ))
            .await
            .unwrap();
        let hello = agent.next().await.unwrap().unwrap();
        assert_eq!(
            serde_json::from_str::<Value>(hello.to_text().unwrap()).unwrap()["protocol"],
            2
        );
        agent
            .send(Message::Text(
                json!({"action":"open","url":"https://example.org"})
                    .to_string()
                    .into(),
            ))
            .await
            .unwrap();
        let open = extension.next().await.unwrap().unwrap();
        let owner = serde_json::from_str::<Value>(open.to_text().unwrap()).unwrap()["owner"]
            .as_str()
            .unwrap()
            .to_owned();
        assert!(!owner.is_empty());
        extension
            .send(Message::Text(
                json!({"result":{"tab_id":42}}).to_string().into(),
            ))
            .await
            .unwrap();
        agent.next().await.unwrap().unwrap();
        agent
            .send(Message::Text(json!({"release":true}).to_string().into()))
            .await
            .unwrap();
        let cleanup = time::timeout(Duration::from_secs(2), extension.next())
            .await
            .unwrap()
            .unwrap()
            .unwrap();
        let cleanup: Value = serde_json::from_str(cleanup.to_text().unwrap()).unwrap();
        assert_eq!(cleanup["action"], "close_owner");
        assert_eq!(cleanup["owner"], owner);
        extension
            .send(Message::Text(
                json!({"result":{"closed":1}}).to_string().into(),
            ))
            .await
            .unwrap();
        let ack = agent.next().await.unwrap().unwrap();
        assert_eq!(
            serde_json::from_str::<Value>(ack.to_text().unwrap()).unwrap()["cleanup_done"],
            true
        );
        server.await.unwrap();
    }

    #[tokio::test]
    async fn abrupt_agent_disconnect_still_requests_owner_cleanup() {
        use tokio_tungstenite::tungstenite::client::IntoClientRequest;
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = format!("ws://127.0.0.1:{}", listener.local_addr().unwrap().port());
        let connection: Connection = Default::default();
        let server = tokio::spawn({
            let connection = connection.clone();
            async move {
                for _ in 0..2 {
                    let (stream, _) = listener.accept().await.unwrap();
                    let state = connection.clone();
                    tokio::spawn(async move { serve_peer(stream, state).await });
                }
            }
        });
        let mut request = address.as_str().into_client_request().unwrap();
        request.headers_mut().insert(
            "origin",
            format!("chrome-extension://{}", "a".repeat(32))
                .parse()
                .unwrap(),
        );
        let (mut extension, _) = tokio_tungstenite::connect_async(request).await.unwrap();
        extension.next().await.unwrap().unwrap();
        let (mut agent, _) = tokio_tungstenite::connect_async(format!("{address}/agent"))
            .await
            .unwrap();
        let owner = Uuid::new_v4().to_string();
        agent
            .send(Message::Text(
                json!({"owner":owner,"protocol":2}).to_string().into(),
            ))
            .await
            .unwrap();
        drop(agent);
        let cleanup = time::timeout(Duration::from_secs(2), extension.next())
            .await
            .unwrap()
            .unwrap()
            .unwrap();
        let cleanup: Value = serde_json::from_str(cleanup.to_text().unwrap()).unwrap();
        assert_eq!(cleanup["action"], "close_owner");
        assert_eq!(cleanup["owner"], owner);
        extension
            .send(Message::Text(
                json!({"result":{"closed":0}}).to_string().into(),
            ))
            .await
            .unwrap();
        server.await.unwrap();
    }

    #[test]
    fn first_call_wait_covers_extension_alarm_retry() {
        assert!(FIRST_CONNECTION_WAIT > Duration::from_secs(30));
    }

    #[tokio::test]
    async fn first_status_waits_briefly_for_extension_to_reconnect() {
        let connection: Connection = Default::default();
        let pending = tokio::spawn({
            let connection = connection.clone();
            async move { dispatch(&connection, json!({"action":"status"})).await }
        });
        time::sleep(Duration::from_millis(50)).await;
        {
            let mut state = connection.lock().await;
            state.sender = Some(mpsc::channel::<PendingRequest>(1).0);
            state.enabled = true;
        }
        let response = pending.await.unwrap().unwrap();
        assert!(response["result"].as_str().unwrap().contains("switched on"));
    }

    #[tokio::test]
    async fn first_browser_action_waits_for_extension_to_reconnect() {
        let connection: Connection = Default::default();
        let pending = tokio::spawn({
            let connection = connection.clone();
            async move {
                dispatch(
                    &connection,
                    json!({"action":"open", "url":"https://example.org"}),
                )
                .await
            }
        });
        time::sleep(Duration::from_millis(50)).await;
        let (tx, mut rx) = mpsc::channel::<PendingRequest>(8);
        {
            let mut state = connection.lock().await;
            state.sender = Some(tx);
            state.enabled = true;
        }
        let responder = tokio::spawn(async move {
            let (_, reply) = rx.recv().await.unwrap();
            reply.send(json!({"result":{"tab_id":42}})).unwrap();
        });
        assert_eq!(pending.await.unwrap().unwrap()["result"]["tab_id"], 42);
        responder.await.unwrap();
    }

    #[test]
    fn handshake_rejects_non_extension_origins() {
        let request = Request::builder()
            .header("origin", "https://attacker.test")
            .body(())
            .unwrap();
        assert!(check_origin(&request, Response::new(())).is_err());
        let request = Request::builder()
            .uri("/")
            .header("origin", format!("chrome-extension://{}", "a".repeat(32)))
            .body(())
            .unwrap();
        assert!(check_origin(&request, Response::new(())).is_ok());
        let request = Request::builder().uri("/agent").body(()).unwrap();
        assert!(check_origin(&request, Response::new(())).is_ok());
    }

    #[test]
    fn rejects_invalid_urls_and_missing_tab_ids() {
        let mut args = ChromeTabArgs {
            action: "open".into(),
            tab_id: None,
            url: Some("file:///etc/passwd".into()),
            selector: None,
            text: None,
            pixels: None,
        };
        assert!(validate(&args).is_err());
        args.url = Some("https://example.org".into());
        assert!(validate(&args).is_ok());
        args.action = "snapshot".into();
        assert!(validate(&args).is_err());
    }
}
