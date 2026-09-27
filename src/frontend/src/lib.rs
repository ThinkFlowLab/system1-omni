//! Jev HTTP transport. The worker owns request parsing and inference.
pub mod engine;

use std::{env, error::Error, net::SocketAddr, time::Duration};

use axum::{
    Router,
    body::Body,
    extract::{Request, State},
    http::{HeaderMap, HeaderName, StatusCode, header},
    response::Response,
    routing::{get, post},
};
use reqwest::{Client, Url};

pub type BoxError = Box<dyn Error + Send + Sync>;

#[derive(Clone, Debug)]
pub struct Config {
    pub bind: SocketAddr,
    pub backend_url: Url,
    pub timeout: Duration,
}

impl Config {
    pub const DEFAULT_BIND: &str = "127.0.0.1:8080";
    pub const DEFAULT_BACKEND_URL: &str = "http://127.0.0.1:8000";
    pub const DEFAULT_TIMEOUT: Duration = Duration::from_secs(60);

    pub fn from_env() -> Result<Self, BoxError> {
        Self::new(
            &env_or("OMNI_JEV_BIND", Self::DEFAULT_BIND)?,
            &env_or("OMNI_JEV_BACKEND_URL", Self::DEFAULT_BACKEND_URL)?,
        )
    }

    pub fn new(bind: &str, backend_url: &str) -> Result<Self, BoxError> {
        let bind = bind
            .parse()
            .map_err(|e| format!("invalid bind address {bind:?}: {e}"))?;
        let mut backend_url =
            Url::parse(backend_url).map_err(|e| format!("invalid OMNI_JEV_BACKEND_URL: {e}"))?;
        // URL credentials would override the caller's Authorization header.
        if !matches!(backend_url.scheme(), "http" | "https")
            || !backend_url.has_host()
            || backend_url.query().is_some()
            || backend_url.fragment().is_some()
            || !backend_url.username().is_empty()
            || backend_url.password().is_some()
        {
            return Err(
                "OMNI_JEV_BACKEND_URL must be http(s) without credentials, query or fragment"
                    .into(),
            );
        }
        if !backend_url.path().ends_with('/') {
            backend_url.set_path(&format!("{}/", backend_url.path()));
        }
        Ok(Self {
            bind,
            backend_url,
            timeout: Self::DEFAULT_TIMEOUT,
        })
    }
}

#[derive(Clone)]
struct Backend {
    client: Client,
    base_url: Url,
}

/// Both routes share one client and its connection pool.
pub fn app(config: &Config) -> Result<Router, BoxError> {
    let client = Client::builder()
        .timeout(config.timeout)
        .retry(reqwest::retry::never())
        .redirect(reqwest::redirect::Policy::none())
        // Always connect to the configured worker, regardless of HTTP_PROXY.
        .no_proxy()
        .build()?;
    let backend = Backend {
        client,
        base_url: config.backend_url.clone(),
    };
    Ok(Router::new()
        .route("/v1/systemone", post(forward))
        .route("/health", get(forward))
        .with_state(backend))
}

async fn forward(
    State(backend): State<Backend>,
    request: Request,
) -> Result<Response, (StatusCode, &'static str)> {
    let (parts, body) = request.into_parts();
    let mut url = backend.base_url;
    url.set_path(&format!(
        "{}{}",
        url.path(),
        parts.uri.path().trim_start_matches('/')
    ));
    url.set_query(parts.uri.query());

    let mut headers = parts.headers;
    remove_hop_by_hop(&mut headers);
    headers.remove(header::HOST);

    let upstream = backend
        .client
        .request(parts.method, url)
        .headers(headers)
        .body(reqwest::Body::wrap_stream(body.into_data_stream()))
        .send()
        .await
        .map_err(backend_error)?;

    let status = upstream.status();
    let mut headers = upstream.headers().clone();
    remove_hop_by_hop(&mut headers);
    // Buffer the response so a body timeout can still become a 504.
    let body = upstream.bytes().await.map_err(backend_error)?;

    let mut response = Response::new(Body::from(body));
    *response.status_mut() = status;
    *response.headers_mut() = headers;
    Ok(response)
}

fn backend_error(error: reqwest::Error) -> (StatusCode, &'static str) {
    let error = error.without_url();
    eprintln!("backend request failed: {error}");
    if error.is_timeout() {
        (StatusCode::GATEWAY_TIMEOUT, "backend timed out\n")
    } else {
        (StatusCode::BAD_GATEWAY, "backend unavailable\n")
    }
}

/// Removes headers that describe a single connection (RFC 9110, section 7.6.1).
fn remove_hop_by_hop(headers: &mut HeaderMap) {
    let nominated: Vec<HeaderName> = headers
        .get_all(header::CONNECTION)
        .iter()
        .filter_map(|value| value.to_str().ok())
        .flat_map(|value| value.split(','))
        .filter_map(|name| HeaderName::from_bytes(name.trim().as_bytes()).ok())
        .collect();
    for name in nominated.iter().chain(&HOP_BY_HOP) {
        headers.remove(name);
    }
}

const HOP_BY_HOP: [HeaderName; 8] = [
    header::CONNECTION,
    HeaderName::from_static("keep-alive"),
    header::PROXY_AUTHENTICATE,
    header::PROXY_AUTHORIZATION,
    header::TE,
    header::TRAILER,
    header::TRANSFER_ENCODING,
    header::UPGRADE,
];

fn env_or(name: &str, default: &str) -> Result<String, BoxError> {
    match env::var(name) {
        Ok(value) => Ok(value),
        Err(env::VarError::NotPresent) => Ok(default.to_owned()),
        Err(error) => Err(format!("{name}: {error}").into()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_match_documented_values() {
        let config = Config::new(Config::DEFAULT_BIND, Config::DEFAULT_BACKEND_URL).unwrap();
        assert_eq!(config.bind.to_string(), "127.0.0.1:8080");
        assert_eq!(config.backend_url.as_str(), "http://127.0.0.1:8000/");
        assert_eq!(config.timeout, Duration::from_secs(60));
    }

    #[test]
    fn backend_path_prefix_gets_trailing_slash() {
        let config = Config::new("127.0.0.1:0", "https://example.com/worker").unwrap();
        assert_eq!(config.backend_url.as_str(), "https://example.com/worker/");
    }

    #[test]
    fn rejects_ambiguous_backend_urls() {
        for url in [
            "not a url",
            "ftp://localhost",
            "file:///tmp/worker",
            "http://user:pass@localhost",
            "http://localhost/?token=x",
            "http://localhost/#x",
        ] {
            assert!(Config::new("127.0.0.1:0", url).is_err(), "accepted {url}");
        }
        assert!(Config::new("localhost", Config::DEFAULT_BACKEND_URL).is_err());
    }

    #[test]
    fn invalid_backend_url_does_not_expose_credentials() {
        for url in [
            "http://user:private-password@localhost",
            "http://localhost/?token=private-token",
            "http://user:private-password@[invalid",
        ] {
            let error = Config::new("127.0.0.1:0", url).unwrap_err().to_string();
            assert!(!error.contains("private-"), "{error}");
        }
    }
}
