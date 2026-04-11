use clap::Parser;
use omnivoice_server::{
    build_router,
    error::ServerError,
    runtime::{AppState, PipelineSpeechRuntime, ServerConfig},
    ServerArgs,
};
use tokio::net::TcpListener;
use tracing::{error, info};

#[tokio::main]
async fn main() -> Result<(), ServerError> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "omnivoice_server=info,tower_http=info".into()),
        )
        .with_target(false)
        .init();

    let args = ServerArgs::parse();
    let runtime_options = args.runtime_options()?;
    let config = ServerConfig::from_args(&args)?;
    let host = args.host.clone();
    let port = args.port;
    let state = AppState::starting(config);

    let app = build_router(state.clone());
    let listener = TcpListener::bind((host.as_str(), port)).await?;

    info!("omnivoice-server listening on http://{host}:{port}");
    tokio::task::spawn_blocking(move || {
        match PipelineSpeechRuntime::from_options(runtime_options) {
            Ok(runtime) => {
                state.install_runtime(runtime);
                info!("omnivoice-server runtime is ready");
            }
            Err(error) => {
                state.mark_failed();
                error!("omnivoice-server runtime failed to initialize: {:?}", error);
            }
        }
    });

    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await?;
    Ok(())
}

async fn shutdown_signal() {
    let ctrl_c = async {
        let _ = tokio::signal::ctrl_c().await;
    };

    #[cfg(unix)]
    let terminate = async {
        if let Ok(mut signal) =
            tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
        {
            let _ = signal.recv().await;
        }
    };

    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c => {},
        _ = terminate => {},
    }
    info!("shutdown signal received");
}
