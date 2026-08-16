use std::sync::Arc;

use clap::Parser;
use omnivoice_server::{
    build_router,
    error::ServerError,
    runtime::{AppState, PipelineSpeechRuntime, ServerConfig},
    wyoming, ServerArgs, VoiceCatalog,
};
use tokio::{net::TcpListener, sync::watch};
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
    let http_listener = TcpListener::bind((host.as_str(), port)).await?;
    let wyoming_listener = if let Some(wyoming_port) = args.wyoming_port {
        let wyoming_host = args.resolved_wyoming_host().to_string();
        let listener = TcpListener::bind((wyoming_host.as_str(), wyoming_port)).await?;
        let catalog = match args.voices_config.as_deref() {
            Some(path) => VoiceCatalog::from_path(path)?,
            None => VoiceCatalog::built_in(),
        };
        Some((listener, wyoming_host, wyoming_port, Arc::new(catalog)))
    } else {
        None
    };

    info!("omnivoice-server OpenAI endpoint listening on http://{host}:{port}");
    if let Some((_, wyoming_host, wyoming_port, catalog)) = &wyoming_listener {
        info!(
            voices = catalog.voices().count(),
            "omnivoice-server Wyoming endpoint listening on tcp://{wyoming_host}:{wyoming_port}"
        );
    }

    let state_for_runtime = state.clone();
    let catalog_for_runtime = wyoming_listener
        .as_ref()
        .map(|(_, _, _, catalog)| Arc::clone(catalog));
    tokio::task::spawn_blocking(move || {
        match PipelineSpeechRuntime::from_options(runtime_options) {
            Ok(runtime) => {
                if let Some(catalog) = catalog_for_runtime {
                    if let Err(error) = catalog.prepare_clone_prompts(&runtime) {
                        state_for_runtime.mark_failed();
                        error!(
                            "omnivoice-server voice prompts failed to initialize: {:?}",
                            error
                        );
                        return;
                    }
                }
                state_for_runtime.install_runtime(runtime);
                info!("omnivoice-server runtime is ready");
            }
            Err(error) => {
                state_for_runtime.mark_failed();
                error!("omnivoice-server runtime failed to initialize: {:?}", error);
            }
        }
    });

    let (shutdown_tx, shutdown_rx) = watch::channel(false);
    let http_shutdown = shutdown_rx.clone();
    let mut http_task = tokio::spawn(async move {
        axum::serve(http_listener, app)
            .with_graceful_shutdown(wait_for_shutdown(http_shutdown))
            .await
            .map_err(ServerError::from)
    });
    let mut wyoming_task = wyoming_listener.map(|(listener, _, _, catalog)| {
        let state = state.clone();
        let shutdown = shutdown_rx.clone();
        tokio::spawn(async move {
            wyoming::serve(listener, state, catalog, shutdown)
                .await
                .map_err(ServerError::from)
        })
    });

    let mut http_done = false;
    let mut wyoming_done = false;
    let mut first_error = None;
    if let Some(wyoming_task_ref) = wyoming_task.as_mut() {
        tokio::select! {
            _ = shutdown_signal() => {},
            result = &mut http_task => {
                http_done = true;
                capture_task_error(result, &mut first_error);
            },
            result = wyoming_task_ref => {
                wyoming_done = true;
                capture_task_error(result, &mut first_error);
            },
        }
    } else {
        tokio::select! {
            _ = shutdown_signal() => {},
            result = &mut http_task => {
                http_done = true;
                capture_task_error(result, &mut first_error);
            },
        }
    }

    let _ = shutdown_tx.send(true);
    if !http_done {
        capture_task_error(http_task.await, &mut first_error);
    }
    if !wyoming_done {
        if let Some(task) = wyoming_task {
            capture_task_error(task.await, &mut first_error);
        }
    }

    if let Some(error) = first_error {
        Err(error)
    } else {
        info!("omnivoice-server shut down");
        Ok(())
    }
}

fn capture_task_error(
    result: Result<Result<(), ServerError>, tokio::task::JoinError>,
    first_error: &mut Option<ServerError>,
) {
    let result = result.map_err(ServerError::from).and_then(|result| result);
    if let Err(error) = result {
        if first_error.is_none() {
            *first_error = Some(error);
        }
    }
}

async fn wait_for_shutdown(mut shutdown: watch::Receiver<bool>) {
    while !*shutdown.borrow() {
        if shutdown.changed().await.is_err() {
            break;
        }
    }
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
