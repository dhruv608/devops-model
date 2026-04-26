# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
FastAPI application for the Safe Sre Env Environment.

This module creates an HTTP server that exposes the SafeSreEnvironment
over HTTP and WebSocket endpoints, compatible with EnvClient.

Endpoints:
    - POST /reset: Reset the environment
    - POST /step: Execute an action
    - GET /state: Get current environment state
    - GET /schema: Get action/observation schemas
    - WS /ws: WebSocket endpoint for persistent sessions

Usage:
    # Development (with auto-reload):
    uvicorn server.app:app --reload --host 0.0.0.0 --port 8000

    # Production:
    uvicorn server.app:app --host 0.0.0.0 --port 8000 --workers 4

    # Or run directly:
    python -m server.app
"""

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:  # pragma: no cover
    raise ImportError(
        "openenv is required for the web interface. Install dependencies with '\n    uv sync\n'"
    ) from e

try:
    from ..models import SafeSreAction, SafeSreObservation
    from .dashboard_ui import build_safe_sre_ui
    from .safe_sre_env_environment import SafeSreEnvironment
except ImportError:
    from models import SafeSreAction, SafeSreObservation
    from server.dashboard_ui import build_safe_sre_ui
    from server.safe_sre_env_environment import SafeSreEnvironment

import gradio as gr


# Create the app with the OpenEnv FastAPI server (mounts the bare playground
# at /web for backward compatibility — judges/agents who hit /web directly
# still see the standard tool form).
app = create_app(
    SafeSreEnvironment,
    SafeSreAction,
    SafeSreObservation,
    env_name="safe_sre_env",
    max_concurrent_envs=1,  # increase this number to allow more concurrent WebSocket sessions
)


# Mount our flat single-page dashboard at /dashboard.
#
# This is what the HF Space's App tab loads (see ``base_path: /dashboard``
# in README front-matter). Bypasses OpenEnv's outer Playground/Custom
# TabbedInterface wrapper — judges land directly on a flat page with
# project description + scenario dropdown + Run Comparison button +
# side-by-side outputs. The OpenEnv playground at /web stays accessible
# for anyone who explicitly visits it.
app = gr.mount_gradio_app(app, build_safe_sre_ui(), path="/dashboard")


def main(host: str = "0.0.0.0", port: int = 8000):
    """
    Entry point for direct execution via uv run or python -m.

    This function enables running the server without Docker:
        uv run --project . server
        uv run --project . server --port 8001
        python -m safe_sre_env.server.app

    Args:
        host: Host address to bind to (default: "0.0.0.0")
        port: Port number to listen on (default: 8000)

    For production deployments, consider using uvicorn directly with
    multiple workers:
        uvicorn safe_sre_env.server.app:app --workers 4
    """
    import uvicorn

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    main(port=args.port)
