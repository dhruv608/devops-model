#!/usr/bin/env bash
# Hour 9 HTTP smoke: spin up uvicorn, hit every public endpoint, kill it.
#
# Run from project root:
#   bash demo/walkthrough_hour_9_http.sh
#
# What we're proving:
# - openenv-core's create_app + our SafeSreEnvironment subclass actually
#   binds to a real HTTP port without import errors.
# - /health returns {"status":"healthy"} so HF Space's container probe will
#   pass (Hour 13).
# - /reset returns a valid Observation JSON whose tool list matches what
#   the trainer's system prompt advertises.
# - /step round-trips an action and returns the next observation.
#
# Note: each /step call without a session header gets a FRESH env, which
# is intentional in OpenEnv's HTTP design. Stateful multi-turn rollouts
# go through /ws (WebSocket) or the EnvClient in client.py.

set -euo pipefail
cd "$(dirname "$0")/.."

PYTHONPATH=. nohup .venv/Scripts/python.exe -m uvicorn server.app:app \
    --host 127.0.0.1 --port 8000 --log-level warning \
    > /tmp/uvicorn.log 2>&1 &
SERVER_PID=$!
trap 'kill $SERVER_PID 2>/dev/null || true' EXIT

# Wait for /health.
for _ in $(seq 1 40); do
    if curl -sf -o /dev/null http://127.0.0.1:8000/health; then break; fi
    sleep 0.25
done

echo "--- /health ---"
curl -s http://127.0.0.1:8000/health
echo

echo "--- /reset seed=0 ---"
curl -s -X POST http://127.0.0.1:8000/reset \
    -H "Content-Type: application/json" \
    -d '{"seed":0}' | head -c 600
echo

echo "--- /step read_log ---"
curl -s -X POST http://127.0.0.1:8000/step \
    -H "Content-Type: application/json" \
    -d '{"action":{"tool":"read_log","args":{"path":"/var/log/nginx/error.log"}}}' \
    | head -c 400
echo

echo "--- /metadata ---"
curl -s http://127.0.0.1:8000/metadata | head -c 400
echo

echo "ok"
