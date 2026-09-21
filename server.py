import json
import os
import threading
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler

from q_learning_2x2 import (
    QLearningAgent, ACTIONS, train_curriculum, export_agent, solve_once,
)

PORT = 8000
AGENT_PATH = "trained_agent.json"
MAX_DEPTH = 8

state_lock = threading.Lock()
state = {"agent": None, "depth_results": {}, "training": False}


class PolicyAgent:
    """Lightweight stand-in for QLearningAgent at serve time: we only ever
    need the greedy action for a state, so we load just {state: action}
    instead of the full Q-value table. Falls back to action 0 for a state
    that was never visited during training, matching the tie-break
    behavior of an all-zero Q-row in the original agent."""
    def __init__(self, policy: dict):
        self.policy = policy

    def choose_action(self, key, greedy=True):
        return self.policy.get(key, 0)


def load_or_train_agent():
    if os.path.exists(AGENT_PATH):
        print(f"Loading existing trained agent from {AGENT_PATH} ...")
        with open(AGENT_PATH) as f:
            payload = json.load(f)
        state["agent"] = PolicyAgent(payload["policy"])
        state["depth_results"] = payload.get("depth_results", {})
        print(f"Loaded {len(payload['policy'])} states.")
    else:
        retrain()


def retrain():
    print(f"Training a fresh agent up to depth {MAX_DEPTH} "
          f"(this takes roughly 30-70 seconds)...")
    agent, depth_results = train_curriculum(max_depth=MAX_DEPTH)
    export_agent(agent, depth_results, MAX_DEPTH, path=AGENT_PATH)
    with open(AGENT_PATH) as f:
        payload = json.load(f)
    with state_lock:
        state["agent"] = PolicyAgent(payload["policy"])
        state["depth_results"] = payload.get("depth_results", {})
    print("Training complete, agent reloaded.")


class Handler(SimpleHTTPRequestHandler):
    def _send_json(self, obj, status=200):
        body = json.dumps(obj).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        if self.path.startswith("/solve"):
            with state_lock:
                busy = state["training"]
                agent = state["agent"]
            if busy:
                self._send_json({"error": "agent is retraining, try again shortly"}, 409)
                return
            result = solve_once(agent, MAX_DEPTH)
            with open("solution.json", "w") as f:
                json.dump(result, f, indent=2)
            self._send_json(result)
            return

        if self.path.startswith("/retrain"):
            with state_lock:
                if state["training"]:
                    self._send_json({"error": "already retraining"}, 409)
                    return
                state["training"] = True
            try:
                retrain()
                self._send_json({
                    "ok": True,
                    "depth_results": state["depth_results"],
                })
            finally:
                with state_lock:
                    state["training"] = False
            return

        self._send_json({"error": "not found"}, 404)

    def do_GET(self):
        if self.path.startswith("/status"):
            with state_lock:
                self._send_json({
                    "training": state["training"],
                    "depth_results": state["depth_results"],
                })
            return
        # everything else (the HTML page, solution.json, etc.) is served
        # as a plain static file from the current directory
        super().do_GET()

    def log_message(self, fmt, *args):
        print("[server]", fmt % args)


if __name__ == "__main__":
    load_or_train_agent()
    httpd = ThreadingHTTPServer(("localhost", PORT), Handler)
    print(f"\nServing on http://localhost:{PORT}  (Ctrl+C to stop)")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping.")
