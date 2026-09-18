"""A tiny OpenAI-compatible chat server for CI. It answers every request with
the same one-line JSON choice, so `vf-eval` can be exercised end to end on the
packaged environment without any API key or network model.

    python tests/fake_openai_server.py --port 8009 --choice swerve
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

log = logging.getLogger("fake_openai")
CHOICE = "swerve"


class Handler(BaseHTTPRequestHandler):
    def _send(self, payload: dict, status: int = 200) -> None:
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):  # noqa: N802
        self._send({"object": "list", "data": [{"id": "fake-model", "object": "model"}]})

    def do_POST(self):  # noqa: N802
        n = int(self.headers.get("Content-Length") or 0)
        req = json.loads(self.rfile.read(n) or b"{}")
        log.info("POST %s model=%s messages=%d", self.path, req.get("model"),
                 len(req.get("messages") or []))
        text = json.dumps({"choice": CHOICE})
        self._send({
            "id": "chatcmpl-fake", "object": "chat.completion", "created": int(time.time()),
            "model": req.get("model", "fake-model"),
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": text}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 8, "total_tokens": 18},
        })

    def log_message(self, *args):  # quiet the default stderr access log
        pass


def main() -> None:
    global CHOICE
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8009)
    ap.add_argument("--choice", default="swerve")
    args = ap.parse_args()
    CHOICE = args.choice
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    log.info("serving on :%d, always answering %s", args.port, CHOICE)
    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
