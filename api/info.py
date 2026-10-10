"""Vercel serverless endpoint: library metadata, zero dependencies.

Only the standard library is used here — heavy ML packages (torch,
transformers) cannot fit Vercel's serverless size limits, so this
handler never imports them.

Copyright (c) 2026 salim-slimani. MIT license.
"""
from http.server import BaseHTTPRequestHandler
import json

INFO = {
    "name": "modetrains",
    "version": "0.2.1",
    "tagline": "Fast, memory-efficient LLM fine-tuning.",
    "author": "salim-slimani",
    "license": "MIT",
    "features": [
        "4-bit QLoRA",
        "Flash / SDPA attention",
        "Sequence packing",
        "SFT / DPO / GRPO trainers",
        "LoRA merge + Hub push + GGUF notes",
        "Desktop app + CLI",
    ],
    "docs": "https://salim-studio.github.io/modetrains/",
    "repo": "https://github.com/salim-studio/modetrains",
}


def get_info() -> dict:
    """Pure payload builder (unit-testable without HTTP machinery)."""
    return dict(INFO)


class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        body = json.dumps(get_info()).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):  # keep function logs clean
        pass
