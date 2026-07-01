#!/usr/bin/env python3
"""
LeafAlert Feedback Sync Server

Receives feedback data from the iOS app over Wi-Fi.
Advertises itself via Bonjour (_leafalert._tcp) so the app can discover it automatically.

Usage:
    python3 scripts/feedback_server.py [--port 8847] [--output-dir feedback]

Files are saved to:
    <output-dir>/
        ├── <timestamp>_<label>_<status>.jpg
        └── manifest.json
"""

import argparse
import json
import os
import signal
import socket
import sys
import threading
from datetime import datetime
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path

# Maximum accepted request body size (bytes). Guards against memory DoS.
MAX_BODY_SIZE = 25 * 1024 * 1024  # 25 MB

# Bonjour registration via pyobjc (macOS built-in)
try:
    from Foundation import NSNetService, NSRunLoop, NSDate
    HAS_BONJOUR = True
except ImportError:
    HAS_BONJOUR = False


class FeedbackHandler(BaseHTTPRequestHandler):
    """Handles feedback uploads from the LeafAlert iOS app."""

    def do_POST(self):
        if self.path == "/upload":
            self._handle_upload()
        elif self.path == "/ping":
            self._respond(200, {"status": "ok"})
        else:
            self._respond(404, {"error": "not found"})

    def do_GET(self):
        if self.path == "/ping":
            self._respond(200, {"status": "ok", "service": "leafalert-feedback"})
        elif self.path == "/status":
            manifest = self._read_manifest()
            count = len(manifest.get("entries", []))
            self._respond(200, {"status": "ok", "entries": count})
        else:
            self._respond(404, {"error": "not found"})

    def _handle_upload(self):
        content_type = self.headers.get("Content-Type", "")

        if "multipart/form-data" in content_type:
            self._handle_multipart()
        elif "application/json" in content_type:
            self._handle_json_batch()
        else:
            self._respond(400, {"error": "unsupported content type"})

    def _read_body(self):
        """Read the request body, validating Content-Length and capping size.

        Returns the body bytes, or None if a response was already sent
        (bad header -> 400, oversized -> 413).
        """
        raw_length = self.headers.get("Content-Length", "0")
        try:
            content_length = int(raw_length)
        except (TypeError, ValueError):
            self._respond(400, {"error": "invalid Content-Length"})
            return None

        if content_length < 0:
            self._respond(400, {"error": "invalid Content-Length"})
            return None

        if content_length > MAX_BODY_SIZE:
            self._respond(413, {"error": "request body too large"})
            return None

        return self.rfile.read(content_length)

    def _safe_image_path(self, output_dir, filename):
        """Resolve a client-supplied filename to a path safely contained in
        output_dir. Returns the path, or None if the name is unsafe (a
        response is sent in that case)."""
        # Only ever trust the basename — strip any directory components.
        base = Path(filename).name
        if not base or base in (".", "..") or base.startswith("."):
            self._respond(400, {"error": "invalid filename"})
            return None

        image_path = output_dir / base
        resolved_dir = output_dir.resolve()
        resolved_path = image_path.resolve()
        if not resolved_path.is_relative_to(resolved_dir):
            self._respond(400, {"error": "invalid filename"})
            return None

        return image_path

    def _handle_multipart(self):
        """Handle multipart upload: image file + JSON metadata."""
        content_type = self.headers["Content-Type"]
        body = self._read_body()
        if body is None:
            return

        # Parse boundary
        boundary = content_type.split("boundary=")[-1].strip()
        parts = self._parse_multipart(body, boundary.encode())

        metadata = None
        image_data = None
        image_filename = None

        for part in parts:
            disposition = part.get("disposition", "")
            if 'name="metadata"' in disposition:
                metadata = json.loads(part["body"].decode("utf-8"))
            elif 'name="image"' in disposition:
                image_data = part["body"]
                # Extract filename from disposition
                for token in disposition.split(";"):
                    token = token.strip()
                    if token.startswith("filename="):
                        image_filename = token.split("=", 1)[1].strip('"')

        if metadata is None:
            self._respond(400, {"error": "missing metadata"})
            return

        output_dir = Path(self.server.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save image
        raw_filename = image_filename or metadata.get("filename", f"upload_{datetime.now().isoformat()}.jpg")
        # Sanitize: take only the basename and confirm containment in output_dir.
        image_path = self._safe_image_path(output_dir, raw_filename)
        if image_path is None:
            return
        filename = image_path.name
        if image_data:
            image_path.write_bytes(image_data)
            print(f"  Saved image: {filename} ({len(image_data)} bytes)")

        # Append to manifest
        manifest_path = output_dir / "manifest.json"
        manifest = self._read_manifest()
        entries = manifest.get("entries", [])

        # Deduplicate by filename
        existing = {e.get("filename") for e in entries if e.get("filename") is not None}
        if filename not in existing:
            entry = {
                "filename": filename,
                "originalPrediction": metadata.get("originalPrediction", ""),
                "correctedLabel": metadata.get("correctedLabel", ""),
                "feedbackStatus": metadata.get("feedbackStatus", ""),
                "confidence": metadata.get("confidence", 0),
                "timestamp": metadata.get("timestamp", ""),
                "latitude": metadata.get("latitude", 0),
                "longitude": metadata.get("longitude", 0),
            }
            entries.append(entry)
            manifest["entries"] = entries
            manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
            print(f"  Added to manifest (total: {len(entries)} entries)")

        self._respond(200, {"status": "ok", "filename": filename})

    def _handle_json_batch(self):
        """Handle a batch metadata sync (no images)."""
        raw_body = self._read_body()
        if raw_body is None:
            return
        try:
            body = json.loads(raw_body)
        except (json.JSONDecodeError, UnicodeDecodeError):
            self._respond(400, {"error": "invalid JSON"})
            return
        entries = body.get("entries", [])

        output_dir = Path(self.server.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        manifest = self._read_manifest()
        existing = {
            e.get("filename")
            for e in manifest.get("entries", [])
            if e.get("filename") is not None
        }
        added = 0
        for entry in entries:
            if entry.get("filename") not in existing:
                manifest.setdefault("entries", []).append(entry)
                added += 1

        manifest_path = output_dir / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
        self._respond(200, {"status": "ok", "added": added})

    def _read_manifest(self):
        manifest_path = Path(self.server.output_dir) / "manifest.json"
        if manifest_path.exists():
            return json.loads(manifest_path.read_text())
        return {"version": 1, "entries": []}

    def _parse_multipart(self, body, boundary):
        """Simple multipart parser."""
        parts = []
        delimiter = b"--" + boundary
        segments = body.split(delimiter)

        for segment in segments:
            if segment in (b"", b"--\r\n", b"--"):
                continue
            segment = segment.strip(b"\r\n")
            if segment == b"--":
                continue

            header_end = segment.find(b"\r\n\r\n")
            if header_end == -1:
                continue

            header_section = segment[:header_end].decode("utf-8", errors="replace")
            body_section = segment[header_end + 4:]
            # Strip trailing \r\n
            if body_section.endswith(b"\r\n"):
                body_section = body_section[:-2]

            disposition = ""
            for line in header_section.split("\r\n"):
                if line.lower().startswith("content-disposition:"):
                    disposition = line.split(":", 1)[1].strip()

            parts.append({"disposition": disposition, "body": body_section})

        return parts

    def _respond(self, code, data):
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())

    def log_message(self, format, *args):
        # Quieter logging — only show uploads, not every request
        msg = format % args
        if "POST" in msg:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")


def get_local_ip():
    """Get the Mac's local network IP."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


def register_bonjour(port):
    """Register the service via Bonjour so iOS can discover it."""
    if not HAS_BONJOUR:
        print("  (pyobjc not available — Bonjour disabled, use IP address)")
        return None

    service = NSNetService.alloc().initWithDomain_type_name_port_(
        "", "_leafalert._tcp.", "LeafAlert Feedback", port
    )
    service.publish()

    # Run the run loop briefly to process the publish
    def run_bonjour():
        while True:
            NSRunLoop.currentRunLoop().runUntilDate_(
                NSDate.dateWithTimeIntervalSinceNow_(0.5)
            )

    thread = threading.Thread(target=run_bonjour, daemon=True)
    thread.start()

    print(f"  Bonjour: advertising as _leafalert._tcp on port {port}")
    return service


def main():
    parser = argparse.ArgumentParser(description="LeafAlert Feedback Sync Server")
    parser.add_argument("--port", type=int, default=8847, help="Port to listen on")
    parser.add_argument("--output-dir", default="feedback", help="Directory to save feedback")
    args = parser.parse_args()

    # Resolve output dir relative to project root
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    output_dir = project_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    local_ip = get_local_ip()

    print("=" * 50)
    print("LeafAlert Feedback Sync Server")
    print("=" * 50)
    print(f"  Saving to:  {output_dir}")
    print(f"  Address:    http://{local_ip}:{args.port}")
    print()

    server = HTTPServer(("0.0.0.0", args.port), FeedbackHandler)
    server.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.output_dir = str(output_dir)

    bonjour_service = register_bonjour(args.port)

    print()
    print("Waiting for feedback from LeafAlert app...")
    print("Press Ctrl+C to stop.")
    print()

    def shutdown(sig, frame):
        print("\nShutting down...")
        if bonjour_service:
            bonjour_service.stop()
        server.shutdown()
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        shutdown(None, None)


if __name__ == "__main__":
    main()
