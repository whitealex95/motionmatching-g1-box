#!/usr/bin/env python3
"""Serve the local comparison with HTTP byte ranges for video seeking."""
import argparse
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import re


class Handler(SimpleHTTPRequestHandler):
    def do_GET(self):
        path = Path(self.translate_path(self.path))
        request = self.headers.get('Range')
        if not request or path.suffix != '.mp4' or not path.is_file():
            return super().do_GET()
        size = path.stat().st_size
        match = re.fullmatch(r'bytes=(\d*)-(\d*)', request)
        if not match or not any(match.groups()):
            self.send_error(416, 'Unsupported byte range')
            return
        first, last = match.groups()
        start = int(first) if first else max(0, size-int(last))
        end = min(int(last), size-1) if first and last else size-1
        if start > end or start >= size:
            self.send_response(416)
            self.send_header('Content-Range', f'bytes */{size}')
            self.end_headers()
            return
        self.send_response(206)
        self.send_header('Content-Type', 'video/mp4')
        self.send_header('Accept-Ranges', 'bytes')
        self.send_header('Content-Range', f'bytes {start}-{end}/{size}')
        self.send_header('Content-Length', str(end-start+1))
        self.end_headers()
        try:
            with path.open('rb') as f:
                f.seek(start)
                remaining = end-start+1
                while remaining:
                    chunk = f.read(min(1024*1024, remaining))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    remaining -= len(chunk)
        except (BrokenPipeError, ConnectionResetError):
            pass


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--port', type=int, default=8766)
    args = ap.parse_args()
    site = Path(__file__).resolve().parent / 'site'
    if not (site / 'data.json').exists():
        raise SystemExit('Run python3 eval_supermimic/build_site.py first')
    server = ThreadingHTTPServer(('127.0.0.1', args.port), partial(Handler, directory=str(site)))
    print(f'Local comparison: http://localhost:{args.port}', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == '__main__':
    main()
