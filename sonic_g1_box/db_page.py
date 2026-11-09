"""The driver's database page (web/): lmm-g1-box's motion database page with the driver's controller states, the clip
the robot plays now and the last matches above it. GET /api/live is the driver's `live` dict, which the driver
replaces at every state publish. The page's scripts ask for the data at lmm-g1-box's paths, and ROUTES maps them
onto data/ (the EMM takes' folder is named in config.json, which is a byte copy)."""
import http.server
import json
import os
import shutil
import threading
import urllib.parse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROUTES = (('data/emm_loco/holosoma-v3/', 'data/emm_loco/'), ('data/', 'data/'), ('', 'web/'))
TYPES = {'.html': 'text/html; charset=utf-8', '.js': 'text/javascript; charset=utf-8', '.css': 'text/css; charset=utf-8',
         '.json': 'application/json', '.bin': 'application/octet-stream', '.txt': 'text/plain; charset=utf-8'}


def resolve(path):
    """The file under ROOT for a URL path, or None for a path outside its route's folder."""
    rel = path.lstrip('/') or 'index.html'
    for prefix, folder in ROUTES:
        if rel.startswith(prefix):
            base = os.path.realpath(os.path.join(ROOT, folder))
            f = os.path.realpath(os.path.join(base, rel[len(prefix):]))
            return f if f.startswith(base + os.sep) and os.path.isfile(f) else None
    return None


class DbPage:
    def __init__(self, port, host='0.0.0.0'):
        self.live = {'ok': False}
        page = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_GET(self):
                path = urllib.parse.urlparse(self.path).path
                if path == '/api/live':
                    body = json.dumps(page.live).encode()
                    return self.reply(200, 'application/json', len(body), lambda: self.wfile.write(body), 'no-store')
                f = resolve(urllib.parse.unquote(path))
                if f is None:
                    return self.reply(404, 'text/plain', 9, lambda: self.wfile.write(b'not found'), 'no-store')
                cache = 'max-age=3600' if f.endswith('.bin') else 'no-cache'
                with open(f, 'rb') as fh:
                    self.reply(200, TYPES.get(os.path.splitext(f)[1], 'application/octet-stream'), os.fstat(fh.fileno()).st_size,
                               lambda: shutil.copyfileobj(fh, self.wfile, 1 << 20), cache)

            def reply(self, code, ctype, n, write, cache):
                try:
                    self.send_response(code)
                    self.send_header('Content-Type', ctype)
                    self.send_header('Content-Length', str(n))
                    self.send_header('Cache-Control', cache)
                    self.end_headers()
                    write()
                except (BrokenPipeError, ConnectionResetError):
                    pass

        self.server = http.server.ThreadingHTTPServer((host, port), Handler)
        self.server.daemon_threads = True
        self.port = port
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self):
        self.server.shutdown()
        self.server.server_close()
