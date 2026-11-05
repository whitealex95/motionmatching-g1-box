#!/usr/bin/env python3
"""Download and verify the exact demo bundle used by the local adapter."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
import urllib.request


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', type=Path, default=Path(__file__).resolve().parents[2] / 'assets/supermimic/demo')
    args = ap.parse_args()
    manifest = json.loads(Path(__file__).with_name('assets_manifest.json').read_text())
    args.out.mkdir(parents=True, exist_ok=True)
    for entry in manifest['files']:
        dest = args.out / entry['file']
        if dest.exists() and hashlib.sha256(dest.read_bytes()).hexdigest() == entry['sha256']:
            print('verified', dest)
            continue
        with urllib.request.urlopen(entry['url'], timeout=90) as response:
            contents = response.read()
        if len(contents) != entry['bytes'] or hashlib.sha256(contents).hexdigest() != entry['sha256']:
            raise RuntimeError(f"Demo version changed: {entry['url']}. Existing assets were preserved.")
        with tempfile.NamedTemporaryFile(dir=args.out, delete=False) as temp:
            temp.write(contents)
            name = temp.name
        os.replace(name, dest)
        print('downloaded', dest)


if __name__ == '__main__':
    main()
