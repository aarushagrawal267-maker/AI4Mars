"""Convert frontend/dist-demo/index.html into the page format of a claude.ai artifact.

The host wraps the page in its own <!doctype>/<html>/<head>/<body> skeleton, so only the title, styles, scripts
and the root element are kept. Run after `VITE_DEMO=1 npx vite build` (inside frontend/).
"""
import re
import sys
from pathlib import Path

p = Path(__file__).resolve().parents[1] / "frontend" / "dist-demo" / "index.html"
s = p.read_text(encoding="utf-8")
if "<head>" not in s:
    sys.exit("already finalized")
head = re.search(r"<head>(.*)</head>", s, re.S).group(1)
body = re.search(r"<body>(.*)</body>", s, re.S).group(1)
for pattern in (r'\s*<meta charset="UTF-8" />', r'\s*<meta name="viewport"[^>]*>', r'\s*<link rel="icon"[^>]*>'):
    head = re.sub(pattern, "", head)
title = re.search(r"<title>.*?</title>", head).group(0)
head = head.replace(title, "").strip()
page = (f"{title}\n{head}\n"
        '<style>html,body{background:rgb(9 10 12)}:root[data-theme="light"] body{background:rgb(242 240 236)}</style>\n'
        f"{body.strip()}\n")
p.write_text(page, encoding="utf-8")
print(f"finalized {p} ({sum(1 for f in p.parent.rglob('*') if f.is_file())} files)")
