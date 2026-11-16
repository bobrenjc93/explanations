"""Assemble the third write-up: prose from prose4.py, figures from figs4.py."""
import sys
import prose4 as prose
from charts import CSS, JS

OUT = sys.argv[1] if len(sys.argv) > 1 else "page/draft4.html"
html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{prose.TITLE}</title>
<meta name="description" content="{prose.DESCRIPTION}">
<link rel="icon" href="../favicon.svg" type="image/svg+xml">
<link rel="icon" href="../favicon.ico" sizes="any">
<link rel="apple-touch-icon" href="../apple-touch-icon.png">
<style>{CSS}</style>
</head>
<body>
{prose.body()}
<script>{JS}</script>
</body>
</html>
"""
open(OUT, "w").write(html)
print("wrote", OUT, len(html) // 1024, "KB")
