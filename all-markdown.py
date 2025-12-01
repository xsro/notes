from pathlib import Path
import re
import urllib.parse
from copy import copy
import os

__proj=Path(__file__).parent


text=""
for md in __proj.glob("**/*.md"):
    rel=md.relative_to(__proj)
    text+=f"- [{rel}]({rel})"+"\n"

__proj.joinpath("all-markdown.md").write_text(text,encoding="utf-8")