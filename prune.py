from pathlib import Path
import re
import urllib.parse
from copy import copy
import os

pics=os.listdir("pics")

links=[]

for md in Path.cwd().glob("**/*.md"):
    content=md.read_text(encoding="utf-8")
    img_links=re.findall(r"!\[.*?\]\((.*?)\)",content)
    
    for img_link in img_links:
        if img_link.startswith("http"):
            continue
        if "%" in img_link:
            content=content.replace(img_link,urllib.parse.unquote(img_link))
    md.write_text(content,encoding="utf-8")

    for img_link in img_links:
        if img_link.startswith("http"):
            continue
        pars=img_link.count("..")
        img_path=copy(md)
        for p in range(pars):
            img_path=img_path.parent
            img_link=img_link.replace("..",".")
        img_path=img_path.parent.joinpath(img_link)
        if not img_path.exists():
            print(md,img_path)
        base=os.path.basename(img_path)
        if base in pics:
            pics.pop(pics.index(base))
for pic in pics:
    if pic != "meme":
        print("remove",pic)
        os.remove("pics/"+pic)
