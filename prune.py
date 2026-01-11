from pathlib import Path
import re
import urllib.parse
from copy import copy
import os
import shutil

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
    
        p=Path(img_link)
        filename=p.name

        image_folder=md.parent.joinpath("image")
        if not image_folder.exists():
            image_folder.mkdir()

        src=Path(__file__).parent.joinpath("pics").joinpath(filename)
        dst=image_folder.joinpath(filename)
        if src.exists() and not dst.exists():
            shutil.move(src,dst)
        new_link=os.path.relpath(dst,md.parent).replace("\\","/")
        content=content.replace(img_link,new_link)

        print(f"Processed image link: {img_link} -> {new_link}")
        
    md.write_text(content,encoding="utf-8")
