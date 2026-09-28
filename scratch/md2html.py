import markdown
import sys
import os
import base64
import re

def img_to_base64(path):
    path = path.lstrip('/')
    if not os.path.exists(path):
        return path
    with open(path, "rb") as f:
        data = f.read()
    b64 = base64.b64encode(data).decode('utf-8')
    ext = path.split('.')[-1].lower()
    return f"data:image/{ext};base64,{b64}"

def convert(md_file, html_file):
    with open(md_file, "r", encoding="utf-8") as f:
        text = f.read()
    
    def replacer(match):
        alt = match.group(1)
        path = match.group(2)
        b64_path = img_to_base64(path)
        return f"![{alt}]({b64_path})"
        
    text = re.sub(r'!\[([^\]]*)\]\(([^\)]+)\)', replacer, text)
    
    html_content = markdown.markdown(text, extensions=['tables'])
    full_html = f"""
    <html>
    <head>
    <style>
        body {{ font-family: 'Segoe UI', Arial, sans-serif; padding: 40px; max-width: 800px; margin: auto; }}
        h1, h2, h3 {{ color: #333; }}
        img {{ max-width: 100%; border: 1px solid #ddd; padding: 5px; border-radius: 5px; }}
        ul {{ line-height: 1.6; }}
    </style>
    </head>
    <body>
    {html_content}
    </body>
    </html>
    """
    with open(html_file, "w", encoding="utf-8") as f:
        f.write(full_html)

if __name__ == "__main__":
    convert(sys.argv[1], sys.argv[2])
