import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='[%(asctime)s:], %(message)s')

list_files  = [
    "src/__init__.py",
    "src/helper.py",
    "src/prompt.py",
    ".env",
    "setup.py",
    "research/trials.ipynb",
    "app.py",
    "store_index.py",
    "static/.gitkeep",
    "templates/chat.html",
    "templates/style.css"
]

for files_ in list_files:
    filepath = Path(files_)
    folder, file = os.path.split(filepath)
    if folder != "":
        os.makedirs(folder, exist_ok=True)
        logging.info(f"Creating FOLDER: {folder}")
    if (not os.path.exists(filepath)):
        with open(filepath, 'w') as f:
            pass
            logging.info(f"Created FILE: {file} in FILEPATH: {filepath}")
    else:
        logging.info(f"Filepath {filepath} allready EXISTS!")