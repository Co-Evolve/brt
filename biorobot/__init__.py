import atexit
import shutil
from pathlib import Path

brb_tmp_directory = ""


def set_brb_tmp_directory(new_path: str) -> None:
    global brb_tmp_directory
    brb_tmp_directory = new_path
    Path(brb_tmp_directory).mkdir(parents=True, exist_ok=True)


set_brb_tmp_directory("/tmp/brb")


def exit_handler():
    global brb_tmp_directory
    if Path(brb_tmp_directory).exists():
        shutil.rmtree(brb_tmp_directory, ignore_errors=True)


atexit.register(exit_handler)
