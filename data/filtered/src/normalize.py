import re
import ftfy

_WS = re.compile(r"\s+")
_DEHYPH = re.compile(r"(\w)-\s*\n\s*(\w)")

def normalize_text(text: str) -> str:
    text = ftfy.fix_text(text)
    text = _DEHYPH.sub(r"\1\2", text)
    text = text.replace("\u00ad", "")
    text = text.replace("\r", "\n")
    text = _WS.sub(" ", text).strip()
    return text