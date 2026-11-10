import re
import unicodedata

# Preserve technical identifiers such as ERR_42, python3.11 and c++.
TOKEN = re.compile(r"[^\W_]+(?:[_.+#-][^\W_]+)*(?:\+\+|#)?", re.UNICODE)


def tokenize(text: str) -> list[str]:
    return TOKEN.findall(unicodedata.normalize("NFKC", text).casefold())
