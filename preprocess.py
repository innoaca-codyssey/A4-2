"""검토한 일반 어휘로 입력을 제한하고 벡터화용 토큰을 만든다."""

import json
from pathlib import Path
import re

RESOURCES = Path(__file__).parent / "resources"
APPROVED = set(json.loads((RESOURCES / "approved-vocabulary.json").read_text()))
WORD = r"[a-z]+(?:['-][a-z]+)*"
PIECES = re.compile(r"\[UNK\]|" + WORD + r"|[.!?;,]", re.I)
STOP = set(
    "a an the and or of to in on at for from by with as is am are was were be been being i me my we us our you your he him his she her they them their it its this that these those do does did have has had can could will would may might must should shall when where who which what how than then there here one two three first last also only own all each some any every such both either".split()
)
ALIASES = {
    "movies": "movie",
    "films": "film",
    "actors": "actor",
    "characters": "character",
    "performances": "performance",
    "effects": "effect",
    "scenes": "scene",
    "reviews": "review",
}


def sequence(text):
    """UNK의 위치와 문장 경계를 보존한 안전한 토큰을 반환한다."""
    return [
        "[UNK]" if t.lower() == "[unk]" else t
        for t in PIECES.findall(str(text).replace("’", "'").lower())
    ]


def sanitize(text):
    """미승인 단어를 가리고 연락처/숫자를 입력 표현에서 제외한다."""
    return " ".join(
        token if token in APPROVED or token in ".!?;," else "[UNK]"
        for token in sequence(text)
    )


def tokenize(text):
    """벡터화에는 승인 단어 중 불용어를 제외한 형태만 사용한다."""
    return [
        ALIASES.get(token, token)
        for token in sequence(text)
        if token in APPROVED and token not in STOP
    ]


def preview(text, limit=180):
    """가린 단어를 제외하고 남은 실제 어휘로 짧게 표시한다."""
    return " ".join(t for t in sequence(text) if t in APPROVED and t not in STOP)[:limit].rstrip()
