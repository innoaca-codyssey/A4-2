"""외부 사전의 점수와 직접 정의한 문맥/Aspect규칙."""

import json
from pathlib import Path
import re
from preprocess import sequence

RESOURCES = Path(__file__).parent / "resources"
BASE = json.loads((RESOURCES / "base-lexicon.json").read_text())
DOMAIN = json.loads((RESOURCES / "domain-additions.json").read_text())
NEGATORS = {
    "not",
    "no",
    "never",
    "neither",
    "nor",
    "cannot",
    "hardly",
    "barely",
    "scarcely",
}
BOOST = {"very": 1.5, "really": 1.3, "extremely": 1.8, "quite": 1.2, "slightly": 0.7}
ASPECTS = {
    "story": {"story", "plot", "script", "screenplay", "narrative"},
    "acting": {"acting", "actor", "actors", "performance", "performances", "cast"},
    "visual": {"visual", "visuals", "effect", "effects", "cinematography"},
    "sound": {"music", "sound", "soundtrack"},
    "food": {"food", "taste"},
    "price": {"price", "cost"},
}


def normalize_korean(text):
    """예시의한국어 부정/강조 패턴만 명시적으로 영어 토큰에 대응한다."""
    result = str(text)
    for pattern, replacement in (
        (r"좋지\s*않(?:다|아요|습니다|음)?", "not good"),
        (r"나쁘지\s*않(?:다|아요|습니다|음)?", "not bad"),
    ):
        result = re.sub(pattern, replacement, result)
    replacements = {
        "맛있다": "tasty",
        "맛있어요": "tasty",
        "비싸다": "expensive",
        "비싸요": "expensive",
        "재미없다": "boring",
        "재미있다": "entertaining",
        "좋아요": "good",
        "좋다": "good",
        "좋은": "good",
        "나쁘다": "bad",
        "나쁜": "bad",
        "매우": "very",
        "정말": "really",
        "아주": "very",
    }
    for old, new in replacements.items():
        result = result.replace(old, " " + new + " ")
    result = re.sub(r"(?<![가-힣])안(?=\s|good\b|bad\b|not\b)", "not", result)
    for old, new in (
        (r"음식(?:은|는|이|가)?", "food"),
        (r"가격(?:은|는|이|가)?", "price"),
        (r"연기(?:은|는|이|가)?", "acting"),
        (r"하지만|그러나", "but"),
    ):
        result = re.sub(old, " " + new + " ", result)
    return result


def analyze(text, rules=True, domain=True):
    """기본 합계와 부정/강조/역접을 적용한 점수를 반환한다."""
    tokens = sequence(normalize_korean(text))
    lexicon = {**BASE, **(DOMAIN if domain else {}), "tasty": 2.0, "expensive": -1.5}
    contributions = []
    boundary = 0
    contrast = next((i for i, t in enumerate(tokens) if t in ("but", "however")), None)
    for i, token in enumerate(tokens):
        if token in ".!?;" or token in ("but", "however"):
            boundary = i + 1
        if token not in lexicon:
            continue
        value = float(lexicon[token])
        negations = 0
        factor = 1.0
        if rules:
            context = tokens[max(boundary, i - 3) : i]
            negations = sum(t in NEGATORS or t.endswith("n't") for t in context)
            if negations % 2:
                value = -value
            if i > 0:
                factor *= BOOST.get(tokens[i - 1], 1.0)
            if contrast is not None:
                factor *= 0.5 if i < contrast else 1.5
        value *= factor
        contributions.append(
            {
                "token": token,
                "position": i,
                "score": value,
                "negations": negations,
                "factor": factor,
            }
        )
    score = sum(c["score"] for c in contributions)
    return {
        "score": score,
        "label": int(score > 0),
        "tokens": tokens,
        "contributions": contributions,
    }


def aspects(text):
    """감성 단어와 같은 문장 안의 가까운Aspect를 연결한다."""
    result = analyze(text)
    tokens = result["tokens"]
    scores = {}
    details = []
    locations = [
        (i, aspect)
        for i, token in enumerate(tokens)
        for aspect, words in ASPECTS.items()
        if token in words
    ]
    for contribution in result["contributions"]:
        i = contribution["position"]
        candidates = []
        for j, aspect in locations:
            left, right = sorted((i, j))
            if abs(i - j) <= 5 and not any(
                t in ".!?;" or t in ("but", "however") for t in tokens[left + 1 : right]
            ):
                candidates.append((abs(i - j), j, aspect))
        if candidates:
            _, _, aspect = min(candidates)
            scores[aspect] = scores.get(aspect, 0) + contribution["score"]
            details.append({**contribution, "aspect": aspect})
    return {
        "aspects": {
            key: {
                "score": value,
                "sentiment": "positive"
                if value > 0
                else "negative"
                if value < 0
                else "neutral",
            }
            for key, value in scores.items()
        },
        "contributions": details,
    }
