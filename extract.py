"""Python re의 정보 추출/정규화와 의심 식별자의 마스킹."""

from datetime import date
from decimal import Decimal
import re
from urllib.parse import urlsplit, urlunsplit

# ASCII 비인용 local-part, 하위 도메인과2~63자 영문 TLD를 허용한다.
EMAIL = re.compile(
    r"(?<![\w.+-])[A-Za-z0-9._%+-]+@(?:[A-Za-z0-9-]+\.)+[A-Za-z]{2,63}(?![\w-])"
)
# 국내 지역/이동전화와+82 표기를 처리하며 긴 숫자열 일부를 잡지 않는다.
PHONE = re.compile(
    r"(?<!\d)(?:\+82[- .]?(?:2|1[016789]|[3-6]\d|70)|0(?:2|1[016789]|[3-6]\d|70))[- .]?\d{3,4}[- .]?\d{4}(?!\d)"
)
# 연-월-일 순서의숫자/한국어 구분자를 허용하고 달력 검증은date에서 수행한다.
DATE = re.compile(
    r"(?<!\d)(?P<y>\d{4})\s*(?:[-/.]|년)\s*(?P<m>\d{1,2})\s*(?:[-/.]|월)\s*(?P<d>\d{1,2})(?:\s*일)?(?!\d)"
)
# 숫자와억/만/천/백/십이 이어진 원화 표현을 잡는다. 한글 수사는 범위 밖이다.
KRW = re.compile(
    r"(?<![\dA-Za-z.,])\d[\d,]*(?:\.\d+)?(?:\s*[억만천백십]\s*(?:\d[\d,]*(?:\.\d+)?)?)*\s*원"
)
# 달러의기호/코드/한국어 표기와 최대소수2자리를 허용한다.
USD = re.compile(
    r"(?<![\w.])(?:(?:\$\s*|USD\s*)\d[\d,]*(?:\.\d{1,2})?|\d[\d,]*(?:\.\d{1,2})?\s*달러)(?!\d|[.,]\d)",
    re.I,
)
# http/https의 경로/질의/fragment를 유지하고 문장 말미 구두점은 별도로 다듬는다.
URL = re.compile(r"\bhttps?://[^\s<>'\"]+", re.I)


def korean_amount(raw):
    """작은 단위의 절을만/억 단위와 결합해 원화 정수를 계산한다."""
    for piece in re.findall(r"\d[\d,]*(?:\.\d+)?", raw):
        if "," in piece and not re.fullmatch(r"\d{1,3}(?:,\d{3})+(?:\.\d+)?", piece):
            raise ValueError("잘못된 천단위 구분")
    raw = raw.replace(",", "").replace(" ", "").removesuffix("원")
    total = Decimal(0)
    section = Decimal(0)
    number = None
    for token in re.findall(r"\d+(?:\.\d+)?|[억만천백십]", raw):
        if token[0].isdigit():
            number = Decimal(token)
        elif token in "천백십":
            section += (number if number is not None else 1) * {
                "천": 1000,
                "백": 100,
                "십": 10,
            }[token]
            number = None
        else:
            subtotal = section + (number if number is not None else 0)
            total += (subtotal or 1) * {"만": 10000, "억": 100000000}[token]
            section = Decimal(0)
            number = None
    value = total + section + (number if number is not None else 0)
    if value != value.to_integral_value():
        raise ValueError("원 미만 금액")
    return int(value)


def phone_value(raw):
    """국제 국가코드를 국내0 접두어로 바꾸고 하이픈을 통일한다."""
    digits = re.sub(r"\D", "", raw)
    if raw.startswith("+82"):
        digits = "0" + digits[2:]
    prefix = 2 if digits.startswith("02") else 3
    middle = digits[prefix:-4]
    if len(middle) not in (3, 4):
        raise ValueError("전화 길이")
    return digits[:prefix] + "-" + middle + "-" + digits[-4:]


def url_value(raw):
    """괄호가 과하게 닫힌 문장 말미를 제외하고 경로 대소문자를 보존한다."""
    result = raw.rstrip(".,;!?")
    while result and any(
        result.endswith(close) and result.count(close) > result.count(opening)
        for opening, close in (("(", ")"), ("[", "]"), ("{", "}"))
    ):
        result = result[:-1]
    parts = urlsplit(result)
    if not parts.hostname:
        raise ValueError("URL 호스트 없음")
    # userinfo는원형 보존하고 호스트/포트 부분만소문자로 만든다.
    auth, separator, host = parts.netloc.rpartition("@")
    netloc = auth + separator + host.lower() if separator else parts.netloc.lower()
    return urlunsplit(
        (parts.scheme.lower(), netloc, parts.path, parts.query, parts.fragment)
    )


def extract(text):
    """원문 범위와 일관된 타입의 정규화 값을 반환한다."""
    results = []
    for kind, pattern in (
        ("email", EMAIL),
        ("phone", PHONE),
        ("date", DATE),
        ("money_krw", KRW),
        ("money_usd", USD),
        ("url", URL),
    ):
        for match in pattern.finditer(text):
            raw = match.group()
            try:
                if kind == "email":
                    local, domain = raw.split("@")
                    if local.startswith(".") or local.endswith(".") or ".." in local:
                        continue
                    value = local + "@" + domain.lower()
                elif kind == "phone":
                    value = phone_value(raw)
                elif kind == "date":
                    value = date(
                        int(match["y"]), int(match["m"]), int(match["d"])
                    ).isoformat()
                elif kind == "money_krw":
                    value = {"amount_minor": korean_amount(raw), "currency": "KRW"}
                elif kind == "money_usd":
                    number_raw = re.search(r"\d[\d,]*(?:\.\d+)?", raw).group()
                    if "," in number_raw and not re.fullmatch(
                        r"\d{1,3}(?:,\d{3})+(?:\.\d+)?", number_raw
                    ):
                        continue
                    number = number_raw.replace(",", "")
                    value = {
                        "amount_minor": int(Decimal(number) * 100),
                        "currency": "USD",
                    }
                else:
                    value = url_value(raw)
            except (ValueError, OverflowError):
                continue
            results.append(
                {
                    "kind": "money" if kind.startswith("money_") else kind,
                    "raw": raw,
                    "value": value,
                    "start": match.start(),
                    "end": match.end(),
                }
            )
    return sorted(results, key=lambda r: (r["start"], r["kind"]))


# 주민번호 모양은 체크섬/실재 여부를판정하지 않고 의심 문자열을 모두가린다.
RESIDENT = re.compile(r"(?<!\d)\d{6}[- ]?[1-8]\d{6}(?!\d)")
# 공백/하이픈이있는16자리도포함한다. 실제발급정보로해석하지않는다.
CARD = re.compile(r"(?<!\d)(?:\d{4}[- ]?){3}\d{4}(?!\d)")
# 시/도, 구/군과 도로명/건물번호 형태에한정한 예시주소 패턴이다.
ADDRESS = re.compile(
    r"(?:서울특별시|부산광역시|대구광역시|인천광역시|광주광역시|대전광역시|울산광역시|경기도)\s+[가-힣]+(?:구|군|시)\s+[가-힣0-9]+(?:로|길)\s+\d+(?:-\d+)?"
)


def mask_pii(text):
    """연락처와 추가 식별자 패턴을 구간 겹침 없이 마스킹한다."""
    candidates = []
    for kind, pattern in (("resident", RESIDENT), ("card", CARD), ("address", ADDRESS)):
        for match in pattern.finditer(text):
            candidates.append((match.start(), match.end(), f"[{kind.upper()}]"))
    for item in extract(text):
        if item["kind"] in ("email", "phone"):
            candidates.append((item["start"], item["end"], f"[{item['kind'].upper()}]"))
    chosen = []
    for item in sorted(candidates, key=lambda r: (-(r[1] - r[0]), r[0])):
        if not any(item[0] < end and start < item[1] for start, end, _ in chosen):
            chosen.append(item)
    result = text
    for start, end, replacement in sorted(chosen, reverse=True):
        result = result[:start] + replacement + result[end:]
    return result
