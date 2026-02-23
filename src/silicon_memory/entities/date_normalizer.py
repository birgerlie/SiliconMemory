"""Canonical date normalization helpers."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
import re


_MONTHS = {
    "january": 1,
    "jan": 1,
    "february": 2,
    "feb": 2,
    "march": 3,
    "mar": 3,
    "april": 4,
    "apr": 4,
    "may": 5,
    "june": 6,
    "jun": 6,
    "july": 7,
    "jul": 7,
    "august": 8,
    "aug": 8,
    "september": 9,
    "sep": 9,
    "sept": 9,
    "october": 10,
    "oct": 10,
    "november": 11,
    "nov": 11,
    "december": 12,
    "dec": 12,
}

_RE_ISO_DAY = re.compile(r"\b((?:19|20)\d{2})-(\d{1,2})-(\d{1,2})\b")
_RE_ISO_MONTH = re.compile(r"\b((?:19|20)\d{2})-(\d{1,2})\b")
_RE_YEAR = re.compile(r"\b((?:19|20)\d{2})\b")
_RE_MONTH_NAME_DAY = re.compile(
    r"\b("
    r"January|February|March|April|May|June|July|August|September|October|November|December|"
    r"Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec"
    r")\s+(\d{1,2}),?\s+((?:19|20)\d{2})\b",
    re.IGNORECASE,
)
_RE_MONTH_NAME_YEAR = re.compile(
    r"\b("
    r"January|February|March|April|May|June|July|August|September|October|November|December|"
    r"Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec"
    r")\s+((?:19|20)\d{2})\b",
    re.IGNORECASE,
)
_RE_NUMERIC = re.compile(r"\b(\d{1,2})/(\d{1,2})/(\d{2,4})\b")


@dataclass(frozen=True)
class DateNormalization:
    raw: str
    canonical: str
    precision: str  # day | month | year
    ambiguous: bool = False
    confidence: float = 1.0
    relative: bool = False
    reference_date: str | None = None


def _norm_year(year: str) -> int:
    y = int(year)
    if len(year) == 2:
        # Simple pivot rule; keep deterministic and explicit.
        return 2000 + y if y <= 69 else 1900 + y
    return y


def _fmt_day(year: int, month: int, day: int) -> str:
    return f"{year:04d}-{month:02d}-{day:02d}"


def _fmt_month(year: int, month: int) -> str:
    return f"{year:04d}-{month:02d}"


def _coerce_reference_date(reference: datetime | date | str | None) -> date:
    """Resolve a reference date for relative expressions."""
    if isinstance(reference, datetime):
        return reference.date()
    if isinstance(reference, date):
        return reference
    if isinstance(reference, str):
        raw = reference.strip()
        if raw:
            try:
                # Accept plain date or ISO timestamp.
                if len(raw) <= 10:
                    return datetime.fromisoformat(raw).date()
                return datetime.fromisoformat(raw.replace("Z", "+00:00")).date()
            except Exception:
                pass
    return datetime.now(timezone.utc).date()


def normalize_date(
    text: str,
    *,
    default_day_first: bool = False,
    reference_datetime: datetime | date | str | None = None,
) -> DateNormalization | None:
    """Normalize a date-like string to canonical ISO-style representation.

    Returns ``None`` when *text* is not recognized as date-like.
    """
    if not text:
        return None
    value = re.sub(r"\s+", " ", text.strip())
    lower = value.lower()
    ref_day = _coerce_reference_date(reference_datetime)

    relative_day_offsets = (
        (r"\bday before yesterday\b", -2),
        (r"\byesterday\b", -1),
        (r"\btoday\b", 0),
        (r"\btomorrow\b", 1),
        (r"\bday after tomorrow\b", 2),
    )
    for pattern, offset in relative_day_offsets:
        if re.search(pattern, lower):
            resolved = ref_day + timedelta(days=offset)
            return DateNormalization(
                raw=text,
                canonical=_fmt_day(resolved.year, resolved.month, resolved.day),
                precision="day",
                ambiguous=False,
                confidence=0.95,
                relative=True,
                reference_date=ref_day.isoformat(),
            )

    if re.search(r"\bthis month\b", lower):
        return DateNormalization(
            raw=text,
            canonical=_fmt_month(ref_day.year, ref_day.month),
            precision="month",
            confidence=0.9,
            relative=True,
            reference_date=ref_day.isoformat(),
        )
    if re.search(r"\blast month\b", lower):
        y, m = ref_day.year, ref_day.month - 1
        if m == 0:
            y, m = y - 1, 12
        return DateNormalization(
            raw=text,
            canonical=_fmt_month(y, m),
            precision="month",
            confidence=0.9,
            relative=True,
            reference_date=ref_day.isoformat(),
        )
    if re.search(r"\bnext month\b", lower):
        y, m = ref_day.year, ref_day.month + 1
        if m == 13:
            y, m = y + 1, 1
        return DateNormalization(
            raw=text,
            canonical=_fmt_month(y, m),
            precision="month",
            confidence=0.9,
            relative=True,
            reference_date=ref_day.isoformat(),
        )

    if re.search(r"\bthis year\b", lower):
        return DateNormalization(
            raw=text,
            canonical=f"{ref_day.year:04d}",
            precision="year",
            confidence=0.85,
            relative=True,
            reference_date=ref_day.isoformat(),
        )
    if re.search(r"\blast year\b", lower):
        return DateNormalization(
            raw=text,
            canonical=f"{(ref_day.year - 1):04d}",
            precision="year",
            confidence=0.85,
            relative=True,
            reference_date=ref_day.isoformat(),
        )
    if re.search(r"\bnext year\b", lower):
        return DateNormalization(
            raw=text,
            canonical=f"{(ref_day.year + 1):04d}",
            precision="year",
            confidence=0.85,
            relative=True,
            reference_date=ref_day.isoformat(),
        )

    m = _RE_ISO_DAY.search(value)
    if m:
        y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if 1 <= mo <= 12 and 1 <= d <= 31:
            return DateNormalization(raw=text, canonical=_fmt_day(y, mo, d), precision="day")

    m = _RE_MONTH_NAME_DAY.search(value)
    if m:
        month = _MONTHS[m.group(1).lower()]
        day = int(m.group(2))
        year = int(m.group(3))
        if 1 <= day <= 31:
            return DateNormalization(raw=text, canonical=_fmt_day(year, month, day), precision="day")

    m = _RE_NUMERIC.search(value)
    if m:
        a, b, ytxt = int(m.group(1)), int(m.group(2)), m.group(3)
        year = _norm_year(ytxt)
        # Disambiguate MM/DD vs DD/MM with deterministic fallback.
        if a <= 12 and b <= 12 and a != b:
            if default_day_first:
                month, day = b, a
            else:
                month, day = a, b
            return DateNormalization(
                raw=text,
                canonical=_fmt_day(year, month, day),
                precision="day",
                ambiguous=True,
                confidence=0.7,
            )
        if a > 12 and b <= 12:
            month, day = b, a
        else:
            month, day = a, b
        if 1 <= month <= 12 and 1 <= day <= 31:
            return DateNormalization(raw=text, canonical=_fmt_day(year, month, day), precision="day")

    m = _RE_MONTH_NAME_YEAR.search(value)
    if m:
        month = _MONTHS[m.group(1).lower()]
        year = int(m.group(2))
        return DateNormalization(raw=text, canonical=_fmt_month(year, month), precision="month", confidence=0.9)

    m = _RE_ISO_MONTH.search(value)
    if m:
        year, month = int(m.group(1)), int(m.group(2))
        if 1 <= month <= 12:
            return DateNormalization(raw=text, canonical=_fmt_month(year, month), precision="month", confidence=0.9)

    # Year-only is lowest precision and should only match if no richer form did.
    m = _RE_YEAR.search(value)
    if m:
        year = int(m.group(1))
        return DateNormalization(raw=text, canonical=f"{year:04d}", precision="year", confidence=0.8)

    return None
