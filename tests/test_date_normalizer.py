"""Tests for canonical date normalization."""

from __future__ import annotations

from datetime import date

from silicon_memory.entities.date_normalizer import normalize_date


def test_normalize_month_name_day_year() -> None:
    d = normalize_date("June 25, 2021")
    assert d is not None
    assert d.canonical == "2021-06-25"
    assert d.precision == "day"
    assert d.ambiguous is False


def test_normalize_numeric_mm_dd_yyyy_default() -> None:
    d = normalize_date("06/25/2021")
    assert d is not None
    assert d.canonical == "2021-06-25"
    assert d.precision == "day"
    assert d.ambiguous is False


def test_normalize_numeric_ambiguous_respects_default() -> None:
    us = normalize_date("03/04/2021")
    eu = normalize_date("03/04/2021", default_day_first=True)
    assert us is not None and eu is not None
    assert us.canonical == "2021-03-04"
    assert eu.canonical == "2021-04-03"
    assert us.ambiguous is True
    assert eu.ambiguous is True


def test_normalize_month_year() -> None:
    d = normalize_date("September 2020")
    assert d is not None
    assert d.canonical == "2020-09"
    assert d.precision == "month"


def test_normalize_year_only() -> None:
    d = normalize_date("In 2019 this happened.")
    assert d is not None
    assert d.canonical == "2019"
    assert d.precision == "year"


def test_normalize_relative_day_with_reference() -> None:
    d = normalize_date("tomorrow", reference_datetime=date(2026, 2, 19))
    assert d is not None
    assert d.canonical == "2026-02-20"
    assert d.precision == "day"
    assert d.relative is True
    assert d.reference_date == "2026-02-19"


def test_normalize_relative_month_with_reference() -> None:
    d = normalize_date("last month", reference_datetime=date(2026, 1, 10))
    assert d is not None
    assert d.canonical == "2025-12"
    assert d.precision == "month"
    assert d.relative is True


def test_normalize_relative_year_with_reference() -> None:
    d = normalize_date("next year", reference_datetime=date(2026, 2, 19))
    assert d is not None
    assert d.canonical == "2027"
    assert d.precision == "year"
    assert d.relative is True
