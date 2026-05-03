"""Tests for `tetris_bot.run_naming`."""

from __future__ import annotations

import random
from datetime import datetime, timezone

from tetris_bot.run_naming import (
    ADJECTIVES,
    ANIMALS,
    generate_run_id,
    is_friendly_run_id,
    short_utc_timestamp,
)


def test_short_utc_timestamp_format() -> None:
    fixed = datetime(2026, 5, 3, 9, 52, tzinfo=timezone.utc)
    assert short_utc_timestamp(fixed) == "20260503-0952"


def test_generate_run_id_shape_is_recognizable() -> None:
    fixed = datetime(2026, 5, 3, 9, 52, tzinfo=timezone.utc)
    rid = generate_run_id(rng=random.Random(0), now=fixed)
    parts = rid.split("-")
    assert len(parts) == 4
    assert parts[0] in ADJECTIVES
    assert parts[1] in ANIMALS
    assert parts[2] == "20260503"
    assert parts[3] == "0952"


def test_is_friendly_run_id_round_trip() -> None:
    fixed = datetime(2026, 5, 3, 9, 52, tzinfo=timezone.utc)
    rid = generate_run_id(rng=random.Random(42), now=fixed)
    assert is_friendly_run_id(rid)
    # Must reject legacy v-numbers and arbitrary strings.
    assert not is_friendly_run_id("v3")
    assert not is_friendly_run_id("custom-name")
    assert not is_friendly_run_id("amber-otter-2026-0503")
    assert not is_friendly_run_id("notanadjective-otter-20260503-0952")
    assert not is_friendly_run_id("amber-notananimal-20260503-0952")


def test_word_lists_have_no_duplicates() -> None:
    # Duplicates would reduce the effective entropy and make the type-check
    # for `is_friendly_run_id` lie about its uniqueness.
    assert len(set(ADJECTIVES)) == len(ADJECTIVES)
    assert len(set(ANIMALS)) == len(ANIMALS)


def test_generate_run_id_uses_system_random_by_default() -> None:
    # Smoke test: two calls produce two valid ids without raising.
    a = generate_run_id()
    b = generate_run_id()
    assert is_friendly_run_id(a)
    assert is_friendly_run_id(b)
