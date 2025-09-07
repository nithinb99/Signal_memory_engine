# tests/test_ingestion_normalize_record.py

"""
Covers normalize_record() only (no external services).
We toggle SME_TEST_MODE so the module imports without side effects.
"""
import importlib
import pytest
import os

def _import_module_in_test_mode():
    os.environ["SME_TEST_MODE"] = "1"
    return importlib.import_module("signal_memory_engine_v1.ingestion.ingest_memory")

def test_normalize_record_variants():
    mod = _import_module_in_test_mode()

    # content under "content"
    rid, content, meta = mod.normalize_record({"id": "abc", "content": "Hello", "x": 1}, "seed")
    assert rid == "abc"
    assert content == "Hello"
    assert meta["source"] == "seed" and meta["x"] == 1

    # content under "text" -> synthesize id
    rid2, content2, meta2 = mod.normalize_record({"text": "Hi"}, "file2")
    assert content2 == "Hi" and meta2["source"] == "file2" and rid2

    # content under "excerpt"
    rid3, content3, meta3 = mod.normalize_record({"excerpt": "Piece"}, "file3")
    assert content3 == "Piece" and meta3["source"] == "file3"

    # content under "event_content"
    rid4, content4, meta4 = mod.normalize_record({"event_content": "Log line"}, "file4")
    assert content4 == "Log line" and meta4["source"] == "file4"

    # missing content -> ValueError
    with pytest.raises(ValueError):
        mod.normalize_record({"id": "nope"}, "bad")
