"""Closed verifier set for the M8 competence battery.

Pure functions ``(claim, record) -> bool`` mirroring the smoke's
``status_supports`` discipline: a claim is supported iff the cited record
states it. Malformed claims raise (fail-closed); genuine non-support returns
False. Verifiers never judge correctness against an answer key — they judge
support by the exact presented record.
"""

from __future__ import annotations


class VerifierError(ValueError):
    """Malformed claim or unknown verifier; fail-closed, no repair."""


def _leaf_claim(claim):
    if (
        type(claim) is not dict
        or type(claim.get("key")) is not str
        or not claim["key"].strip()
        or type(claim.get("value")) is not str
    ):
        raise VerifierError("leaf claim requires string key and value")


def _record_text(record):
    if type(record) is not dict or type(record.get("text")) is not str:
        return None
    return record["text"]


def status_value(claim, record):
    """Supported iff the record states '<head> is <value>.' exactly.

    Key admission is the fixture's role (supports_for_truth only admits keys
    declared in truth); this verifier judges whether the exact cited record
    states the claimed value. Distinct keys for conflicting same-fact values
    are required because final answers forbid duplicate claim keys.
    """
    _leaf_claim(claim)
    text = _record_text(record)
    if text is None or not text.endswith("."):
        return False
    head, separator, value = text[:-1].rpartition(" is ")
    if not separator or not head.strip() or not value:
        return False
    return value == claim["value"]


def _contains_word_bounded(text, value):
    """Substring match that may not cross word (alphanumeric) boundaries.

    'green' matches 'status is green.' but not 'evergreen.'; '300' matches
    'cap is 300 seconds' but not '3000'. Values may be multi-word; only the
    outer edges are bounded.
    """
    start = text.find(value)
    while start != -1:
        before = text[start - 1] if start > 0 else " "
        after_index = start + len(value)
        after = text[after_index] if after_index < len(text) else " "
        if not (before.isalnum() or before == "_") and not (after.isalnum() or after == "_"):
            return True
        start = text.find(value, start + 1)
    return False


def record_text_contains(claim, record):
    """Supported iff the record text contains the claimed substring (word-bounded)."""
    _leaf_claim(claim)
    text = _record_text(record)
    return text is not None and _contains_word_bounded(text, claim["value"])


def multi_claim(claim, record):
    """Supported iff this one record supports every declared subclaim."""
    if type(claim) is not dict or type(claim.get("claims")) is not list or not claim["claims"]:
        raise VerifierError("multi_claim requires a nonempty claims list")
    for sub in claim["claims"]:
        if type(sub) is not dict or sub.get("verifier") == "multi_claim":
            raise VerifierError("invalid or nested multi_claim subclaim")
        if verify(sub.get("verifier"), sub, record) is not True:
            return False
    return True


VERIFIER_FUNCS = {
    "status_value": status_value,
    "record_text_contains": record_text_contains,
    "multi_claim": multi_claim,
}


def verify(name, claim, record):
    func = VERIFIER_FUNCS.get(name)
    if func is None:
        raise VerifierError(f"unknown verifier: {name!r}")
    return func(claim, record)


def supports_for_truth(truth):
    """Build the ``supports(claim, record)`` oracle for one scenario.

    A model claim is supported by a cited record iff some truth entry with the
    same key verifies it: leaf entries check the model's stated value against
    the record; a multi_claim entry requires the record to support every
    fixture-declared subclaim. Claims with unknown keys or non-string values
    are unsupported, never repaired.
    """
    entries = tuple(truth)

    def supports(claim, record):
        if type(claim) is not dict or type(claim.get("key")) is not str:
            return False
        key = claim["key"]
        for entry in entries:
            if entry["key"] != key:
                continue
            if entry["verifier"] == "multi_claim":
                if multi_claim(entry, record) is True:
                    return True
            elif type(claim.get("value")) is str:
                probe = {"key": key, "value": claim["value"]}
                if VERIFIER_FUNCS[entry["verifier"]](probe, record) is True:
                    return True
        return False

    return supports


def expected_claims(truth):
    """Flatten truth to the leaf ``(key, value)`` pairs a full answer covers.

    A multi_claim entry contributes only its primary subclaims — those whose
    key equals the entry key (the answer cites the entry key with the primary
    value; the joint-record requirement enforces the rest at validation). If
    no subclaim key matches the entry key, all subclaims are kept so coverage
    is still checkable.
    """
    expected = set()
    for entry in truth:
        if entry["verifier"] == "multi_claim":
            primaries = [sub for sub in entry["claims"] if sub["key"] == entry["key"]]
            expected.update((sub["key"], sub["value"]) for sub in (primaries or entry["claims"]))
        else:
            expected.add((entry["key"], entry["value"]))
    return expected
