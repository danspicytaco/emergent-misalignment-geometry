"""Explicit composite identity and fail-closed joins, including legacy adapters."""
from dataclasses import dataclass


@dataclass(frozen=True)
class ResponseIdentity:
    run: str
    checkpoint: str
    training_seed: int
    prompt: str
    response_sample: int

    def __post_init__(self):
        if any(not isinstance(v, str) or not v for v in
               (self.run, self.checkpoint, self.prompt)):
            raise ValueError("run, checkpoint and prompt identities must be nonempty strings")
        for value in (self.training_seed, self.response_sample):
            if type(value) is not int or value < 0:
                raise ValueError("seed and sample must be nonnegative integers")


@dataclass(frozen=True)
class JudgmentIdentity:
    response: ResponseIdentity
    judge: str
    pass_id: str

    def __post_init__(self):
        if not isinstance(self.response, ResponseIdentity):
            raise ValueError("judgment requires a ResponseIdentity")
        if any(not isinstance(v, str) or not v for v in (self.judge, self.pass_id)):
            raise ValueError("judge and pass_id must be nonempty strings")


def unique_index(records, key):
    result = {}
    for record in records:
        identity = key(record)
        if identity in result:
            raise ValueError(f"duplicate record identity: {identity!r}")
        result[identity] = record
    return result


def strict_join(left, right, key):
    """One-to-one exact-key join, preserving left order; never silently drops rows."""
    li, ri = unique_index(left, key), unique_index(right, key)
    if li.keys() != ri.keys():
        raise ValueError(f"join identity mismatch: left-only={len(li.keys() - ri.keys())}, "
                         f"right-only={len(ri.keys() - li.keys())}")
    return [(record, ri[identity]) for identity, record in li.items()]


def align_legacy_pairs(reference, candidate, *, expected_count):
    """Validate a legacy single-response pass, then return it in idx order.

    Legacy idx is only meaningful inside this explicitly supplied comparison;
    it is NOT a globally unique response or judgment identity.
    """
    def key(record):
        idx = record.get("idx")
        if type(idx) is not int:
            raise ValueError("legacy idx must be an integer")
        return idx

    pairs = strict_join(reference, candidate, key)
    if {key(a) for a, _ in pairs} != set(range(expected_count)):
        raise ValueError(f"legacy indices must be exactly 0..{expected_count - 1}")
    for a, b in pairs:
        for field in ("prompt", "response"):
            if not isinstance(a.get(field), str) or a[field] != b.get(field):
                raise ValueError(f"legacy {field} mismatch at idx={a['idx']}")
    return sorted((b for _, b in pairs), key=key)
