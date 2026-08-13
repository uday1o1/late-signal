"""Fake Negative Weighted method event stream."""

from latesignal.methods.immediate_fake_negative import ImmediateFakeNegativeMethod


class FakeNegativeWeightedMethod(ImmediateFakeNegativeMethod):
    """Use the immediate duplicate stream with exposure-time FNW weights."""

    name = "fnw"
