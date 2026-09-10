"""OmniVoice waveform conversion.

`OmniVoice.generate()` returns `list[np.ndarray]` with one-dimensional float
waveforms. The original conversion called `squeeze(0)` — which raises when
axis 0 is not size one — and then `.numpy()` on an ndarray, so every TTS entry
point was dead.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from tts import _to_pcm16

SAMPLE = np.array([0.0, 0.5, -0.5, 1.0, -1.0], dtype=np.float32)
EXPECTED = np.array([0, 16383, -16383, 32767, -32767], dtype=np.int16)


def test_documented_return_shape():
    """The shape OmniVoice actually returns: a list of 1-D float arrays."""
    assert np.array_equal(_to_pcm16([SAMPLE]), EXPECTED)


def test_int16_conversion_does_not_overflow():
    """Out-of-range samples clip instead of wrapping to the opposite sign."""
    loud = np.array([2.0, -3.0], dtype=np.float32)
    assert np.array_equal(_to_pcm16([loud]), np.array([32767, -32767], dtype=np.int16))


@pytest.mark.parametrize(
    "audio",
    [
        pytest.param(SAMPLE, id="bare-array"),
        pytest.param([SAMPLE.reshape(1, -1)], id="leading-axis"),
        pytest.param([torch.tensor(SAMPLE)], id="torch-tensor"),
    ],
)
def test_tolerated_variants(audio):
    assert np.array_equal(_to_pcm16(audio), EXPECTED)


def test_empty_output_is_not_an_exception():
    assert _to_pcm16([]) is None


def test_result_is_int16():
    assert _to_pcm16([SAMPLE]).dtype == np.int16


def test_conversion_does_not_mutate_the_model_output():
    """`reshape` can hand back a view of OmniVoice's own buffer."""
    loud = np.array([2.0, -2.0, 0.25], dtype=np.float32)
    original = loud.copy()
    _to_pcm16([loud])
    np.testing.assert_array_equal(loud, original)
