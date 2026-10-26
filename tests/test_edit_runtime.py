from types import SimpleNamespace

import pytest
import torch

from dots_tts.edit_runtime import DotsTtsEditRuntime


@pytest.mark.parametrize("profile", [False, True])
def test_streaming_tts_without_speaker_guidance(profile):
    runtime = DotsTtsEditRuntime.__new__(DotsTtsEditRuntime)
    runtime.device = torch.device("cpu")
    runtime.precision = "float32"
    runtime.max_generate_length = 32
    runtime.vocoder_merge_steps = 4
    runtime.sample_rate = 48000
    runtime.resolve_sampling_options = lambda **kwargs: ("euler", 10, 1.2)
    inputs = dict(
        fid="test", text="Hello", prompt_text="Hi", template_name="tts", language="en"
    )
    runtime._prepare_audio_fill_tts_inputs = lambda **kwargs: inputs
    chunks = [torch.ones(1, 480), torch.ones(1, 240)]

    def generate_audio_stream(data, **kwargs):
        assert data is inputs
        yield from chunks

    runtime.model = SimpleNamespace(generate_audio_stream=generate_audio_stream)
    actual = list(
        runtime.generate_stream(
            text="Hello",
            prompt_audio_path="reference.wav",
            prompt_text="Hi",
            use_xvector=False,
            profile_inference=profile,
        )
    )
    assert len(actual) == len(chunks)
    for output, expected in zip(actual, chunks):
        torch.testing.assert_close(output, expected)
