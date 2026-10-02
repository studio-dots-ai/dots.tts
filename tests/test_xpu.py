"""Device selection tests and small operator checks on an Intel GPU.

Run with ``python -m unittest discover -s tests -v``.
"""

import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from dots_tts.edit_runtime import DotsTtsEditRuntime
from dots_tts.models.dots_tts.edit_model import DotsTtsEditModel
from dots_tts.models.dots_tts.model import DotsTtsModel
from dots_tts.modules.backbone.dit_inference import (
    CachedDiTRunner,
    _resolve_kv_attention_backend,
)
from dots_tts.modules.backbone.inference_utils import compile_module_forward
from dots_tts.modules.backbone.layers import (
    MultiHeadAttention,
    RotaryEmbedding,
    _compiled_block_mask_flex_attention,
)
from dots_tts.modules.vocoder.bigvgan import AudioVAE
from dots_tts.modules.vocoder.config import AudioVAEConfig
from dots_tts.modules.vocoder.vocoder_inference import VocoderInference
from dots_tts.runtime import DotsTtsRuntime
from dots_tts.runtime_double_streaming import DotsTtsRuntimeDoubleStreaming
from dots_tts.training.checkpoint import _restore_rng_state, _rng_state
from dots_tts.utils.device import resolve_device, xpu_available
from dots_tts.utils.profiling import InferenceProfiler
from dots_tts.utils.util import seed_everything


class TestDeviceSelection(unittest.TestCase):
    def test_auto_prefers_cuda(self):
        with (
            patch("torch.cuda.is_available", return_value=True),
            patch("dots_tts.utils.device.xpu_available", return_value=True),
        ):
            self.assertEqual(resolve_device().type, "cuda")

    def test_auto_uses_xpu_without_cuda(self):
        with (
            patch("torch.cuda.is_available", return_value=False),
            patch("dots_tts.utils.device.xpu_available", return_value=True),
        ):
            self.assertEqual(resolve_device("auto").type, "xpu")

    def test_cpu_fallback_and_explicit_override(self):
        with (
            patch("torch.cuda.is_available", return_value=False),
            patch("dots_tts.utils.device.xpu_available", return_value=False),
        ):
            self.assertEqual(resolve_device().type, "cpu")
            with self.assertRaisesRegex(RuntimeError, "Intel XPU is not available"):
                resolve_device("xpu")
        self.assertEqual(resolve_device("cpu").type, "cpu")
        with self.assertRaisesRegex(ValueError, "device must be"):
            resolve_device("meta")

    def test_cpu_half_precision_is_rejected(self):
        for precision in ("bf16", "fp16", "torch.bfloat16", "torch.float16"):
            with self.subTest(precision=precision):
                with self.assertRaisesRegex(RuntimeError, "Half-precision"):
                    DotsTtsRuntime._check_torch_env(precision, "cpu")

    def test_profiler_synchronizes_xpu(self):
        with patch("torch.xpu.synchronize") as synchronize:
            InferenceProfiler(torch.device("xpu:0"))._sync()
            synchronize.assert_called_once_with(torch.device("xpu:0"))

    def test_flex_backend_selection(self):
        with patch.dict("os.environ", {}, clear=True):
            for device_type in ("cuda", "xpu"):
                self.assertEqual(
                    _resolve_kv_attention_backend(
                        optimize=True, device_type=device_type
                    ),
                    "flex",
                )
            self.assertEqual(
                _resolve_kv_attention_backend(optimize=True, device_type="cpu"),
                "sdpa",
            )
            self.assertEqual(
                _resolve_kv_attention_backend(
                    optimize=True, device_type="xpu", default_backend="sdpa"
                ),
                "sdpa",
            )
        with patch.dict("os.environ", {"DOTS_TTS_DELAYED_DIT_BACKEND": "flex"}):
            self.assertEqual(
                _resolve_kv_attention_backend(
                    optimize=True, device_type="xpu", default_backend="sdpa"
                ),
                "flex",
            )
            with self.assertRaisesRegex(ValueError, "requires CUDA or XPU"):
                _resolve_kv_attention_backend(optimize=True, device_type="cpu")


class _SmallRuntimeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.core = nn.Linear(4, 4)
        self.vocoder = nn.Linear(4, 4)
        self.config = SimpleNamespace(vocoder=SimpleNamespace(sample_rate=24000))

    def set_optimize(self, optimize, *, max_sequence_length):
        self.optimize = optimize


class _SmallAudioVAE(AudioVAE):
    """Use real adapter operations without allocating the pretrained vocoder."""

    def __init__(self):
        nn.Module.__init__(self)
        self.h = SimpleNamespace(latent_dim=4)
        self.audio_encoder = nn.Conv1d(1, 4, 1)
        self.enc_mi_layer = nn.Linear(4, 4)
        self.pre_proj = nn.Conv1d(4, 8, 1)
        self.post_proj = nn.Conv1d(4, 4, 1)
        self.dec_mi_layer = nn.Linear(4, 4)
        self.decoder = nn.Conv1d(4, 1, 1)


@unittest.skipUnless(xpu_available(), "Requires an Intel XPU")
class TestXpuOperators(unittest.TestCase):
    def test_runtime_places_model_and_preserves_audio_precision(self):
        runtime = DotsTtsRuntime(
            _SmallRuntimeModel(),
            Path("."),
            device="xpu:0",
            precision="bfloat16",
        )
        self.assertEqual(runtime.device, torch.device("xpu:0"))
        self.assertEqual(runtime.model.core.weight.dtype, torch.bfloat16)
        self.assertEqual(runtime.model.vocoder.weight.dtype, torch.float32)
        self.assertEqual(runtime.model.core.weight.device.type, "xpu")
        self.assertEqual(runtime.model.vocoder.weight.device.type, "xpu")
        with torch.autocast("xpu", dtype=torch.bfloat16):
            output = runtime.model.core(torch.ones(1, 4, device="xpu"))
        self.assertTrue(torch.isfinite(output).all())

    def test_optimized_runtime_warms_up_xpu(self):
        with patch.object(DotsTtsRuntime, "run_warmup") as warmup:
            DotsTtsRuntime(_SmallRuntimeModel(), Path("."), device="xpu", optimize=True)
        warmup.assert_called_once()

    def test_pretrained_apis_forward_device(self):
        for runtime_type, model_type in (
            (DotsTtsRuntime, DotsTtsModel),
            (DotsTtsEditRuntime, DotsTtsEditModel),
            (DotsTtsRuntimeDoubleStreaming, DotsTtsModel),
        ):
            with (
                self.subTest(runtime=runtime_type.__name__),
                patch.object(
                    runtime_type, "_resolve_pretrained_path", return_value=Path(".")
                ),
                patch.object(
                    model_type, "from_pretrained", return_value=_SmallRuntimeModel()
                ),
            ):
                runtime = runtime_type.from_pretrained("test-model", device="xpu:0")
                self.assertEqual(runtime.device, torch.device("xpu:0"))
                self.assertEqual(runtime.model.core.weight.device.type, "xpu")

    def test_rotary_and_sdpa_match_cpu(self):
        seed_everything(42)
        attention = MultiHeadAttention(
            hidden_size=32, num_heads=4, rotary_bias=True
        ).eval()
        inputs = torch.randn(1, 8, 32)
        mask = torch.ones(1, 8, 8, dtype=torch.bool).tril()
        with torch.no_grad():
            expected = attention(inputs, mask=mask)
            attention = attention.to("xpu", dtype=torch.bfloat16)
            with torch.autocast("xpu", dtype=torch.bfloat16):
                actual = attention(inputs.to("xpu"), mask=mask.to("xpu"))
                positions = RotaryEmbedding(8).to("xpu")(torch.arange(8, device="xpu"))
                compiled = compile_module_forward(attention)
                compiled_output = compiled(inputs.to("xpu"), mask=mask.to("xpu"))
        self.assertEqual(positions.dtype, torch.float32)
        torch.testing.assert_close(actual.float().cpu(), expected, atol=0.01, rtol=0.05)
        torch.testing.assert_close(compiled_output, actual, atol=0.01, rtol=0.05)

    def test_cached_dit_flex_matches_sdpa(self):
        runner = CachedDiTRunner.__new__(CachedDiTRunner)
        runner.capacity_tokens = 128
        runner.unit_len = 4
        runner.block_mask_size = 64
        runner.device = torch.device("xpu")
        q = torch.randn(1, 2, 8, 64, device="xpu", dtype=torch.bfloat16)
        k = torch.randn(1, 2, 136, 64, device="xpu", dtype=torch.bfloat16)
        v = torch.randn_like(k)
        with torch.no_grad():
            # Exercise partially filled prefixes and the previous/current patch
            # masks used during incremental KV-cache decoding.
            for prefix_len in (0, 63, 128):
                runner.attn_backend = "flex"
                runner._mask_cache = {}
                flex_mask, _ = runner.masks_for(valid_persistent_tokens=prefix_len)
                actual = _compiled_block_mask_flex_attention(q, k, v, flex_mask)
                runner.attn_backend = "sdpa"
                runner._mask_cache = {}
                _, sdpa_mask = runner.masks_for(valid_persistent_tokens=prefix_len)
                expected = torch.nn.functional.scaled_dot_product_attention(
                    q, k, v, attn_mask=sdpa_mask
                )
                torch.testing.assert_close(actual, expected, atol=0.02, rtol=0.02)

    def test_vocoder_disables_xpu_autocast(self):
        vocoder = _SmallAudioVAE().to("xpu").eval()
        adapter = VocoderInference(vocoder)
        audio = torch.randn(1, 1, 16, device="xpu")
        with torch.autocast("xpu", dtype=torch.bfloat16):
            latents = adapter.extract_latents(audio)
            decoded = adapter.decode_latents(latents[:, :4].transpose(1, 2))
            direct_latents = vocoder.extract_latents(audio)
        for output in (latents, decoded, direct_latents):
            self.assertEqual(output.dtype, torch.float32)
            self.assertTrue(torch.isfinite(output).all())

    def test_vocoder_can_disable_compilation(self):
        adapter = VocoderInference(_SmallAudioVAE().to("xpu").eval())
        expected = torch.ones(1, 16, device="xpu")
        with (
            patch.object(adapter, "_stream_step_eager", return_value=expected) as eager,
            patch.object(adapter, "_get_compiled_stream_step") as compiled,
        ):
            actual = adapter.stream_step(
                torch.randn(1, 2, 4, device="xpu"),
                SimpleNamespace(),
                optimize=True,
                use_compiled=False,
            )
        self.assertIs(actual, expected)
        eager.assert_called_once()
        compiled.assert_not_called()

    def test_compiled_vocoder_stream_matches_eager(self):
        config = AudioVAEConfig(
            latent_dim=4,
            causal=True,
            mi_num_layers=1,
            downsample_rates=[2, 2],
            downsample_channels=[4, 8, 16],
            upsample_rates=[2, 2],
            upsample_kernel_sizes=[4, 4],
            upsample_initial_channel=16,
            resblock_kernel_sizes=[3],
            resblock_dilation_sizes=[[1, 1, 1]],
        )
        vocoder = AudioVAE(config).eval()
        vocoder.remove_weight_norm()
        adapter = VocoderInference(vocoder.to("xpu"))
        eager_state = adapter.init_stream_state(chunk_size=4)
        compiled_state = adapter.init_stream_state(chunk_size=4)
        for _ in range(3):
            latent_patch = torch.randn(1, 4, 4, device="xpu")
            expected = adapter.stream_step(
                latent_patch, eager_state, optimize=True, use_compiled=False
            )
            actual = adapter.stream_step(latent_patch, compiled_state, optimize=True)
            torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-4)
        torch.testing.assert_close(
            adapter.flush(compiled_state),
            adapter.flush(eager_state),
            atol=1e-5,
            rtol=1e-4,
        )
        self.assertTrue(adapter._compiled_stream_steps)

    def test_seed_and_checkpoint_restore_xpu_rng(self):
        seed_everything(123)
        state = _rng_state()
        expected = torch.rand(16, device="xpu")
        torch.rand(16, device="xpu")
        _restore_rng_state(state)
        self.assertTrue(torch.equal(torch.rand(16, device="xpu"), expected))
        seed_everything(123)
        self.assertTrue(torch.equal(torch.rand(16, device="xpu"), expected))
        # Older checkpoints have no XPU RNG state.
        state.pop("xpu")
        _restore_rng_state(state)


if __name__ == "__main__":
    unittest.main()
