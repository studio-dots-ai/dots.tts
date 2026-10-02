from __future__ import annotations

import argparse
from pathlib import Path


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="dots.tts inference CLI.")
    template_choices = ("tts", "instruction_tts", "text_to_audio", "tts_interleave")
    parser.add_argument(
        "--model-name-or-path",
        required=True,
        help="Local pretrained directory or Hugging Face repo id",
    )
    parser.add_argument(
        "--revision", default=None, help="Optional Hugging Face revision"
    )
    parser.add_argument(
        "--cache-dir", default=None, help="Optional Hugging Face cache dir"
    )
    parser.add_argument("--text", type=str, required=True, help="Input text")
    parser.add_argument("--output", default="output.wav", help="Output wav file path")
    parser.add_argument(
        "--precision", type=str, default="bfloat16", help="Inference precision"
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Inference device: auto, cpu, cuda[:index], or xpu[:index]",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for inference.",
    )
    parser.add_argument(
        "--prompt-audio", type=str, default=None, help="Path to prompt audio"
    )
    parser.add_argument(
        "--prompt-text", type=str, default=None, help="Transcript of prompt audio"
    )
    parser.add_argument(
        "--language",
        type=str,
        default=None,
        help="Language tag mode. Default: none. Supported values: none, auto_detect, or a language code/name such as EN/en/english/chinese.",
    )
    parser.add_argument(
        "--template-name",
        choices=template_choices,
        default=None,
        help="Named template preset for generation.",
    )
    parser.add_argument(
        "--ode-method",
        type=str,
        default=None,
        help="ODE solver method (default: artifact setting, otherwise euler)",
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=None,
        help="Sampling steps (default: artifact setting, otherwise 10)",
    )
    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=None,
        help="Classifier-free guidance scale (default: artifact setting, otherwise 1.2)",
    )
    parser.add_argument(
        "--speaker-scale",
        type=float,
        default=1.5,
        help="Scale applied to the reference speaker embedding",
    )
    parser.add_argument(
        "--max-generate-length",
        type=int,
        default=500,
        help="Maximum total audio patch count (prompt + generated)",
    )
    parser.add_argument(
        "--max-sequence-length",
        type=int,
        default=2048,
        help="Maximum LLM sequence length used for schedule validation and optimized StaticCache",
    )
    parser.add_argument(
        "--normalize-text",
        action="store_true",
        help="Whether to normalize text before inference",
    )
    parser.add_argument(
        "--profile-inference",
        action="store_true",
        help="Collect per-module inference timing statistics",
    )
    parser.add_argument(
        "--profile-inference-calls",
        action="store_true",
        help="Log each profiled inference call timing",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    import soundfile as sf
    from loguru import logger

    from dots_tts.runtime import DotsTtsRuntime
    from dots_tts.utils.logging import categorized_log as logc
    from dots_tts.utils.logging import configure_logging
    from dots_tts.utils.util import seed_everything

    configure_logging()
    seed_everything(args.seed)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(
        logc("cli", "CLI command started: model={} output={} seed={}"),
        args.model_name_or_path,
        output_path,
        args.seed,
    )

    try:
        runtime = DotsTtsRuntime.from_pretrained(
            args.model_name_or_path,
            revision=args.revision,
            cache_dir=args.cache_dir,
            precision=args.precision,
            device=args.device,
            max_generate_length=args.max_generate_length,
            max_sequence_length=args.max_sequence_length,
        )
        result = runtime.generate(
            text=args.text,
            prompt_audio_path=args.prompt_audio,
            prompt_text=args.prompt_text,
            language=args.language,
            template_name=args.template_name,
            ode_method=args.ode_method,
            num_steps=args.num_steps,
            guidance_scale=args.guidance_scale,
            speaker_scale=args.speaker_scale,
            normalize_text=args.normalize_text,
            profile_inference=args.profile_inference or args.profile_inference_calls,
            log_profile_calls=args.profile_inference_calls,
        )
        sf.write(
            output_path,
            result["audio"].float().cpu().squeeze().numpy(),
            result["sample_rate"],
        )
    except Exception:
        logger.exception(
            logc("cli", "CLI inference failed: model={} output={}"),
            args.model_name_or_path,
            output_path,
        )
        raise

    logger.info(
        logc(
            "cli",
            "CLI output written: request_id={} output={} sample_rate={} samples={}",
        ),
        result["fid"],
        output_path,
        result["sample_rate"],
        int(result["audio"].shape[-1]),
    )


if __name__ == "__main__":
    raise SystemExit(main())
