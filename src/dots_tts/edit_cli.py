from __future__ import annotations

import argparse
from pathlib import Path


def _parse_use_xvector(value: str) -> bool | str:
    if value == "auto":
        return "auto"
    if value == "on":
        return True
    if value == "off":
        return False
    raise argparse.ArgumentTypeError(
        "--use-xvector must be one of: auto, on, off"
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="dots.tts.edit inference CLI")
    parser.add_argument("--model-name-or-path", required=True)
    parser.add_argument("--source-audio", required=True)
    parser.add_argument("--instruction", required=True)
    parser.add_argument("--source-text", default=None)
    parser.add_argument("--target-text", default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--revision", default=None)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--precision", default="bfloat16")
    parser.add_argument(
        "--device",
        default="auto",
        help="Inference device: auto, cpu, cuda[:index], or xpu[:index]",
    )
    parser.add_argument("--optimize", action="store_true")
    parser.add_argument("--max-generate-length", type=int, default=500)
    parser.add_argument("--max-sequence-length", type=int, default=2048)
    parser.add_argument("--speaker-scale", type=float, default=1.5)
    parser.add_argument(
        "--use-xvector",
        nargs="?",
        const=True,
        default="auto",
        type=_parse_use_xvector,
        metavar="{auto,on,off}",
        help=(
            "Source-speaker guidance mode (default: auto). Passing the flag "
            "without a value forces it on."
        ),
    )
    parser.add_argument(
        "--ode-method",
        default=None,
        help="ODE solver method (default: artifact setting, otherwise euler).",
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=None,
        help="Sampling steps (default: artifact setting, otherwise 10).",
    )
    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=None,
        help="Guidance scale (default: artifact setting, otherwise 1.2).",
    )
    parser.add_argument("--seed", type=int, default=20260414)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    import soundfile as sf

    from dots_tts.edit_runtime import DotsTtsEditRuntime
    from dots_tts.utils.util import seed_everything

    seed_everything(args.seed)
    runtime = DotsTtsEditRuntime.from_pretrained(
        args.model_name_or_path,
        revision=args.revision,
        cache_dir=args.cache_dir,
        precision=args.precision,
        device=args.device,
        optimize=args.optimize,
        max_generate_length=args.max_generate_length,
        max_sequence_length=args.max_sequence_length,
    )
    result = runtime.generate_edit(
        source_audio_path=args.source_audio,
        instruction=args.instruction,
        source_text=args.source_text,
        target_text=args.target_text,
        use_xvector=args.use_xvector,
        speaker_scale=args.speaker_scale,
        ode_method=args.ode_method,
        num_steps=args.num_steps,
        guidance_scale=args.guidance_scale,
    )
    output_path = Path(args.output).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(
        output_path,
        result["audio"].detach().cpu().squeeze().float().numpy(),
        result["sample_rate"],
    )


if __name__ == "__main__":
    main()
