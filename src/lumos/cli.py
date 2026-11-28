"""Command-line entry point: `lumos ui | edit | eval`."""

import argparse
import json
import sys
from io import BytesIO
from pathlib import Path

DEMO_IMAGE = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"


def is_url(s: str) -> bool:
    return s.startswith(("http://", "https://"))


def load_image(src: str):
    from PIL import Image
    if is_url(src):
        import requests
        resp = requests.get(src, timeout=30)
        resp.raise_for_status()
        return Image.open(BytesIO(resp.content)).convert("RGB")
    return Image.open(src).convert("RGB")


def default_output(src: str) -> Path:
    if is_url(src):
        return Path("lumos_out.jpg")
    p = Path(src)
    return p.with_name(f"{p.stem}_lumos.jpg")


def cmd_ui(args):
    from lumos.app import build_app
    from lumos.vlm import load_model
    model, processor = load_model()
    asr = None
    if not args.no_voice:
        from lumos.speech import load_asr
        asr = load_asr()
    build_app(model, processor, asr).launch(share=args.share)


def cmd_edit(args):
    from lumos.editor import apply_edits
    from lumos.vlm import get_edits, load_model
    image = load_image(args.image)
    model, processor = load_model()
    edits = get_edits(model, processor, image, args.command)
    if not isinstance(edits, dict):
        print(f"Model didn't return JSON:\n{edits}", file=sys.stderr)
        sys.exit(1)
    print(json.dumps(edits, indent=2))
    out = Path(args.output) if args.output else default_output(args.image)
    apply_edits(image, edits).save(out)
    print(f"Saved {out}")


def cmd_eval(args):
    from lumos.evaluate import run_evals
    from lumos.vlm import load_model
    model, processor = load_model()
    run_evals(model, processor, args.image, cases_path=args.cases)


def main(argv=None):
    parser = argparse.ArgumentParser(prog="lumos", description="Edit photos by talking to them.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    ui = sub.add_parser("ui", help="launch the Gradio web UI")
    ui.add_argument("--share", action="store_true", help="create a public Gradio link")
    ui.add_argument("--no-voice", action="store_true", help="don't load Whisper (text only)")
    ui.set_defaults(func=cmd_ui)

    edit = sub.add_parser("edit", help="edit one photo from the terminal")
    edit.add_argument("image", help="path or http(s) URL of the photo")
    edit.add_argument("command", help='what to change, e.g. "make it warmer"')
    edit.add_argument("-o", "--output", help="where to save (default: <name>_lumos.jpg)")
    edit.set_defaults(func=cmd_edit)

    ev = sub.add_parser("eval", help="run the eval suite")
    ev.add_argument("--image", default=DEMO_IMAGE, help="photo to evaluate on (path or URL)")
    ev.add_argument("--cases", default="evals/cases.json", help="eval cases file")
    ev.set_defaults(func=cmd_eval)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
