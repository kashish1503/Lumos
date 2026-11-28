# Lumos

Edit photos by talking to them, powered by a vision-language model.

Type (or say) a command like "make it a bit brighter and warmer". Qwen2.5-VL looks at the photo
and returns edit settings as JSON, and PIL applies them.

## Quick start

```bash
uv sync
uv run lumos ui
```

This opens the web UI. Upload a photo, then type or say what to change.

To edit a single photo from the terminal:

```bash
uv run lumos edit photo.jpg "make it warmer and a bit brighter"
# prints the settings and saves photo_lumos.jpg
```

## Commands

| Command | Description |
|---|---|
| `lumos ui` | Launch the Gradio web UI (text + microphone input) |
| `lumos ui --share` | Same, with a public Gradio link (useful on Kaggle/Colab) |
| `lumos ui --no-voice` | Skip loading Whisper; text commands only |
| `lumos edit IMAGE "COMMAND" [-o OUT]` | Edit one photo (a path or a URL) and save the result |
| `lumos eval [--image URL]` | Run the eval suite in `evals/cases.json` |

Prefix each command with `uv run` after `uv sync`. `python -m lumos ...` works too.

In the UI, tick **Apply to the edited photo** to stack edits. Uploading a new photo resets the stack.

## Setup

Requires Python 3.11+ and a CUDA GPU (developed on Kaggle T4s).

Local:

```bash
uv sync
```

Kaggle (reuses Kaggle's preinstalled CUDA torch):

```bash
!pip install -q uv && uv pip install --system -e .
# restart the session, then:
!lumos ui --share
```

## How it works

```
command (text or voice) ──► Whisper (voice only) ──► Qwen2.5-VL-3B + photo ──► JSON settings ──► PIL ──► edited photo
```

| Setting      | Neutral | Applied range | Notes |
|--------------|---------|---------------|-------|
| `brightness` | 1.0     | 0.2 – 2.0     | |
| `contrast`   | 1.0     | 0.2 – 2.0     | "softer" lowers it |
| `saturation` | 1.0     | 0.0 – 2.0     | 0 = black and white |
| `sharpness`  | 1.0     | 0.0 – 3.0     | "crisper" / "sharpen" raises it |
| `warmth`     | 0       | -50 – 50      | positive = warmer, negative = cooler |

- **Intensity words:** "slightly" / "a bit" changes a setting by 0.1 (warmth 10), a plain request by 0.2 (warmth 20), and "a lot" / "very" by 0.4 (warmth 35).
- **Explicit values win:** a value stated in the command, such as "set warmth to 20" or "brightness 1.4", replaces whatever the model returned for that setting.

## Evaluation

`evals/cases.json` lists commands with the settings each must produce:

```json
{"command": "vintage film", "expect": {"saturation": "<1", "warmth": ">0"}}
```

- Rules use `>`, `<` or `==`.
- Settings not named in `expect` must stay neutral, unless they're listed in `allow`.

`uv run lumos eval` prints PASS or FAIL for each case and saves a report to `evals/runs/run-<timestamp>.json`.

## Project layout

```
src/lumos/
  cli.py        # `lumos` command
  vlm.py        # load Qwen2.5-VL, prompt it, parse the JSON, apply explicit values
  editor.py     # apply_edits(): PIL enhancements + warmth
  speech.py     # Whisper large-v3-turbo speech-to-text
  app.py        # Gradio UI
  evaluate.py   # eval runner
evals/
  cases.json    # eval cases
  runs/         # saved eval reports
```

## Notes

- Load in `bfloat16`. `float16` overflows to NaN on T4 GPUs.
- On a machine with 2 GPUs, `load_model` caps each at 12 GiB.
- Pinned to `transformers==4.57.1`. Newer releases mixed with Kaggle's preinstalled packages broke imports.
- `src/lumos.ipynb` is the original Kaggle prototyping notebook.
