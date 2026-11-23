"""Turn a natural-language edit command into edit parameters with Qwen2.5-VL."""

import json
import re

import torch
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"

PROMPT = """You are a photo editing assistant. Convert the user's request into edit settings.
Return ONLY a JSON object with exactly these 5 keys:
brightness, contrast, saturation, sharpness (floats, 1.0 = no change) and warmth (int, 0 = no change).

Step sizes:
- "slightly" / "a bit" / "a little": change by 0.1 (warmth by 10)
- plain request ("brighter", "warmer"): change by 0.2 (warmth by 20)
- "a lot" / "very" / "much": change by 0.4 (warmth by 35)

Rules:
- Only change settings the request asks for or clearly implies. Every other key stays neutral.
- Never add warmth unless the request mentions color temperature or a mood that implies it.
- An explicit number is used exactly as given.
- Black and white means saturation 0 and warmth 0.
- "Softer" means lower contrast. "Crisper" or "sharpen" means sharpness above 1.

Examples:
"a little brighter" -> {{"brightness": 1.1, "contrast": 1.0, "saturation": 1.0, "warmth": 0, "sharpness": 1.0}}
"much more contrast" -> {{"brightness": 1.0, "contrast": 1.4, "saturation": 1.0, "warmth": 0, "sharpness": 1.0}}
"dark and dramatic" -> {{"brightness": 0.8, "contrast": 1.3, "saturation": 0.9, "warmth": 0, "sharpness": 1.0}}
"faded retro look" -> {{"brightness": 1.0, "contrast": 0.85, "saturation": 0.7, "warmth": 20, "sharpness": 1.0}}
"make the details crisper" -> {{"brightness": 1.0, "contrast": 1.0, "saturation": 1.0, "warmth": 0, "sharpness": 1.5}}
"a bit cooler" -> {{"brightness": 1.0, "contrast": 1.0, "saturation": 1.0, "warmth": -10, "sharpness": 1.0}}
"more saturated" -> {{"brightness": 1.0, "contrast": 1.0, "saturation": 1.2, "warmth": 0, "sharpness": 1.0}}
"vibrant and punchy" -> {{"brightness": 1.0, "contrast": 1.2, "saturation": 1.3, "warmth": 0, "sharpness": 1.0}}
"gloomy and brooding" -> {{"brightness": 0.85, "contrast": 1.2, "saturation": 0.85, "warmth": 0, "sharpness": 1.0}}

User request: {command}"""

SETTINGS = ("brightness", "contrast", "saturation", "warmth", "sharpness")
EXPLICIT = re.compile(
    r"\b(" + "|".join(SETTINGS) + r")\s*(?:to|=|:|at|of)?\s*(-?\d+(?:\.\d+)?)\b",
    re.IGNORECASE,
)


def apply_explicit_values(edits: dict, command: str) -> dict:
    """Override the model with any value the user stated outright, e.g. "set warmth to 2"."""
    for name, value in EXPLICIT.findall(command):
        name = name.lower()
        edits[name] = int(float(value)) if name == "warmth" else float(value)
    return edits


def load_model(max_memory=None, max_pixels=512 * 28 * 28):
    """Load model + processor. bfloat16 avoids the float16 NaN crash on T4 GPUs."""
    if max_memory is None and torch.cuda.device_count() == 2:
        max_memory = {0: "12GiB", 1: "12GiB"}
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        max_memory=max_memory,
    )
    processor = AutoProcessor.from_pretrained(MODEL_ID, max_pixels=max_pixels)
    return model, processor


def get_edits(model, processor, image, command: str):
    """Return a dict of edit parameters, or the raw model text if no JSON was found."""
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": PROMPT.format(command=command)},
        ],
    }]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text], images=image_inputs, videos=video_inputs,
        padding=True, return_tensors="pt",
    ).to(model.device)

    out = model.generate(**inputs, max_new_tokens=128, do_sample=False)
    raw = processor.batch_decode(
        [out[0][len(inputs.input_ids[0]):]], skip_special_tokens=True
    )[0]

    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        return raw
    edits = json.loads(match.group())
    return apply_explicit_values(edits, command) if isinstance(edits, dict) else edits
