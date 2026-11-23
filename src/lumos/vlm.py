"""Turn a natural-language edit command into edit parameters with Qwen2.5-VL."""

import json
import re

import torch
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"

PROMPT = """You are a photo editing assistant. Look at the image and the user's request.
Return ONLY a JSON object with these keys (use neutral values if a setting shouldn't change):
- brightness: float, 1.0 = no change (0.5 darker, 1.5 brighter)
- contrast: float, 1.0 = no change
- saturation: float, 1.0 = no change (0 = black and white)
- warmth: int, 0 = no change (-50 cooler to +50 warmer)
- sharpness: float, 1.0 = no change

User request: {command}"""


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
    return json.loads(match.group()) if match else raw
