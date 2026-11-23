# Lumos

Edit photos by talking to them, powered by a vision-language model.

You type a command like "make it a bit brighter and warmer". Qwen2.5-VL looks at the photo and
returns edit settings as JSON, and PIL applies them.

## Setup

Local:

```bash
uv sync
```

Kaggle (reuses Kaggle's preinstalled CUDA torch):

```python
!pip install -q uv
!uv pip install --system -e .
# then restart the session before importing
```

## Usage

```python
from io import BytesIO
import requests
from PIL import Image
from lumos.vlm import load_model, get_edits
from lumos.editor import apply_edits

url = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"
model, processor = load_model()
edits = get_edits(model, processor, url, "make it a bit brighter and warmer")
edited = apply_edits(Image.open(BytesIO(requests.get(url).content)), edits)
```

## Notes

- Load in `bfloat16`. `float16` overflows to NaN on T4 GPUs.
- Pinned to `transformers==4.57.1`. Newer releases mixed with Kaggle's preinstalled packages broke imports.
