"""Gradio UI for Lumos."""

try:
    import spaces
    gpu = spaces.GPU
except ImportError:
    gpu = lambda f: f   # no-op on Kaggle / locally

import gradio as gr

from lumos.editor import apply_edits
from lumos.vlm import get_edits


def build_app(model, processor):
    @gpu
    def edit(image, command, chain, state):
        if image is None or not command.strip():
            raise gr.Error("Upload a photo and type a command.")
        source = state if (chain and state is not None) else image
        edits = get_edits(model, processor, source, command)
        if not isinstance(edits, dict):
            raise gr.Error(f"Model didn't return JSON: {edits}")
        result = apply_edits(source, edits)
        return result, edits, result

    with gr.Blocks(title="Lumos") as demo:
        gr.Markdown("# Lumos\nEdit photos by talking to them.")
        state = gr.State(None)
        with gr.Row():
            inp = gr.Image(type="pil", label="Original")
            out = gr.Image(type="pil", label="Edited")
        command = gr.Textbox(label="What should I change?",
                             placeholder="make it warmer and a bit brighter")
        chain = gr.Checkbox(label="Apply to the edited photo (stack edits)")
        btn = gr.Button("Edit", variant="primary")
        settings = gr.JSON(label="Edit settings")

        btn.click(edit, [inp, command, chain, state], [out, settings, state])
        command.submit(edit, [inp, command, chain, state], [out, settings, state])
        gr.Examples(["make it warmer and a bit brighter", "moody",
                     "black and white", "vintage film"], inputs=command)
    return demo
