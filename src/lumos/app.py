"""Gradio UI for Lumos."""

import gradio as gr

from lumos.editor import apply_edits
from lumos.vlm import get_edits

try:
    import spaces
    gpu = spaces.GPU
except ImportError:
    gpu = lambda f: f   # no-op on Kaggle / locally


def build_app(model, processor, asr=None):
    @gpu
    def edit(image, command, chain, state):
        if image is None or not command.strip():
            raise gr.Error("Upload a photo and type or say a command.")
        source = state if (chain and state is not None) else image
        edits = get_edits(model, processor, source, command)
        if not isinstance(edits, dict):
            raise gr.Error(f"Model didn't return JSON: {edits}")
        result = apply_edits(source, edits)
        return result, edits, result

    @gpu
    def hear(audio_path):
        from lumos.speech import transcribe
        text = transcribe(asr, audio_path)
        if not text:
            raise gr.Error("Couldn't hear anything. Try again.")
        return text

    with gr.Blocks(title="Lumos") as demo:
        gr.Markdown("# Lumos\nEdit photos by talking to them.")
        state = gr.State(None)
        with gr.Row():
            inp = gr.Image(type="pil", label="Original")
            out = gr.Image(type="pil", label="Edited")
        command = gr.Textbox(label="What should I change?",
                             placeholder="make it warmer and a bit brighter")
        if asr is not None:
            mic = gr.Audio(sources=["microphone"], type="filepath",
                           label="Or say it")
        chain = gr.Checkbox(label="Apply to the edited photo (stack edits)")
        btn = gr.Button("Edit", variant="primary")
        settings = gr.JSON(label="Edit settings")

        io = [inp, command, chain, state]
        outs = [out, settings, state]
        btn.click(edit, io, outs)
        command.submit(edit, io, outs)
        inp.upload(lambda: None, None, state)   # new photo resets stacking

        if asr is not None:
            mic.stop_recording(hear, mic, command).then(edit, io, outs)

        gr.Examples(["make it warmer and a bit brighter", "dark and dramatic",
                     "black and white", "faded retro look"], inputs=command)
    return demo
