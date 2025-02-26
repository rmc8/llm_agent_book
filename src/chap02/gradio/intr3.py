import gradio as gr

with gr.Blocks() as demo:
    audio = gr.Audio(label="Input Audio", type="filepath")
    checkbox = gr.Checkbox(label="Check Box")
    f = gr.File(label="File", file_types=["image"])
    number = gr.Number(label="Number")
    markdown = gr.Markdown(
        label="Markdown",
        value="# Title\n## Subtitle\nBody text here.",
    )
    slider = gr.Slider(label="Slider", minimum=-10, maximum=10)
    textbox = gr.Textbox(label="Text Box")

demo.launch(height=1200)