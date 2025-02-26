import gradio as gr

with gr.Blocks() as demo:
    with gr.Accordion(label="Accordion 1"):
        gr.Text(value="Content of Accordion 1")
    with gr.Row():
        gr.Text(value="Left Column")
        gr.Text(value="Right Column")
    with gr.Row():
        with gr.Column():
            gr.Text(value="(0, 0)")
            gr.Text(value="(1, 0)")
        with gr.Column():
            gr.Text(value="(0, 1)")
            gr.Text(value="(1, 1)")
    with gr.Tab(label="Tab 1"):
        gr.Text(value="Content of Tab 1")
    with gr.Tab(label="Tab 2"):
        gr.Text(value="Content of Tab 2")

demo.launch(height=800)
