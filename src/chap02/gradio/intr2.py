import gradio as gr


def text_to_text(text: str):
    text = f"<<{text}>>"
    return text


def text_to_text_rich(text: str):
    top = "^" * len(text)
    bottom = "v" * len(text)
    return f"{top}\n<<{text}>>\n{bottom}"


with gr.Blocks() as demo:
    input_text = gr.Text(label="Input Text")
    btn1 = gr.Button(value="Normal")
    btn2 = gr.Button(value="Rich")
    output_text = gr.Textbox(label="Output Text", lines=5)
    btn1.click(inputs=input_text, outputs=output_text, fn=text_to_text)
    btn2.click(inputs=input_text, outputs=output_text, fn=text_to_text_rich)

demo.launch()
