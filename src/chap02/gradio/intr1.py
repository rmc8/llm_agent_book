import gradio as gr


def text_to_text(text: str):
    text = f"<<{text}>>"
    return text


input_text = gr.Text(label="Input Text")
output_text = gr.Text(label="Output Text")

demo = gr.Interface(inputs=input_text, outputs=output_text, fn=text_to_text)
demo.launch(debug=True)
