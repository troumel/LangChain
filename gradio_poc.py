import gradio as gr
 
def greet(name):
    return f"Hello, {name}. Welcome to AI for Developers"
 
demo = gr.Interface(
    fn = greet,
    inputs = gr.Textbox(label="Your name", placeholder="Type your name..."),
    outputs = gr.Textbox(label="Greeting"),
    title="Welcome AI for Developers",
    description="Type your name and get a greeting",
    flagging_mode="never",
)
 
if __name__ == "__main__":
    demo.launch()