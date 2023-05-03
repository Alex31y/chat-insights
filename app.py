from pygpt4all.models.gpt4all_j import GPT4All_J

def new_text_callback(text):
    print(text, end="")

model = GPT4All_J(r"C:\Users\xlits\Downloads\ggml-gpt4all-j-v1.3-groovy.bin")
model.prompt('what is the capital of norway?')