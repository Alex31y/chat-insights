import tkinter as tk
import re
from tkinter import filedialog
from langchain.document_loaders import PyPDFLoader
import urllib.request
import fitz
import re
import numpy as np
from sentence_transformers import SentenceTransformer
import openai
import gradio as gr
import os
from sklearn.neighbors import NearestNeighbors
from sentence_transformers import SentenceTransformer, util
import torch
import shutil

chunk_size = 50
n_chunks = 15
model = SentenceTransformer("bert model")


def preprocess(text):
    text = text.replace('\n', ' ')
    text = re.sub('\s+', ' ', text)
    return text

def to_text(file_paths):
    if file_paths:
        text = ""
        for file in file_paths:
                loader = PyPDFLoader(file)
                pages = loader.load_and_split()
                for page in pages:
                    page = page.page_content
                    page = preprocess(page)
                    text += page
    return text

def text_to_chunks(texts, word_length=chunk_size, start_page=1):
    text_toks = [t.split(' ') for t in texts]
    page_nums = []
    chunks = []

    for idx, words in enumerate(text_toks):
        for i in range(0, len(words), word_length):
            chunk = words[i:i + word_length]
            if (i + word_length) > len(words) and (len(chunk) < word_length) and (
                    len(text_toks) != (idx + 1)):
                text_toks[idx + 1] = chunk + text_toks[idx + 1]
                continue
            chunk = ' '.join(chunk).strip()
            chunk = f'[{idx + start_page}]' + ' ' + '"' + chunk + '"'
            chunks.append(chunk)
    return chunks

class SemanticSearch:

    def __init__(self):
        self.fitted = False

    def encode(self, chunk):
        embeddings = model.encode(chunk, convert_to_tensor=True)
        return embeddings

    # applico il nearest neighbors sull'embedding del pdf
    def fit(self, data, batch=1000, n_neighbors=5):
        self.data = data  # salvo i chunks del pdf in data
        self.corpus_embeddings = self.get_text_embedding(data, batch=batch)  # qui creo gli embedding
        self.fitted = True

    # restituisco i top n chunks più simili alla domanda
    def __call__(self, text): # text è la domanda input dell'utente
        top_k = min(n_chunks, self.dim_corpus)
        domanda_embeddings = self.encode([text]) # embedding applicato alla domanda
        cos_scores = util.cos_sim(domanda_embeddings, self.corpus_embeddings)[0]
        top_results = torch.topk(cos_scores, k=top_k)
        return [self.data[i] for i in top_results[1]]

    # questa è la classe che fa l'embedding sul contenuto del pdf
    def get_text_embedding(self, texts, batch):
        embeddings = []
        print("Numero di chunks nel documento: ")
        print(len(texts))
        self.dim_corpus = len(texts)

        for i in range(0, len(texts), batch):
            text_batch = texts[i:(i + batch)]

            emb_batch = self.encode(text_batch)  # chiamo il modello
            embeddings.append(emb_batch)
        embeddings = np.vstack(embeddings)  #a function provided by NumPy that takes a sequence of arrays and stacks them vertically to form a new array.
        return embeddings

def generate_answer(question, openAI_key):
    # genero i chunks
    topn_chunks = recommender(question) # metodo __call__ : confronto l'embedding della domanda all'embedding del pdf ed ottengo gli n snippet di testo più vicini
    prompt = ""
    prompt += 'search results:\n\n'
    for c in topn_chunks:
        prompt += c + '\n\n'

    prompt += "Instructions: Compose a comprehensive reply to the query using the search results given. " \
              "If the search results mention multiple subjects with the same name, create separate answers for each. " \
              "Only include information found in the results and don't add any additional information. " \
              "Make sure the answer is correct and don't output false content. " \
              "If the text does not relate to the query, simply state 'Non è stata trovata una risposta alla tua domanda nel testo'." \
              "Ignore outlier search results which has nothing to do with the question. Only answer what is asked. " \
              "The answer should be short and concise. Answer step-by-step. \n\nQuery: {question}\nAnswer: "

    prompt += f"Query: {question}\nAnswer:"

    """
    file_object = open('docs\domande.txt', 'a')
    file_object.write('\n\n\nprompt:\n')
    file_object.write(prompt)
    # Close the file
    file_object.close()
    """

    #answer = generate_text(openAI_key, prompt, "text-davinci-003")
    answer = prompt
    return answer

def run():
    query = query_entry.get()
    api_key = key_entry.get()
    global file_paths
    text = to_text(file_paths)
    chunks = text_to_chunks(text)

    recommender.fit(chunks)
    answer = generate_answer(query, api_key)

    text_area.delete(1.0, tk.END)
    text_area.insert(tk.END, answer)

def input_file():
    global file_paths
    file_paths = filedialog.askopenfilenames(filetypes=[("PDF Files", "*.pdf")])

recommender = SemanticSearch()
# Create the main window
window = tk.Tk()

file_paths = None
select_button = tk.Button(window, text="Select PDFs", command=input_file)
select_button.pack()

key_label = tk.Label(window, text="API Key:")
key_label.pack()
key_entry = tk.Entry(window)
key_entry.pack()
query_label = tk.Label(window, text="Question Query:")
query_label.pack()
query_entry = tk.Entry(window)
query_entry.pack()

# Create button to retrieve the query and API key
submit_button = tk.Button(window, text="Submit", command=run)
submit_button.pack()

# Create a text area to display the extracted text
text_area = tk.Text(window)
text_area.pack()

# Start the main event loop
window.mainloop()
