def get_text_embedding(texts, batch):
    corpus = []
    print("Numero di chunks nel documento: ")
    print(len(texts))

    for i in range(0, len(texts), batch):
        text_batch = texts[i:(i + batch)]
        for text in text_batch:
            chunk = ['a fixed part', text]
            print(chunk)
            corpus.append(chunk)
    return corpus



texts = [
    "Text 1",
    "Text 2",
    "Text 3"
]

batch_size = 1
embeddings = get_text_embedding(texts, batch=1000)
print(embeddings)