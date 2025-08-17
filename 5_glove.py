#5. GloVe Pre-trained Embeddings
import gensim.downloader as api

glove_model = api.load("glove-wiki-gigaword-50")

print("Vector for 'computer':", glove_model['computer'])
print("Similarity between 'computer' and 'laptop':", glove_model.similarity('computer', 'laptop'))
