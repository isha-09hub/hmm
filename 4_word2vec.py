#4. Word2Vec Embeddings
from gensim.models import Word2Vec

sentences = [
    ["artificial", "intelligence", "is", "cool"],
    ["machine", "learning", "is", "fun"],
    ["ai", "learning", "uses", "neural", "networks"]
]
model = Word2Vec(sentences, vector_size=10, window=2, min_count=1, sg=1)

print("Vector for 'learning':", model.wv['learning'])
print("Most similar to 'learning':", model.wv.most_similar('learning'))

