 
# test_sentences.py

from sentence_transformer import SentenceTransformer

model = SentenceTransformer()

sentences = [
    "Transformers are powerful models for NLP.",
    "Machine learning is fascinating.",
    "Let's build a multi-task learning model!"
]

embeddings = model.encode(sentences)

print("Shape of embeddings:", embeddings.shape)
print("First sentence embedding:\n", embeddings[0])
