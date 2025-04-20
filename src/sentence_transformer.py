 
# sentence_transformer.py

from transformers import AutoTokenizer, AutoModel
import torch

class SentenceTransformer:
    def __init__(self, model_name="bert-base-uncased"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output.last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def encode(self, sentences):
        if isinstance(sentences, str):
            sentences = [sentences]
        inputs = self.tokenizer(sentences, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
        sentence_embeddings = self.mean_pooling(outputs, inputs['attention_mask'])
        return sentence_embeddings
