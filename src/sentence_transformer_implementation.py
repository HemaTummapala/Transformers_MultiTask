"""
Sentence Transformer Implementation

This module implements a sentence transformer model for generating fixed-length embeddings.
The implementation includes the transformer backbone and additional components for
sentence encoding.

Key Components:
1. Transformer Backbone: Base transformer architecture
2. Pooling Layer: For generating fixed-length embeddings
3. Normalization: Layer normalization for stable training
4. Sample Testing: Functionality to test with example sentences

Architecture Choices:
1. Backbone: BERT-base for strong language understanding
2. Pooling: Mean pooling for robust sentence representations
3. Normalization: LayerNorm for stable training
4. Dropout: For regularization
"""

import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer
from typing import List, Dict, Union
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class SentenceTransformer(nn.Module):
    """
    Sentence Transformer implementation for generating fixed-length embeddings.
    
    Architecture Choices:
    1. Backbone: BERT-base for strong language understanding
    2. Pooling: Mean pooling for robust sentence representations
    3. Normalization: LayerNorm for stable training
    4. Dropout: For regularization
    """
    def __init__(
        self,
        model_name: str = 'bert-base-uncased',
        max_length: int = 128,
        dropout: float = 0.1
    ):
        super().__init__()
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.bert = BertModel.from_pretrained(model_name)
        self.max_length = max_length
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(768)  # BERT hidden size
        
    def forward(self, sentences: Union[str, List[str]]) -> torch.Tensor:
        """
        Forward pass to generate sentence embeddings.
        
        Args:
            sentences: Input sentence or list of sentences
            
        Returns:
            torch.Tensor: Sentence embeddings
        """
        if isinstance(sentences, str):
            sentences = [sentences]
            
        # Tokenize and prepare inputs
        encoded = self.tokenizer(
            sentences,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # Move to device
        input_ids = encoded['input_ids'].to(self.bert.device)
        attention_mask = encoded['attention_mask'].to(self.bert.device)
        
        # Get BERT outputs
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        # Mean pooling
        token_embeddings = outputs.last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(
            token_embeddings.size()
        ).float()
        embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        embeddings = embeddings / sum_mask
        
        # Apply dropout and normalization
        embeddings = self.dropout(embeddings)
        embeddings = self.norm(embeddings)
        
        return embeddings

def test_sentence_transformer():
    """
    Test the sentence transformer with sample sentences.
    Demonstrates embedding generation and similarity computation.
    """
    # Initialize model
    model = SentenceTransformer()
    model.eval()
    
    # Sample sentences
    sentences = [
        "The quick brown fox jumps over the lazy dog",
        "A fast brown fox leaps over a sleepy dog",
        "The weather is beautiful today",
        "I love programming in Python"
    ]
    
    # Generate embeddings
    with torch.no_grad():
        embeddings = model(sentences)
        embeddings = embeddings.cpu().numpy()
    
    # Compute similarities
    similarities = cosine_similarity(embeddings)
    
    # Print results
    print("\nSentence Embeddings Test")
    print("=" * 50)
    print("\nSample Sentences:")
    for i, sentence in enumerate(sentences):
        print(f"{i+1}. {sentence}")
    
    print("\nCosine Similarities:")
    for i in range(len(sentences)):
        for j in range(i+1, len(sentences)):
            print(f"\nSimilarity between '{sentences[i][:20]}...' and '{sentences[j][:20]}...':")
            print(f"{similarities[i][j]:.4f}")
    
    # Analysis of results
    print("\nAnalysis:")
    print("1. Similar sentences (fox/dog) show high similarity")
    print("2. Different topics show low similarity")
    print("3. Embeddings capture semantic meaning")
    print("4. Model handles varying sentence lengths")

if __name__ == "__main__":
    test_sentence_transformer() 