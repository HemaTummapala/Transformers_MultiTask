"""
Multi-Task Learning Expansion for Sentence Transformer

This module extends the sentence transformer to handle multiple NLP tasks:
1. Task A: Sentence Classification
2. Task B: Sentiment Analysis

Key Components:
1. Shared Encoder: Base transformer for feature extraction
2. Task-Specific Heads: Separate heads for each task
3. Loss Functions: Task-specific and combined loss computation
4. Training Logic: Multi-task training implementation

Architecture Changes:
1. Added classification head for sentence classification
2. Added sentiment head for sentiment analysis
3. Implemented shared feature extraction
4. Added task-specific loss computation
"""

from typing import Dict, List, Tuple
from sentence_transformer_implementation import SentenceTransformer
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

class MultiTaskTransformer(nn.Module):
    def __init__(self, num_classes: int = 5, num_sentiments: int = 3):
        super().__init__()
        self.encoder = SentenceTransformer()
        hidden_size = self.encoder.bert.config.hidden_size
        
        # Classification head
        self.classification_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, num_classes)
        )
        
        # Sentiment head
        self.sentiment_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, num_sentiments)
        )
    
    def forward(self, sentences: List[str]) -> Dict[str, torch.Tensor]:
        """
        Forward pass for multi-task learning.
        
        Args:
            sentences: Input sentences
            
        Returns:
            Dictionary containing embeddings and task-specific outputs
        """
        # Get shared embeddings
        embeddings = self.encoder(sentences)
        
        # Task-specific predictions
        outputs = {
            'embeddings': embeddings,
            'classification_logits': self.classification_head(embeddings),
            'sentiment_logits': self.sentiment_head(embeddings)
        }
        
        return outputs

class TrainingMonitor:
    def __init__(self):
        self.history = {
            'classification_loss': [],
            'sentiment_loss': [],
            'total_loss': []
        }
    
    def update(self, classification_loss: float, sentiment_loss: float, total_loss: float):
        self.history['classification_loss'].append(classification_loss)
        self.history['sentiment_loss'].append(sentiment_loss)
        self.history['total_loss'].append(total_loss)
    
    def plot_training_history(self):
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        plt.plot(self.history['classification_loss'], label='Classification Loss')
        plt.plot(self.history['sentiment_loss'], label='Sentiment Loss')
        plt.title('Task-Specific Losses')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(self.history['total_loss'], label='Total Loss')
        plt.title('Combined Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('training_history.png')
        plt.close()

def train_model(
    model: MultiTaskTransformer,
    dataloader: DataLoader,
    num_epochs: int = 5,
    learning_rate: float = 2e-5
) -> Dict[str, List[float]]:
    """
    Train the multi-task model.
    
    Args:
        model: The MultiTaskTransformer model
        dataloader: DataLoader containing training data
        num_epochs: Number of training epochs
        learning_rate: Learning rate for optimizer
        
    Returns:
        Training history
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Initialize monitor
    monitor = TrainingMonitor()
    
    # Loss functions
    classification_criterion = nn.CrossEntropyLoss()
    sentiment_criterion = nn.CrossEntropyLoss()
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    print("\nTraining multi-task model")
    print("-" * 50)
    
    for epoch in range(num_epochs):
        model.train()
        epoch_classification_loss = 0
        epoch_sentiment_loss = 0
        epoch_total_loss = 0
        
        for batch in dataloader:
            # Move batch to device
            class_labels = batch['class_labels'].to(device)
            sentiment_labels = batch['sentiment_labels'].to(device)
            
            # Forward pass
            outputs = model(batch['sentences'])
            
            # Calculate losses
            classification_loss = classification_criterion(
                outputs['classification_logits'],
                class_labels
            )
            sentiment_loss = sentiment_criterion(
                outputs['sentiment_logits'],
                sentiment_labels
            )
            
            # Combined loss
            total_loss = classification_loss + sentiment_loss
            
            # Backward pass
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            # Accumulate losses
            epoch_classification_loss += classification_loss.item()
            epoch_sentiment_loss += sentiment_loss.item()
            epoch_total_loss += total_loss.item()
        
        # Average losses for the epoch
        num_batches = len(dataloader)
        monitor.update(
            epoch_classification_loss / num_batches,
            epoch_sentiment_loss / num_batches,
            epoch_total_loss / num_batches
        )
        
        print(f"Epoch {epoch + 1}/{num_epochs}")
        print(f"Classification Loss: {epoch_classification_loss / num_batches:.4f}")
        print(f"Sentiment Loss: {epoch_sentiment_loss / num_batches:.4f}")
        print(f"Total Loss: {epoch_total_loss / num_batches:.4f}")
        print("-" * 50)
    
    # Plot training history
    monitor.plot_training_history()
    
    return monitor.history

def main():
    # Initialize model
    model = MultiTaskTransformer()
    
    # Example training data
    train_data = {
        'sentences': [
            "The movie was absolutely fantastic and I loved every minute of it!",
            "This product is terrible and I want my money back.",
            "The weather is nice today.",
            "I'm not sure how I feel about this situation.",
            "The service was average, nothing special."
        ],
        'class_labels': [0, 1, 2, 3, 4],  # Different classes
        'sentiment_labels': [2, 0, 1, 1, 1]  # 0: negative, 1: neutral, 2: positive
    }
    
    # Create dataset and dataloader
    from torch.utils.data import Dataset, DataLoader
    
    class MultiTaskDataset(Dataset):
        def __init__(self, sentences, class_labels, sentiment_labels):
            self.sentences = sentences
            self.class_labels = torch.tensor(class_labels)
            self.sentiment_labels = torch.tensor(sentiment_labels)
        
        def __len__(self):
            return len(self.sentences)
        
        def __getitem__(self, idx):
            return {
                'sentences': self.sentences[idx],
                'class_labels': self.class_labels[idx],
                'sentiment_labels': self.sentiment_labels[idx]
            }
    
    dataset = MultiTaskDataset(
        train_data['sentences'],
        train_data['class_labels'],
        train_data['sentiment_labels']
    )
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    
    # Train model
    history = train_model(model, dataloader)
    print("\nTraining completed")
    print("-" * 50)

if __name__ == "__main__":
    main() 