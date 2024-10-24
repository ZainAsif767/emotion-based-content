# 1. BERT Fine-tuning Setup and Process

# 1.1 Data Preparation
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
import torch

class EmotionDataset(Dataset):
    def _init_(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def _len_(self):
        return len(self.texts)
    
    def _getitem_(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            return_token_type_ids=True,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def prepare_data(data_path: str):
    """
    Prepare dataset for BERT fine-tuning
    """
    # Load your dataset
    df = pd.read_csv(data_path)
    
    # Convert emotions to numerical labels
    unique_emotions = df['emotion'].unique()
    emotion_to_id = {emotion: idx for idx, emotion in enumerate(unique_emotions)}
    
    # Create numerical labels
    texts = df['text'].values
    labels = df['emotion'].map(emotion_to_id).values
    
    # Split dataset
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=0.2, random_state=42
    )
    
    return train_texts, val_texts, train_labels, val_labels, emotion_to_id

# 1.2 Fine-tuning Process
from transformers import (
    BertForSequenceClassification,
    AdamW,
    get_linear_schedule_with_warmup
)

def train_model(train_dataset, val_dataset, num_labels, epochs=3):
    """
    Fine-tune BERT model on emotion dataset
    """
    # Initialize model
    model = BertForSequenceClassification.from_pretrained(
        'bert-base-uncased',
        num_labels=num_labels
    )
    
    # Setup training parameters
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16)
    
    # Initialize optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=2e-5)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )
    
    # Training loop
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for batch in train_loader:
            optimizer.zero_grad()
            
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            total_loss += loss.item()
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
        
        avg_train_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1} average training loss: {avg_train_loss}")
        
        # Validation
        model.eval()
        val_loss = 0
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                val_loss += outputs.loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        print(f"Validation loss: {avg_val_loss}")
    
    return model

# 2. Environment Setup
def setup_environment():
    """
    Install required packages and setup environment
    """
    # Requirements to be saved in requirements.txt:
    """
    transformers==4.30.0
    torch==2.0.0
    pandas==1.5.3
    numpy==1.24.3
    scikit-learn==1.2.2
    """
    
    # Install packages
    # !pip install -r requirements.txt
    
    # Download required models
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    return tokenizer

# 3. System Configuration and Usage
class EmotionRecommenderConfig:
    def _init_(
        self,
        bert_weight: float = 0.6,
        llm_weight: float = 0.4,
        max_length: int = 128,
        num_recommendations: int = 3
    ):
        self.bert_weight = bert_weight
        self.llm_weight = llm_weight
        self.max_length = max_length
        self.num_recommendations = num_recommendations

# Main execution
def main():
    # 1. Setup environment
    tokenizer = setup_environment()
    
    # 2. Prepare dataset
    data_path = "your_dataset.csv"  # Replace with your dataset path
    train_texts, val_texts, train_labels, val_labels, emotion_to_id = prepare_data(data_path)
    
    # 3. Create datasets
    train_dataset = EmotionDataset(train_texts, train_labels, tokenizer)
    val_dataset = EmotionDataset(val_texts, val_labels, tokenizer)
    
    # 4. Fine-tune model
    num_labels = len(emotion_to_id)
    model = train_model(train_dataset, val_dataset, num_labels)
    
    # 5. Save fine-tuned model
    model.save_pretrained('emotion_bert_model')
    tokenizer.save_pretrained('emotion_bert_model')
    
    # 6. Initialize recommender system
    config = EmotionRecommenderConfig()
    
    # Save emotion mapping
    emotion_mapping = {idx: emotion for emotion, idx in emotion_to_id.items()}
    
    print("Setup complete! The system is ready to use.")

if __name__ == "__main__":
    main()