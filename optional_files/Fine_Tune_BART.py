import pandas as pd
import torch
from sentence_transformers import SentenceTransformer, util
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from transformers import BartForSequenceClassification, BartTokenizer, AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import torch.nn as nn

# Function to compute similarities between claim and reason sentences
def compute_similarities(claim_sentences, reason_sentences, sbert_model):
    claim_embeddings = [sbert_model.encode(sentence, convert_to_tensor=True, device='cuda') for sentence in claim_sentences]
    reason_embeddings = [sbert_model.encode(sentence, convert_to_tensor=True, device='cuda') for sentence in reason_sentences]
    
    similarities = []
    for claim_emb in claim_embeddings:
        for reason_emb in reason_embeddings:
            cos_sim = util.pytorch_cos_sim(claim_emb, reason_emb).item()
            similarities.append(cos_sim)
    
    return similarities

# Function to generate labels based on similarities
def generate_labels(similarities, num_reasons, threshold):
    similarities.sort(reverse=True)
    top_similarities = similarities[:num_reasons]
    labels = [1 if sim >= threshold else 0 for sim in top_similarities]
    return labels

# Load the data
data_path = '/home/shubhankar/Downloads/text_analysis_lib/processed_data_pretrained.csv'
df = pd.read_csv(data_path)

# Ensure all values are strings and handle NaNs
df['Claim_Sentences'] = df['Claim_Sentences'].astype(str).fillna('')
df['Reasons'] = df['Reasons'].astype(str).fillna('')

claim_sentences_list = df['Claim_Sentences'].apply(lambda x: x.split('\n')).tolist()
reason_sentences_list = df['Reasons'].apply(lambda x: x.split('\n')).tolist()

# Flatten the data for training
all_labels = []
flattened_claims = []
flattened_reasons = []
flattened_labels = []

# Load the setence embeddings model
sbert_model = SentenceTransformer('bert-base-nli-mean-tokens').to('cuda')

# Compute similarities and generate labels for each claim and reason pair
for claim_sentences, reason_sentences in zip(claim_sentences_list, reason_sentences_list):
    similarities = compute_similarities(claim_sentences, reason_sentences, sbert_model)
    num_reasons = len(reason_sentences)
    labels = generate_labels(similarities, num_reasons, threshold=0.1)
    
    flattened_claims.extend(claim_sentences)
    flattened_reasons.extend(reason_sentences)
    flattened_labels.extend(labels)

# Ensure all lists are of the same length
assert len(flattened_claims) == len(flattened_reasons) == len(flattened_labels)

# Split the data into train, validation and test sets
train_sentences, test_sentences, train_reasons, test_reasons, train_labels, test_labels = train_test_split(
    flattened_claims, flattened_reasons, flattened_labels, test_size=0.1, random_state=42)

train_sentences, val_sentences, train_reasons, val_reasons, train_labels, val_labels = train_test_split(
    train_sentences, train_reasons, train_labels, test_size=0.05, random_state=42)

# Load the BART model with dropout
model = BartForSequenceClassification.from_pretrained('facebook/bart-large-mnli', attention_dropout=0.2, dropout=0.2).to('cuda')
tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-mnli')

class TextPairDataset(Dataset):
    def __init__(self, sentences, reasons, labels):
        self.sentences = sentences
        self.reasons = reasons
        self.labels = labels

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        return self.sentences[idx], self.reasons[idx], self.labels[idx]

max_length_model = tokenizer.model_max_length

# Function to encode a batch of data
def encode_batch(batch):
    sentences, reasons, labels = zip(*batch)
    inputs = tokenizer(list(sentences), list(reasons), truncation=True, padding=True, max_length=max_length_model, return_tensors='pt')
    inputs = {key: value.to('cuda') for key, value in inputs.items()}
    inputs['labels'] = torch.tensor(labels).to('cuda')
    return inputs

batch_size = 3

train_dataset = TextPairDataset(train_sentences, train_reasons, train_labels)
val_dataset = TextPairDataset(val_sentences, val_reasons, val_labels)
test_dataset = TextPairDataset(test_sentences, test_reasons, test_labels)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=encode_batch)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=encode_batch)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=encode_batch)

# Mixed Precision Training
scaler = torch.cuda.amp.GradScaler()

# Optimizer and scheduler
optimizer = AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=3, verbose=True)

epochs = 50
best_val_accuracy = 0

# Training loop
for epoch in range(epochs):
    total_loss = 0
    model.train()

    # Training loop
    for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False):
        optimizer.zero_grad()
        
        with torch.cuda.amp.autocast():
            outputs = model(**batch)
            loss = outputs.loss

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

    avg_train_loss = total_loss / len(train_loader)
    
    # Validation loop
    model.eval()
    val_predictions = []
    val_labels = []
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validation", leave=False):
            with torch.cuda.amp.autocast():
                outputs = model(**batch)
                logits = outputs.logits
                val_predictions.extend(torch.argmax(logits, dim=1).cpu().numpy())
                val_labels.extend(batch['labels'].cpu().numpy())

    val_accuracy = accuracy_score(val_labels, val_predictions)
    
    # Learning rate scheduling based on validation accuracy
    scheduler.step(val_accuracy)
    
    # Checkpoint the best model
    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        model.save_pretrained('best_model')
        tokenizer.save_pretrained('best_model')
    
    print(f"Epoch {epoch+1}/{epochs}: Train Loss = {avg_train_loss:.4f}, Validation Accuracy = {val_accuracy:.4f}, Best Accuracy = {best_val_accuracy:.4f}")

# Testing loop
model.eval()
test_predictions = []
test_labels = []
with torch.no_grad():
    for batch in tqdm(test_loader, desc="Testing", leave=False):
        with torch.cuda.amp.autocast():
            outputs = model(**batch)
            logits = outputs.logits
            test_predictions.extend(torch.argmax(logits, dim=1).cpu().numpy())
            test_labels.extend(batch['labels'].cpu().numpy())

test_accuracy = accuracy_score(test_labels, test_predictions)
print(f"Test Accuracy: {test_accuracy:.4f}")
