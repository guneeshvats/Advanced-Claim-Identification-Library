
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer, util
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from transformers import BartForSequenceClassification, BartTokenizer, AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import torch.nn as nn
import argparse

def fine_tune_model(dataset_path, epochs, lr):
    df = pd.read_csv(dataset_path)

    df['Claim_Sentences'] = df['Claim_Sentences'].astype(str).fillna('')
    df['Reasons'] = df['Reasons'].astype(str).fillna('')
    
    claim_sentences_list = df['Claim_Sentences'].apply(lambda x: x.split('\n')).tolist()
    reason_sentences_list = df['Reasons'].apply(lambda x: x.split(',')).tolist()
    
    flattened_claims = []
    flattened_reasons = []
    flattened_labels = []
    
    sbert_model = SentenceTransformer('bert-base-nli-mean-tokens').to('cuda')

    def compute_similarities(claim_sentences, reason_sentences, sbert_model):
        claim_embeddings = [sbert_model.encode(sentence, convert_to_tensor=True, device='cuda') for sentence in claim_sentences]
        reason_embeddings = [sbert_model.encode(sentence, convert_to_tensor=True, device='cuda') for sentence in reason_sentences]
        
        similarities = []
        for claim_emb in claim_embeddings:
            for reason_emb in reason_embeddings:
                cos_sim = util.pytorch_cos_sim(claim_emb, reason_emb).item()
                similarities.append(cos_sim)
        
        return similarities

    def generate_labels(similarities, num_reasons, threshold):
        similarities.sort(reverse=True)
        top_similarities = similarities[:num_reasons]
        labels = [1 if sim >= threshold else 0 for sim in top_similarities]
        return labels
    
    for claim_sentences, reason_sentences in zip(claim_sentences_list, reason_sentences_list):
        similarities = compute_similarities(claim_sentences, reason_sentences, sbert_model)
        num_reasons = len(reason_sentences)
        labels = generate_labels(similarities, num_reasons, threshold=0.1)
        
        flattened_claims.extend(claim_sentences)
        flattened_reasons.extend(reason_sentences)
        flattened_labels.extend(labels)
    
    train_sentences, test_sentences, train_reasons, test_reasons, train_labels, test_labels = train_test_split(
        flattened_claims, flattened_reasons, flattened_labels, test_size=0.1, random_state=42)
    
    train_sentences, val_sentences, train_reasons, val_reasons, train_labels, val_labels = train_test_split(
        train_sentences, train_reasons, train_labels, test_size=0.05, random_state=42)

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

    def encode_batch(batch):
        sentences, reasons, labels = zip(*batch)
        inputs = tokenizer(list(sentences), list(reasons), truncation=True, padding=True, max_length=tokenizer.model_max_length, return_tensors='pt')
        inputs = {key: value.to('cuda') for key, value in inputs.items()}
        inputs['labels'] = torch.tensor(labels).to('cuda')
        return inputs

    train_dataset = TextPairDataset(train_sentences, train_reasons, train_labels)
    val_dataset = TextPairDataset(val_sentences, val_reasons, val_labels)
    
    train_loader = DataLoader(train_dataset, batch_size=3, shuffle=True, collate_fn=encode_batch)
    val_loader = DataLoader(val_dataset, batch_size=3, shuffle=False, collate_fn=encode_batch)
    
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scaler = torch.cuda.amp.GradScaler()
    
    best_val_accuracy = 0
    
    for epoch in range(epochs):
        total_loss = 0
        model.train()

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False):
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                outputs = model(**batch)
                loss = outputs.loss
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{epochs}: Train Loss = {total_loss/len(train_loader)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tuning script.")
    parser.add_argument('--dataset_path', type=str, required=True, help='The path to the dataset for fine-tuning.')
    parser.add_argument('--epochs', type=int, default=3, help='Number of epochs for fine-tuning.')
    parser.add_argument('--lr', type=float, default=5e-5, help='Learning rate for fine-tuning.')
    args = parser.parse_args()
    
    fine_tune_model(args.dataset_path, args.epochs, args.lr)
