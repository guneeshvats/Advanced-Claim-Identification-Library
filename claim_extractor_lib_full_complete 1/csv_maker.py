
import pandas as pd
import torch
import json
from transformers import BartForSequenceClassification, BartTokenizer
from sentence_transformers import SentenceTransformer, util
import argparse

def clean_text(text):
    text = re.sub(r'<.*?>', '', text)  
    text = re.sub(r'[^a-zA-Z\s]', '', text)  
    text = text.lower()  
    return text

def split_text(text):
    doc = nlp(text)
    return [sent.text for sent in doc.sents]

def identify_claims(text, sentences, model, tokenizer, candidate_labels, threshold=0.2):
    indices = []
    claim_sentences = []
    for sentence in sentences:
        inputs = tokenizer(sentence, return_tensors="pt", truncation=True, max_length=512).to('cuda')
        with torch.no_grad():
            outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.nn.functional.softmax(logits, dim=-1)
        scores = {label: prob for label, prob in zip(candidate_labels, probs.cpu().numpy()[0])}
        for label, score in scores.items():
            if label in ["claim", "feedback"] and score > threshold:
                start_index = text.find(sentence)
                indices.append(start_index)
                claim_sentences.append(sentence)
                break
    return indices, claim_sentences

def make_csv(trained_model_path, input_file):
    df = pd.read_excel(input_file)
    df = df[~df['Content'].str.contains('<AUDIO_CONTENT>', na=False)]
    df['Content'] = df['Content'].apply(lambda x: clean_text(x) if isinstance(x, str) else x)
    df['Content'] = df['Content'].fillna('')
    df['Sentences'] = df['Content'].apply(lambda x: split_text(x))
    
    model = BartForSequenceClassification.from_pretrained(trained_model_path).to('cuda')
    tokenizer = BartTokenizer.from_pretrained(trained_model_path)
    
    candidate_labels = ["claim", "feedback", "request"]
    
    df[['Claim_Indices', 'Claim_Sentences']] = df.apply(
        lambda row: pd.Series(identify_claims(row['Content'], row['Sentences'], model, tokenizer, candidate_labels)), axis=1)

    # Save the results as CSV and JSON
    df.to_csv("predictions.csv", index=False)
    
    id_claim_indices = df[['ID', 'Claim_Indices']].to_dict(orient='records')
    with open("predictions.json", 'w') as json_file:
        json.dump(id_claim_indices, json_file, indent=4)
    
    print("CSV and JSON files created successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CSV Maker script.")
    parser.add_argument('--trained_model_path', type=str, required=True, help='Path to the fine-tuned model.')
    parser.add_argument('--input_file', type=str, required=True, help='The input file for generating CSV and JSON outputs.')
    
    args = parser.parse_args()
    
    make_csv(args.trained_model_path, args.input_file)
