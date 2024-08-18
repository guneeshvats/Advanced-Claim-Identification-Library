
import pandas as pd
import re
import nltk
from transformers import pipeline
import spacy
from sentence_transformers import SentenceTransformer, util
import torch
import argparse

# Download necessary NLTK data
nltk.download('punkt')

# Download and load the SpaCy model
import subprocess
subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])

nlp = spacy.load('en_core_web_sm')

def clean_text(text):
    text = re.sub(r'<.*?>', '', text)  
    text = re.sub(r'[^a-zA-Z\s]', '', text)  
    text = text.lower()  
    return text

def split_text(text):
    doc = nlp(text)
    return [sent.text for sent in doc.sents]

def identify_claims(text, sentences, candidate_labels, threshold=0.2):
    classifier = pipeline('zero-shot-classification', model='facebook/bart-large-mnli', device=0)
    indices = []
    claim_sentences = []
    for sentence in sentences:
        result = classifier(sentence, candidate_labels, truncation=True, max_length=512)
        for label, score in zip(result['labels'], result['scores']):
            if label in ["claim", "feedback"] and score > threshold:
                start_index = text.find(sentence)
                indices.append(start_index)
                claim_sentences.append(sentence)
                break
    return indices, claim_sentences

def zero_shot_classification(input_file):
    df = pd.read_excel(input_file)
    df = df[~df['Content'].str.contains('<AUDIO_CONTENT>', na=False)]
    df['Content'] = df['Content'].apply(lambda x: clean_text(x) if isinstance(x, str) else x)
    df['Content'] = df['Content'].fillna('')
    df['Sentences'] = df['Content'].apply(lambda x: split_text(x))

    candidate_labels = ["claim", "feedback", "request"]
    
    df[['Claim_Indices', 'Claim_Sentences']] = df.apply(
        lambda row: pd.Series(identify_claims(row['Content'], row['Sentences'], candidate_labels)), axis=1)

    df.to_csv("zsc_output.csv", index=False)
    print("Zero-shot classification completed and saved to 'zsc_output.csv'.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Zero-shot classification script.")
    parser.add_argument('--input_file', type=str, required=True, help='The input file for zero-shot classification.')
    args = parser.parse_args()
    
    zero_shot_classification(args.input_file)
