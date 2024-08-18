import pandas as pd
import re
import nltk
from transformers import pipeline
import spacy
from sentence_transformers import SentenceTransformer, util
import torch

# Download necessary NLTK data
nltk.download('punkt')

# Download and load the SpaCy model
import subprocess
subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])

nlp = spacy.load('en_core_web_sm')

# Load the sample data
data_path = '/home/shubhankar/Downloads/text_analysis_lib/enterpret_data.xlsx'
df = pd.read_excel(data_path)

# Drop rows where content contains <AUDIO_CONTENT>
df = df[~df['Content'].str.contains('<AUDIO_CONTENT>', na=False)]

# Define the columns to keep based on the assignment's concerns
columns_to_keep = ['ID', 'Content', 'URL', 'Source', 'Type', 'Reasons', 'CreatedAt']  # Replace with actual column names from the assignment
df = df[columns_to_keep]

# Clean text function
def clean_text(text):
    text = re.sub(r'<.*?>', '', text)  # Remove HTML tags
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove special characters and digits
    text = text.lower()  # Convert to lowercase
    return text

# Apply cleaning to the 'Content' column
df['Content'] = df['Content'].apply(lambda x: clean_text(x) if isinstance(x, str) else x)

# Fill missing values in 'Content' with an empty string
df['Content'] = df['Content'].fillna('')

# Split text into sentences using SpaCy for better parsing
def split_text(text):
    doc = nlp(text)
    return [sent.text for sent in doc.sents]

df['Sentences'] = df['Content'].apply(lambda x: split_text(x))

# Load the pre-trained model for claim identification and move it to GPU
classifier = pipeline('zero-shot-classification', model='facebook/bart-large-mnli', device=0)

# Define candidate labels
candidate_labels = ["claim", "feedback", "request"]

# Identify claims and feedback in sentences with a threshold
def identify_claims(text, sentences, candidate_labels, threshold=0.4):
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

df[['Claim_Indices', 'Claim_Sentences']] = df.apply(lambda row: pd.Series(identify_claims(row['Content'], row['Sentences'], candidate_labels)), axis=1)

# Load Sentence-BERT model and move it to GPU
# sbert_model = SentenceTransformer('paraphrase-MiniLM-L6-v2').to('cuda')
sbert_model = SentenceTransformer('bert-base-nli-mean-tokens').to('cuda')


# Compute semantic similarity and count matches
def compute_similarity(row, threshold=0.4):
    claim_sentences = row['Claim_Sentences']
    if not isinstance(row['Reasons'], str):
        return 0, 0  # No reasons provided
    
    reason_sentences = row['Reasons'].split(',')
    
    if not claim_sentences or not reason_sentences:
        return 0, len(reason_sentences)
    
    claim_embeddings = sbert_model.encode(claim_sentences, convert_to_tensor=True, device='cuda')
    reason_embeddings = sbert_model.encode([sentence.strip() for sentence in reason_sentences], convert_to_tensor=True, device='cuda')
    
    matches = 0
    total = len(reason_embeddings)
    
    for claim_emb in claim_embeddings:
        similarities = util.pytorch_cos_sim(claim_emb, reason_embeddings)
        if similarities.max() > threshold:
            matches += 1
    
    # Ensure matches do not exceed the total reasons
    matches = min(matches, total)
    
    return matches, total

df['Matches'], df['Total_Reasons'] = zip(*df.apply(lambda row: compute_similarity(row), axis=1))

# Calculate the match score and log it
df['Match_Score'] = df.apply(lambda row: f"{row['Matches']}/{row['Total_Reasons']}", axis=1)

# Calculate accuracy
df['Accuracy'] = df.apply(lambda row: min((row['Matches'] / row['Total_Reasons']) * 100, 100) if row['Total_Reasons'] > 0 else 0, axis=1)

# Calculate overall accuracy
total_matches = df['Matches'].sum()
total_reasons = df['Total_Reasons'].sum()
overall_accuracy = (total_matches / total_reasons) * 100 if total_reasons > 0 else 0

# Log the results
with open('evaluation_log_pretrained_BART.txt', 'w') as f:
    f.write(f"Average Accuracy: {overall_accuracy}\n")
    for index, row in df.iterrows():
        f.write(f"ID: {row['ID']}, Match_Score: {row['Match_Score']}, Accuracy: {row['Accuracy']}\n")

# Save the resulting DataFrame to a CSV file
output_path = 'processed_data_pretrained.csv'
df.to_csv(output_path, index=False)

# Create structured output for verification
output = df[['ID', 'Claim_Indices', 'Claim_Sentences', 'Match_Score', 'Accuracy']].to_dict(orient='records')
