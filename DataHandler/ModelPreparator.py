
import pandas as pd
import torch
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from DataHandler.utils import tokenize_reviews
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class ModelPreparator():

    def __init__(self):
        pass

    def prepare_for_bert(self, df: pd.DataFrame, cols_mapping: dict = None) -> pd.DataFrame:

        for new_col, existing_col in cols_mapping.items():
            df = tokenize_reviews(df=df, new_col=new_col, existing_col=existing_col, max_length=512)
        return df
            

    def add_sarcasm_score_to_review(self, df: pd.DataFrame, batch_size: int) -> pd.DataFrame:
        
        MODEL_PATH = "helinivan/english-sarcasm-detector"

        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH).to(device)

        def predict_sarcasm_batch(texts: list, tokenizer, model) -> list:
            inputs = tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = model(**inputs)
            probs = outputs.logits.softmax(dim=-1).cpu().tolist()
            results = []

            for sarcasm_confidence in probs:
                confidence_score = max(sarcasm_confidence)

                if confidence_score < 0.4:
                    level = "LOW"
                elif 0.4 <= confidence_score < 0.7:
                    level = "MEDIUM"
                else:
                    level = "HIGH"

                results.append(f"SARCASM CONFIDENCE: {level}")
            return results

        modified_reviews = []
        reviews = df['cleaned_review'].tolist()

        # Iterate through the DataFrame in batches
        for i in tqdm(range(0, len(reviews), batch_size), desc="Processing reviews"):
            batch_reviews = reviews[i:i + batch_size]
            batch_results = predict_sarcasm_batch(batch_reviews, tokenizer, model)
            modified_reviews.extend([f"{review} [{result}]" for review, result in zip(batch_reviews, batch_results)])

        # Add the new column to the DataFrame
        df['cleaned_review_sarcasm'] = modified_reviews

        return df
    

    def make_train_test_split(self, df: pd.DataFrame):

        train_df, val_df = train_test_split(df, test_size=0.2, stratify=df['recommended'], random_state=42)
        
        print(f"Training Set: {len(train_df)} Reviews")
        print(f"Validation Set: {len(val_df)} Reviews")

        return train_df, val_df




