import pandas as pd
from transformers import DistilBertTokenizer

def filter_df_by_threshold(df: pd.DataFrame, limits: dict) -> pd.DataFrame:

    realistic_limits = {
        'votes_helpful': limits["votes_helpful"], # mmaximal 1500 helpful votes
        'votes_funny': limits["votes_funny"], # maximal 1000 funny votes
        'author.playtime_forever': limits["author.playtime_forever"], # Maximale Spielzeit von 15000 Minuten haben
        'timestamp_created': limits["timestamp_created"], # auf das Jahr 2020 beschränken
    }

    df = df[(df['votes_helpful'] <= realistic_limits['votes_helpful']) &
            (df['votes_funny'] <= realistic_limits['votes_funny']) &
            (df['author.playtime_forever'] <= realistic_limits['author.playtime_forever']) &
            (df['timestamp_created'] >= realistic_limits['timestamp_created'])]
    
    return df 

def get_label_distributons(df: pd.DataFrame):
    return (df['recommended'].value_counts() / len(df))


def tokenize_reviews(df: pd.DataFrame, new_col: str, 
                     existing_col: str, max_length: int = 512) -> pd.DataFrame:

    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    df[new_col] = df[existing_col].apply(
        lambda x: tokenizer(x, truncation=True, padding='max_length', max_length=max_length, return_tensors='pt')
    )

    return df
