from typing import Dict
import pandas as pd
import re

from DataHandler.utils import filter_df_by_threshold, tokenize_reviews


class DataHandler():

    def __init__(self):
        
        self.DEFAULT_LIMITS = {
            'votes_helpful': 1500, 
            'votes_funny': 1000, 
            'author.playtime_forever': 15000, 
            'timestamp_created': pd.Timestamp(2020, 1, 1)
        }

    def read_data(self, file_path: str) -> pd.DataFrame:

        df = pd.read_parquet(file_path)
        return df
    
    def preprocess_data(self, df: pd.DataFrame, limits: dict = None) -> pd.DataFrame:

        if limits is None:
            limits = self.DEFAULT_LIMITS.copy()

        df["timestamp_created"] = pd.to_datetime(df["timestamp_created"], unit='s')
        df = filter_df_by_threshold(df=df, limits=limits)

        df = df.drop_duplicates()
        df = df.dropna(subset=["review"])

        df = df.sort_values(by=['votes_helpful', 'votes_funny'], ascending=False)
        df = df[(df['votes_helpful'] > 0) | (df['votes_funny'] > 0)]

        reviews_per_game = df['app_id'].value_counts()
        lower_bound = reviews_per_game.quantile(0.25)
        filtered_games = reviews_per_game[(reviews_per_game >= lower_bound)].index
        df_filtered = df[df['app_id'].isin(filtered_games)]
        df_filtered['cleaned_review'] = df_filtered['review'].apply(
            lambda x: re.sub(r'\s+', ' ', re.sub(r'\[.*?\]', '', x)).strip() if isinstance(x, str) else ''
        )
        df_filtered['cleaned_review'] = df_filtered['cleaned_review'].str.lower()

        return df_filtered

    def choose_dataset_size(self, df: pd.DataFrame, 
                            row_number: int = 100000) -> pd.DataFrame:
        # keeps the distribution (could add weights here)
        return df.sample(n=row_number)

        