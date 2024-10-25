from src.models.embedding import QuestionEmbedding

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.preprocessing import normalize

from src.models.embedding import QuestionEmbedding

class PreprocessX:
    def __init__(self) -> None:
        self.df = None
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()

    def encode(self, df=None):

        embedder = QuestionEmbedding(
            vector_size=50,
            window=4,
            min_count=2,
            epochs=100
        )

        texts, stats = embedder.prepare_data(df)
        print("Thống kê dữ liệu:", stats)
 
        embedder.train_model(texts)

        df = embedder.create_embeddings(df)

        df['chapter_encoded'] = self.label_encoder.fit_transform(df['chapter'])
        df['difficulty_scaled'] = self.scaler.fit_transform(df[['difficulty']])
        df = df[['question_id', 'chapter_encoded', 'difficulty_scaled', 'combined_embedding']]
        combined_embedding_df = pd.DataFrame(df['combined_embedding'].tolist(), index=df.index)
        df = pd.concat([df, combined_embedding_df], axis=1)
    
        df = df[['chapter_encoded', 'difficulty_scaled'] + list(combined_embedding_df.columns)]
        df.columns = df.columns.astype(str)

        self.df = df
