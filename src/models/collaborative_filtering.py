import logging
import pandas as pd
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
from surprise.accuracy import rmse

logging.basicConfig(filename='logs/model_training.log', level=logging.INFO)

class CollaborativeFiltering:
    def __init__(self):
        self.model = SVD()
        self.trainset = None

    def load_data(self, df):
        """Load the dataset from a pandas DataFrame."""
        reader = Reader(rating_scale=(0, 1))  # Rating scale can be adjusted as needed.
        data = Dataset.load_from_df(df[['user_id', 'question_id', 'rating']], reader)
        self.trainset, self.testset = train_test_split(data, test_size=0.25)
        logging.info("Collaborative filtering data loaded successfully.")

    def train(self):
        """Train the SVD model on the dataset."""
        if self.trainset is None:
            raise ValueError("Training data not loaded. Use `load_data` to provide data.")
        
        self.model.fit(self.trainset)
        logging.info("Collaborative filtering model training completed.")
        
        predictions = self.model.test(self.testset)
        rmse_score = rmse(predictions)
        logging.info(f"Collaborative filtering RMSE: {rmse_score:.4f}")

    def predict(self, user_id, question_id):
        """Predict the rating for a given user and question."""
        return self.model.predict(user_id, question_id).est

    def recommend(self, user_id, top_n=10):
        """Recommend top N questions for a given user based on the SVD model."""
        all_items = self.trainset.all_items()
        question_ids = [self.trainset.to_raw_iid(i) for i in all_items]

        predicted_ratings = [self.predict(user_id, question_id) for question_id in question_ids]
        top_question_ids = sorted(range(len(predicted_ratings)), key=lambda i: predicted_ratings[i], reverse=True)[:top_n]
        
        recommended_questions = [question_ids[i] for i in top_question_ids]
        return recommended_questions
