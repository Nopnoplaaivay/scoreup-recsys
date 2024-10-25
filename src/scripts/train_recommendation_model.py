import logging
from src.models.recommendation import RecommendationModel
from db.questions import QuestionDB

logging.basicConfig(filename='logs/scripts.log', level=logging.INFO)

def train_recommendation_model():
    """Train the recommendation model using collaborative filtering."""
    question_db = QuestionDB()
    questions = question_db.fetch_all_questions()
    
    if not questions:
        logging.error("No questions found in the database for training.")
        print("No questions available for training.")
        return
    
    model = RecommendationModel()
    questions_data = [[q["id"], q["title"], q["difficulty"]] for q in questions]
    
    model.train(questions_data)
    model.save_model("models/recommendation_model.pkl")
    
    logging.info("Trained and saved the recommendation model.")
    print("Recommendation model trained and saved successfully.")

if __name__ == "__main__":
    train_recommendation_model()
