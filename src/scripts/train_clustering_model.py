import logging
from models.clustering import KMeansModel
from db.questions import QuestionDB

logging.basicConfig(filename='logs/scripts.log', level=logging.INFO)

def train_clustering_model():
    """Train the clustering model using questions from the database."""
    question_db = QuestionDB()
    questions = question_db.fetch_all_questions()
    
    if not questions:
        logging.error("No questions found in the database for training.")
        print("No questions available for training.")
        return
    
    model = KMeansModel()
    questions_data = [[q["difficulty"]] for q in questions]
    n_clusters = 3  # You can tune this number or use a heuristic
    model.train(questions_data, n_clusters)
    model.save_model("src/models/clustering_model.pkl")
    
    logging.info("Trained and saved the clustering model.")
    print("Clustering model trained and saved successfully.")

if __name__ == "__main__":
    train_clustering_model()
