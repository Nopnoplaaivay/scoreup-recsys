import logging
from db.questions import QuestionDB
from db.logs import LogsDB

logging.basicConfig(filename='logs/scripts.log', level=logging.INFO)

def insert_initial_questions():
    """Insert some initial questions into the MongoDB collection."""
    question_db = QuestionDB()
    initial_questions = [
        {"id": "ea548305-a5f2-4ff5-bf08-5d07fe556800", "chapter": "chuong-1", "difficulty": 0.50, "title": "giai-quyet-bai-toan-tren-mtdt"},
        {"id": "b96b0f67-56ba-49b6-a619-69c7f770ef3e", "chapter": "chuong-1", "difficulty": 0.12, "title": "giai-quyet-bai-toan-tren-mtdt"},
        {"id": "86b4b5d1-110e-4b57-990a-1f4a608ea141", "chapter": "chuong-1", "difficulty": 0.50, "title": "may-tinh-dien-tu"}
    ]
    
    for question in initial_questions:
        question_db.insert_question(question)
        logging.info(f"Inserted question with ID: {question['id']}")
    
    print("Initial questions inserted successfully.")

def insert_log_entry():
    """Insert a log entry to track database operations."""
    logs_db = LogsDB()
    log_data = {
        "event": "Database initialization",
        "description": "Inserted initial questions into the MongoDB collection."
    }
    logs_db.insert_log(log_data)
    logging.info("Inserted log entry for database initialization.")
    
if __name__ == "__main__":
    insert_initial_questions()
    insert_log_entry()
