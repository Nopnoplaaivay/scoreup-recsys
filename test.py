from src.db.questions import QuestionsCollection
from src.models.clustering import QuestionClustering    

questions_collection = QuestionsCollection()
questions = questions_collection.fetch_all_questions()
print(type(questions))

data = questions_collection.preprocess_questions(raw_questions=questions)
print(data)