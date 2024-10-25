from src.db.questions import QuestionsCollection
from src.models.clustering import QuestionClustering    
from src.utils.encode_utils import EncodeQuestionsUtils

questions_collection = QuestionsCollection()
questions = questions_collection.fetch_all_questions()

data = questions_collection.preprocess_questions(raw_questions=questions)

encode_utils = EncodeQuestionsUtils()
print(encode_utils.encode(df=data))