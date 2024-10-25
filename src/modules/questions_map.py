from src.db.questions import QuestionsCollection
from src.models.clustering import QuestionClustering   
from src.utils.encode_utils import EncodeQuestionsUtils

class QuestionsMap:
    def __init__(self):
        
        self.encoder = EncodeQuestionsUtils()
        self.clustering_model = QuestionClustering()

    def get_map(self, notion_database_id="c3a788eb31f1471f9734157e9516f9b6"):
        questions_collection = QuestionsCollection(notion_database_id=notion_database_id)
        raw_questions = questions_collection.fetch_all_questions()
        
        questions_df = questions_collection.preprocess_questions(raw_questions=raw_questions)
        
        # Prepare X
        X = self.encoder.encode(questions_df)
        self.clustering_model.get_optimal_clusters(X)
        self.clustering_model.fit(X)

        # Predict cluster
        cluster_labels = self.clustering_model.predict(X)
        questions_df['cluster'] = cluster_labels

        cluster_map = questions_df.groupby('cluster')['question_id'].apply(list).to_dict()
        return cluster_map