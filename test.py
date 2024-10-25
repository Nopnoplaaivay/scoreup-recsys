# import matplotlib.pyplot as plt
# import pandas as pd
# from sklearn.manifold import TSNE

# from src.db.questions import QuestionsCollection
# from src.models.clustering import QuestionClustering   
# from src.utils.encode_utils import EncodeQuestionsUtils

# questions_collection = QuestionsCollection()
# questions = questions_collection.fetch_all_questions()

# if __name__ == "__main__":

#     df = questions_collection.preprocess_questions(raw_questions=questions)

#     encoder = EncodeQuestionsUtils()
#     X = encoder.encode(df)
#     print(X)

#     clustering_model = QuestionClustering()
#     optimal_clusters = clustering_model.find_optimal_clusters(X)
#     clustering_model.fit(X)

#     # Dự đoán nhãn cụm cho dữ liệu
#     cluster_labels = clustering_model.predict(X)
#     print("Cluster labels:", cluster_labels)
#     df['cluster'] = cluster_labels  

#     # Lấy tọa độ các tâm cụm
#     cluster_centers = clustering_model.get_cluster_centers()
#     print("Cluster centers:", cluster_centers)

#     # Kiểm tra kích thước của X và cluster_centers
#     print("Shape of X:", X.shape)
#     print("Shape of cluster_centers:", cluster_centers.shape)

#     print(df[['question_id', 'chapter', 'difficulty', 'concept', 'cluster']])
import json
from src.modules.questions_map import QuestionsMap

question_map = QuestionsMap().get_map()
print(json.dumps(question_map, indent=4))