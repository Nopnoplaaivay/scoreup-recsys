import pandas as pd

from src.db.questions import QuestionsCollection
from src.models.clustering import QuestionClustering   
from src.models.preprocess import PreprocessX
# from src.utils.encode_utils import DataEncoder

questions_collection = QuestionsCollection()
questions = questions_collection.fetch_all_questions()

if __name__ == "__main__":
    df = questions_collection.preprocess_questions(raw_questions=questions)

    preprocess = PreprocessX()
    preprocess.encode(df)
    X = preprocess.df


    # data = pd.read_pickle('embeddings.pkl')
    # print(data.columns)

    # encoder = DataEncoder(df)
    # X = encoder.prepare_X()

    clustering_model = QuestionClustering()

    # # Tìm số cụm tối ưu
    optimal_clusters = clustering_model.find_optimal_clusters(X)

    # # Huấn luyện mô hình với số cụm tối ưu
    clustering_model.fit(X)

    # Dự đoán nhãn cụm cho dữ liệu
    cluster_labels = clustering_model.predict(X)
    print("Cluster labels:", cluster_labels)

    # Lấy tọa độ các tâm cụm
    cluster_centers = clustering_model.get_cluster_centers()
    print("Cluster centers:", cluster_centers)