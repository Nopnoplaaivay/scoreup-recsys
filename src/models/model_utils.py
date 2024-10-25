import pandas as pd
import numpy as np

def preprocess_questions(data):
    """
    Preprocess the questions dataset by extracting the required features for clustering.
    
    Args:
    data (pd.DataFrame): A DataFrame with question details.
    
    Returns:
    np.array: A matrix ready for clustering (numerical features only).
    """
    # Here we assume that 'chapter' and 'difficulty' are relevant features for clustering.
    # You may add more features as needed (e.g., question length, tags, etc.).
    chapter_encoded = pd.get_dummies(data['chapter'], drop_first=True)
    X = np.hstack([chapter_encoded, data[['difficulty']].values])
    return X

def evaluate_clustering(labels, X):
    """Evaluate the clustering performance with silhouette score."""
    from sklearn.metrics import silhouette_score
    return silhouette_score(X, labels)

def evaluate_collaborative_filtering(model, testset):
    """
    Evaluate collaborative filtering model using RMSE.
    
    Args:
    model: The collaborative filtering model (e.g., SVD).
    testset: The testset to evaluate on.
    
    Returns:
    float: RMSE score.
    """
    from surprise.accuracy import rmse
    predictions = model.test(testset)
    return rmse(predictions)
