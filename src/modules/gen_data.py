import pandas as pd
import os
import json

from src.db.logs import LogsDB
from src.modules.rec_score import RecScore
from src.modules.questions_map import QuestionsMap
from src.utils.logger import LOGGER

class YData:
    def __init__(self, notion_database_id="c3a788eb31f1471f9734157e9516f9b6"):
        self.questions_map = QuestionsMap().get_map(notion_database_id=notion_database_id)[0]
        self.logs_collection = LogsDB()
        self.logs_df = self.logs_collection.preprocess_logs(raw_logs=self.logs_collection.fetch_all_logs())
        self.prepare_logs_df()

    def prepare_logs_df(self):
        LOGGER.info("Updating logs_df with cluster column.")
        self.logs_df['cluster'] = self.logs_df['question_id'].map(self.questions_map)
        self.logs_df.dropna(subset=['cluster'], inplace=True)
        self.logs_df['cluster'] = self.logs_df['cluster'].astype(int)

    def gen_Y_data(self):
        """Generate the Y matrix."""
        cols = ['user_id', 'cluster', 'rec_score']
        Y = pd.DataFrame(columns=cols)
        group_user_cluster = self.logs_df.groupby(['user_id', 'cluster'])

        rows = []   
        for (user_id, cluster), group_df in group_user_cluster:
            rec_score = RecScore(group_df=group_df)
            rows.append([user_id, cluster, rec_score.total_rec_score])

        if rows:
            Y = pd.concat([Y, pd.DataFrame(rows, columns=cols)], ignore_index=True)
            
            user_ids = Y['user_id'].unique()
            num_to_user_map = {idx: user_id for idx, user_id in enumerate(user_ids)}
            user_to_num_map = {user_id: idx for idx, user_id in enumerate(user_ids)}
 
            with open("src/data/user_map.json", "w") as json_file:
                json.dump(num_to_user_map, json_file, indent=4)

            Y['user_id'] = Y['user_id'].map(user_to_num_map)

            os.makedirs("src/data/", exist_ok=True)
            Y.to_csv("src/data/y_data.csv", index=False)