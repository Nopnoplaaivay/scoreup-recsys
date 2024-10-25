from db.connection import MongoDBConnection
import logging

logging.basicConfig(filename='logs/db.log', level=logging.INFO)

class LogsDB:
    def __init__(self):
        self.connection = MongoDBConnection()
        self.collection_name = "logs"

    def insert_log(self, log_data):
        """Insert a log entry into the MongoDB collection."""
        try:
            self.connection.connect()
            db = self.connection.get_database()
            collection = db[self.collection_name]
            
            result = collection.insert_one(log_data)
            logging.info(f"Inserted log with ID: {result.inserted_id}")
            return result.inserted_id
        except Exception as e:
            logging.error(f"Error inserting log: {e}")
            raise e
        finally:
            self.connection.close()

    def fetch_logs(self, filter_criteria=None):
        """Fetch logs from the MongoDB collection based on filter criteria."""
        try:
            self.connection.connect()
            db = self.connection.get_database()
            collection = db[self.collection_name]
            
            logs = list(collection.find(filter_criteria or {}))
            logging.info(f"Fetched {len(logs)} logs from the database.")
            return logs
        except Exception as e:
            logging.error(f"Error fetching logs: {e}")
            raise e
        finally:
            self.connection.close()
