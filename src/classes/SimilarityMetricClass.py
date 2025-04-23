import logging
import numpy as np

# ----------------------------
# Logging Setup
# ----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)

class SimilarityMetric:
    """
    Similarity measures between vector embeddings with descending ranking of highest scores 
    and return of highest score with corresponding index
    """
    def __init__(self,
                 query: np.ndarray,
                 db_text: dict):
        
        self.query = query
        self.db_text = db_text
        self.db_text['similarity_score'] = []

    def cosine_similarity(self, db_text: np.ndarray) -> float:
        """
        Computes the Cosine Similarity between two vectors.

        Args:
        db_text (np.ndarray): The second embedding of abstract text in database.

        Returns:
        float: The cosine similarity value between the two vector embeddings.
        """

        logging.info("Calculating cosine similarity...")
        # Compute the dot product of the two vectors
        dot_product = np.dot(self.query, db_text)
        
        # Compute the norms (magnitudes) of the vectors
        norm_query = np.linalg.norm(self.query)
        norm_db_text = np.linalg.norm(db_text)
        
        # Compute the cosine similarity
        similarity = dot_product / (norm_query * norm_db_text)
        
        return similarity
    
    def ranking(self):
        """
        Ranking in descending order of similarity score

        Args: 
        None
        
        Returns:
        index with highest similarty score (tuple)
        """

        logging.info("Ranking of cosine similarity values...")
        for paper in self.db_text['embeddings']:
            similarity_query_text = self.cosine_similarity(paper)
            self.db_text['similarity_score'].append(similarity_query_text)
            
        # Each tuple contains the index of the list and the cosine similarity value
        indexed_similarities = [(index, value) for index, value in enumerate(self.db_text['similarity_score'])]
        # Sort the list by cosine similarity values in descending order (highest value first)
        sorted_similarities = sorted(indexed_similarities, key=lambda x: x[1], reverse=True)
        logging.info("Ranking done.")
        #return index with highest cosine similarity score
        return sorted_similarities[0]