import os
import chromadb
import logging

# ----------------------------
# Logging Setup
# ----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)

class DatabaseChromaDB:
    """
    Query and insert data from/to the vector database ChromaDB
    """
    def __init__(self,  
                 collection_name:str = "PubMed_research_papers"):
        
        self.collection_name = collection_name

        logging.info(f"Connecting to ChromaDB...")
        # Connect to ChromaDB
        self.client = chromadb.Client()
        logging.info("Connection to ChromaDB done.")
    
    def insert(self, data:dict) -> None:
        """
        Inserting data into chrombaDB

        Parameters:
            data (dict): input data
        
        Returns:
            None
        """
        logging.info("Inserting data into ChromaDB...")
        # Get a collection object from an existing collection, by name. If it doesn't exist, create it.
        collection = self.client.get_or_create_collection(self.collection_name) 

        for paper in data:
            for i in collection.get()['ids']:
                if i == paper['id']:
                    logging.info(f"ID {i} already exists. Skip document...")

            collection.add(
                documents=[paper['abstract']],  
                embeddings=[paper['embedding']],  
                metadatas=[paper['metadata']],  
                uris=[paper['url']],
                ids=[paper['id']]
            )
        logging.info("Inserting done.")
    
    def query_collection(self):
        """
        Querying data from the database
        
        Parameters:
            None

        Returns:
            Documents with embeddings and metadata
        """
        logging.info("Getting all data embeddings from the database...")
        collection = self.client.get_collection(self.collection_name)
        documents = collection.get(include=['embeddings', 'documents', 'uris', 'metadatas'])
        logging.info("Getting data from the database done.")
        return documents