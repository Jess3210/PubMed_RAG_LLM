import logging
import re
import pubmed_parser as pp
from typing import List
from sentence_transformers import SentenceTransformer
from datetime import datetime

# ----------------------------
# Logging Setup
# ----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)

class ExtractPubMedData:
    """
    Extracts the PubMed Ids with corresponding abstracts and metadata from input urls of PubMed papers, creates vector embeddings
    and returns the data with vector embeddings and metadata
    """
    def __init__(self, 
                 urls: List[str] = None, 
                 embedding_model:str = "all-MiniLM-L6-v2"):
        self.urls = urls
        self.paper_data = []
        
        self.embedding_model = embedding_model

    def extract_pubmed_id(self, 
                          url: str, 
                          pm_id_pattern: str = r'pubmed\.ncbi\.nlm\.nih\.gov/(\d+)/?') -> str | None:
        """
        Extracts a PubMed ID (PMID) from a given PubMed URL.

        Parameters:
            url (str): The PubMed article URL.
            pm_id_pattern (str): Regex pattern to extract the ID.

        Returns:
            str | None: The extracted PMID or None if not found or error occurs.
        """
        try:
            if not url:
                raise ValueError("Empty or invalid URL.")
            pattern = re.compile(pm_id_pattern)
            match = pattern.search(url)
            if match:
                return match.group(1)
            else:
                logging.warning(f"Warning: No PMID found in URL '{url}'")
                return None
        except Exception as e:
            logging.error(f"Error while extracting PMID from '{url}': {e}")
            return None
    
    def embedding(self, data:dict) -> dict:
        """
        Embedding of input text data

        Parameters:
            data (dict): data as dictionary
        
        Returns:
            data (dict): data with embedding as another key
        """
        #embedding leng: 384
        logging.info("Start embedding of text data...")
        model = SentenceTransformer(self.embedding_model)
        embeddings = model.encode(data["abstract"]).tolist()
        data["embedding"] = embeddings
        logging.info("Text data embedding done...")
        return data
    
    def transform_data_structure(self, 
                                 data:dict, 
                                 main_keys:List[str] = ['embedding', 'abstract', 'pmid']):
        """
        Transform data into the final data structure with metadata

        Parameters:
            data (dict): input data
            main_keys (List[str]): main keys for chromaDB input, rest of the keys will be under metadata
        
        Returns:
            None
        """
        metadata = {key: str(data[key]) for key in data if key not in main_keys}

        return {'embedding' : data[main_keys[0]],
                'abstract' : data[main_keys[1]],
                'id' : data[main_keys[2]],
                'metadata' : metadata}

    def extract_pubmed_data(self):
        """
        Extracts abstract and metadata for each URL in self.urls and stores it in self.paper_data.

        Uses the pubmed-parser library to fetch data from PubMed using the extracted PMIDs.
        Skips any URLs that do not yield a valid PMID or if fetching fails.
        """
        for url in self.urls:
            pmid = self.extract_pubmed_id(url)
            if pmid:
                try:
                    logging.info("Extracting data from url...")
                    paper = pp.parse_xml_web(pmid)
                    data_with_embedding = self.embedding(paper)
                    #Create time stamp in format: YYYY-MM-DD
                    data_with_embedding['created'] = datetime.now().strftime("%Y-%m-%d")  
                    data_correct_structure = self.transform_data_structure(data_with_embedding)
                    #Add url as metadata
                    data_correct_structure['url'] = url
                    self.paper_data.append(data_correct_structure)
                    
                except Exception as e:
                    logging.error(f"Error retrieving data for PMID {pmid}: {e}")
            else:
                logging.warning(f"Skipping URL due to missing or invalid PMID: {url}")