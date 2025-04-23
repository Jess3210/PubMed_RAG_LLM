import logging
from vertexai.generative_models import GenerativeModel
import vertexai
from google.oauth2 import service_account

# ----------------------------
# Logging Setup
# ----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)

class AnswerLLM:
    """
    Answers the question based on the input query and the retrieved context data
    """
    def __init__(self, credentials:str, 
                 project_id:str, 
                 model:str = "gemini-1.5-pro-002"):
        logging.info("Initializing Vertex AI...")
        self.credentials = self.enter_credentials(credentials)
        # Initialize Vertex AI 
        vertexai.init(credentials=self.credentials, project=project_id)
        self.model = GenerativeModel(model_name=model)
        logging.info("Initializing Vertex AI done.")
    
    def enter_credentials(self, credentials_path:str):
        """
        Enter google cloud credentials

        Parameters:
            credentials_path (str): path to credentials of project
        
        Returns:
            None
        """
        credentials = service_account.Credentials.from_service_account_file(
                    credentials_path,
                    scopes=["https://www.googleapis.com/auth/cloud-platform"],
                )
        return credentials

    def generate_answer(self, query:str, context:str) -> str:
        """
        Generate Answer based on input query and the retrieved context data

        Parameters:
            query (str): input question
            context (str): retrieved context as abstract of papers
        
        Returns:
            response text
        """
        logging.info("Generating answer to the question...")
        prompt = f"Please answer the following question based on the context. Question: {query}, Context: {context}."
        response = self.model.generate_content(prompt)
        logging.info("Answer generation done.")
        return response.text