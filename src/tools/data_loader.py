"""Data loading utilities for structured and unstructured data."""

import logging, sys
from pathlib import Path
from typing import Dict, List, Optional, Any
import pandas as pd
import weaviate
from weaviate.classes.config import Configure, Property, DataType
from langchain.document_loaders import PyPDFLoader, TextLoader, UnstructuredPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document
from langchain_experimental.text_splitter import SemanticChunker


from ..config.settings import settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),                 # Console output
        logging.FileHandler("app.log", mode='a', encoding='utf-8')  # File output
    ]
)
logger = logging.getLogger(__name__)


class DataLoader:
    """Handles loading and preprocessing of structured and unstructured data."""
    
    def __init__(self):
        self.structured_data: Optional[pd.DataFrame] = None
        self.weaviate_client: Optional[weaviate.Client] = None
        self.embeddings = HuggingFaceEmbeddings(
            model_name=settings.embedding_model,
            model_kwargs={'device': 'cpu'}
        )
        self.collection_already_populated = False  # Default until checked
        self._initialize_weaviate()
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        self.close()
    
    def __del__(self):
        """Destructor to ensure cleanup."""
        self.close()
    
    def _initialize_weaviate(self) -> None:
        """Initialize Weaviate client and create collection if needed."""
        try:
            headers = {}
            if getattr(settings, 'huggingface_api_key', None):
                headers["X-HuggingFace-Api-Key"] = settings.huggingface_api_key

            self.weaviate_client = weaviate.connect_to_weaviate_cloud(
                cluster_url=settings.weaviate_url,
                auth_credentials=weaviate.auth.AuthApiKey(settings.weaviate_api_key),
                headers=headers if headers else None
            )

            if not self.weaviate_client.is_ready():
                logger.warning("Weaviate client is not ready, waiting...")

            collections = self.weaviate_client.collections
            if collections.exists(settings.collection_name):
                collection = collections.get(settings.collection_name)

                # Check if collection already has data
                count_result = collection.aggregate.over_all(total_count=True)
                total_count = getattr(count_result, "total_count", 0)

                if total_count and total_count > 0:
                    logger.info(f"Collection '{settings.collection_name}' already exists with {total_count} objects. Skipping data load.")
                    self.collection_already_populated = True
                    return
                else:
                    logger.info(f"Collection '{settings.collection_name}' exists but is empty. Will load data.")
                    self.collection_already_populated = False
            else:
                logger.info(f"Collection '{settings.collection_name}' does not exist. Creating it.")
                vectorizer_config = Configure.Vectorizer.text2vec_huggingface(
                    model=settings.embedding_model
                )
                collections.create(
                    name=settings.collection_name,
                    vectorizer_config=vectorizer_config,
                    properties=[
                        Property(name="content", data_type=DataType.TEXT, description="Document content"),
                        Property(name="source", data_type=DataType.TEXT, description="Document source path"),
                        Property(name="document_type", data_type=DataType.TEXT, description="Type of document (pdf, text, etc.)"),
                    ]
                )
                self.collection_already_populated = False

        except Exception as e:
            logger.error(f"Failed to initialize Weaviate: {e}")
            raise

    
    def load_structured_data(self, file_path: Optional[Path] = None) -> pd.DataFrame:
        """Load structured data from CSV/Excel files"""
        if file_path is None:
            # Search for Excel and CSV files
            # excel_files = list(settings.structured_data_dir.glob("*.xlsx")) + list(settings.structured_data_dir.glob("*.xls"))
            csv_files = list(settings.structured_data_dir.glob("*.csv"))
            
            all_files =  csv_files
            
            if not all_files:
                raise FileNotFoundError("No Excel or CSV files found in structured data directory")
            
            file_path = all_files[0]

        try:
            if file_path.suffix.lower() == '.csv':
                self.structured_data = pd.read_csv(file_path, low_memory=False)
            # elif file_path.suffix.lower() in ['.xlsx', '.xls']:
            #     self.structured_data = pd.read_excel(file_path)
            else:
                raise ValueError(f"Unsupported file format: {file_path.suffix}")

            # Clean column names
            self.structured_data.columns = self.structured_data.columns.str.strip()

            logger.info(f"Loaded structured data: {self.structured_data.shape}")
            return self.structured_data

        except Exception as e:
            logger.error(f"Failed to load structured data from {file_path}: {e}")
            raise
    
    def load_unstructured_data(self) -> List[Document]:
        """Load and process unstructured documents."""
        documents = []
        
        # Process PDF files
        pdf_files = list(settings.unstructured_data_dir.rglob("*.pdf"))
        for pdf_file in pdf_files:
            try:
                loader = PyPDFLoader(str(pdf_file))
                pdf_docs = loader.load()
                for doc in pdf_docs:
                    # Ensure content is not empty
                    if doc.page_content.strip():
                        doc.metadata.update({
                            "source": str(pdf_file),
                            "document_type": "pdf"
                        })
                        documents.append(doc)
                logger.info(f"Loaded PDF: {pdf_file}")
            except Exception as e:
                logger.error(f"Failed to load PDF {pdf_file}: {e}")
        
        # Process text files
        txt_files = list(settings.unstructured_data_dir.rglob("*.txt"))
        for txt_file in txt_files:
            try:
                loader = TextLoader(str(txt_file))
                txt_docs = loader.load()
                for doc in txt_docs:
                    # Ensure content is not empty
                    if doc.page_content.strip():
                        doc.metadata.update({
                            "source": str(txt_file),
                            "document_type": "text"
                        })
                        documents.append(doc)
                logger.info(f"Loaded text file: {txt_file}")
            except Exception as e:
                logger.error(f"Failed to load text file {txt_file}: {e}")
        
        if not documents:
            logger.warning("No documents loaded from unstructured data directory")
            return []
        
        # Split documents into chunks
        # text_splitter = RecursiveCharacterTextSplitter(
        #     chunk_size=settings.chunk_size,
        #     chunk_overlap=settings.chunk_overlap,
        #     separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
        # )
        text_splitter = SemanticChunker(
            self.embeddings, 
            breakpoint_threshold_type="percentile", 
            breakpoint_threshold_amount=90
        )
        
        chunked_documents = text_splitter.split_documents(documents)
        # Filter out empty chunks
        chunked_documents = [doc for doc in chunked_documents if doc.page_content.strip()]
        
        logger.info(f"Created {len(chunked_documents)} document chunks")
        return chunked_documents
    
    def index_documents(self, documents: List[Document]) -> None:
        """Index documents in Weaviate."""
        if not self.weaviate_client:
            raise RuntimeError("Weaviate client not initialized")
        
        if not documents:
            logger.warning("No documents to index")
            return
        
        if self.collection_already_populated is True:
            logger.info("Collection already populated. Skipping indexing.")
            return
        
        collection = self.weaviate_client.collections.get(settings.collection_name)
        
        try:
            # First, let's test if the collection is accessible
            logger.info(f"Testing collection access for: {settings.collection_name}")
            
            # Clear existing data if needed (optional - remove if you want to keep existing data)
            # collection.data.delete_many(where=weaviate.classes.query.Filter.by_property("document_type").exists())
            
            # Prepare data for batch import with validation
            data_objects = []
            for i, doc in enumerate(documents):
                content = doc.page_content.strip()
                if not content:
                    logger.warning(f"Skipping empty document at index {i}")
                    continue
                    
                # Truncate very long content to avoid issues
                if len(content) > 8000:  # Reasonable limit for most embedding models
                    content = content[:8000] + "..."
                    
                data_objects.append({
                    "content": content,
                    "source": str(doc.metadata.get("source", "unknown")),
                    "document_type": doc.metadata.get("document_type", "unknown")
                })
            
            if not data_objects:
                logger.warning("No valid data objects to index after filtering")
                return
            
            logger.info(f"Attempting to index {len(data_objects)} documents")
            
            # Use smaller batch sizes to avoid timeout issues
            batch_size = 10
            successful_inserts = 0
            
            for i in range(0, len(data_objects), batch_size):
                batch = data_objects[i:i+batch_size]
                try:
                    with collection.batch.fixed_size(batch_size=len(batch)) as batch_context:
                        for obj in batch:
                            batch_context.add_object(obj)
                    
                    successful_inserts += len(batch)
                    logger.info(f"Successfully indexed batch {i//batch_size + 1}/{(len(data_objects)-1)//batch_size + 1}")
                    
                except Exception as batch_error:
                    logger.error(f"Failed to index batch {i//batch_size + 1}: {batch_error}")
                    
                    # Try to get failed objects info
                    if hasattr(collection.batch, 'failed_objects'):
                        failed_objects = collection.batch.failed_objects
                        if failed_objects:
                            logger.error(f"Failed objects details: {failed_objects[:5]}")  # Show first 5
                    
                    # Try inserting objects one by one for this batch
                    for j, obj in enumerate(batch):
                        try:
                            collection.data.insert(obj)
                            successful_inserts += 1
                            logger.debug(f"Individual insert successful for object {i+j}")
                        except Exception as individual_error:
                            logger.error(f"Individual insert failed for object {i+j}: {individual_error}")
            
            logger.info(f"Successfully indexed {successful_inserts} out of {len(data_objects)} documents in Weaviate")
            
        except Exception as e:
            logger.error(f"Failed to index documents: {e}")
            # Log more details about the error
            if hasattr(e, 'response'):
                logger.error(f"Response details: {e.response}")
            raise
    
    def get_structured_data_info(self) -> Dict[str, Any]:
        """Get information about the structured data."""
        if self.structured_data is None:
            return {}
        
        return {
            "shape": self.structured_data.shape,
            "columns": list(self.structured_data.columns),
            "dtypes": self.structured_data.dtypes.to_dict(),
            "null_counts": self.structured_data.isnull().sum().to_dict(),
            "sample_values": {
                col: self.structured_data[col].dropna().head(3).tolist()
                for col in self.structured_data.columns[:10]  # Limit to first 10 columns
            }
        }
    
    def search_similar_documents(self, query: str, limit: int = 5) -> List[Dict]:
        """Search for similar documents in Weaviate."""
        if not self.weaviate_client:
            raise RuntimeError("Weaviate client not initialized")
        
        collection = self.weaviate_client.collections.get(settings.collection_name)
        
        try:
            response = collection.query.near_text(
                query=query,
                limit=limit,
                return_metadata=weaviate.classes.query.MetadataQuery(distance=True)
            )
            
            results = []
            for obj in response.objects:
                results.append({
                    "content": obj.properties.get("content", ""),
                    "source": obj.properties.get("source", "unknown"),
                    "document_type": obj.properties.get("document_type", "unknown"),
                    "distance": obj.metadata.distance if obj.metadata else None
                })
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to search documents: {e}")
            return []
    
    def close(self) -> None:
        """Close connections."""
        if self.weaviate_client:
            try:
                self.weaviate_client.close()
                logger.debug("Weaviate connection closed")
            except Exception as e:
                logger.warning(f"Error closing Weaviate connection: {e}")
            finally:
                self.weaviate_client = None