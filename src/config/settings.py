"""Configuration settings for the clinical chatbot."""

import os
from pathlib import Path
from typing import List, Optional

from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Application settings."""
    
    # API Keys
    openai_api_key: Optional[str] = Field(None, env="OPENAI_API_KEY")
    langchain_api_key: Optional[str] = Field(None, env="LANGCHAIN_API_KEY")
    weaviate_url: str = Field("https://your-cluster.weaviate.network", env="WEAVIATE_URL")
    weaviate_api_key: Optional[str] = Field(None, env="WEAVIATE_API_KEY")
    huggingface_api_key: Optional[str] = Field(None, env="HUGGINGFACE_API_KEY")
    # LangSmith Configuration
    langchain_tracing_v2: bool = Field(True, env="LANGCHAIN_TRACING_V2")
    langchain_project: str = Field("clinical-chatbot", env="LANGCHAIN_PROJECT")
    langchain_endpoint: str = Field(None, env="LANGCHAIN_ENDPOINT")
    # Model Configuration
    # llm_model: str = Field("mistral:7b", env="LLM_MODEL")  # For Ollama
    llm_model: str = Field("llama-3.1-8b-instruct", env="LLM_MODEL")
    embedding_model: str = Field("sentence-transformers/all-MiniLM-L6-v2", env="EMBEDDING_MODEL")
    temperature: float = Field(0.1, env="TEMPERATURE")
    max_tokens: int = Field(1000, env="MAX_TOKENS")
    
    # Data Paths
    project_root: Path = Path(__file__).parent.parent.parent
    model_dir: Path = project_root / "models"
    data_dir: Path = project_root / "data"
    structured_data_dir: Path = data_dir / "structured"
    unstructured_data_dir: Path = data_dir / "unstructured"
    
    # Vector Store Configuration
    collection_name: str = Field("clinical_Stroke_documents", env="COLLECTION_NAME")
    # chunk_size: int = Field(1000, env="CHUNK_SIZE")
    # chunk_overlap: int = Field(200, env="CHUNK_OVERLAP")
    
    # Agent Configuration
    max_iterations: int = Field(5, env="MAX_ITERATIONS")
    memory_buffer_size: int = Field(10, env="MEMORY_BUFFER_SIZE")
    
    # Clinical Domain Configuration
    clinical_synonyms: dict = {
        "symptom start": ["onset time", "onset date", "symptom onset"],
        "patient death": ["patient died", "date of death", "death causes"],
        "stroke type": ["stroke mechanism", "stroke type"],
        "hypodense": ["low density", "hypodensity", "decreased attenuation"],
        "blood pressure": ["systolic", "diastolic", "bp", "blood pressure"],
        "glucose": ["blood glucose", "glucose level", "bg"],
    }
    
    # Column mappings for structured data
    column_mappings: dict = {
        "patient_id": ["INSPIRE ID", "Site", "Study ID"],
        "demographics": ["Onset Age", "Gender"],
        "timing": ["Onset Date", "Date Last Seen Well", 
                  "Onset Time", "Time Last Seen Well"],
        "clinical_measures": ["Baseline NIHSS", "Baseline Blood Glucose (mmol/L)",
                            "Baseline Systolic Blood Pressure (mmHg)",
                            "Baseline Diastolic Blood Pressure (mmHg)"],
        "outcomes": ["Modified Rankin Score at discharge", "Patient died before discharge",
                   "Date of Death", "Death Causes"],
        "stroke_details": ["Stroke Type", "Stroke Mechanism", "Occlusion Sites"],
    }

    class Config:
        env_file = ".env"
        case_sensitive = False


# Global settings instance
settings = Settings()