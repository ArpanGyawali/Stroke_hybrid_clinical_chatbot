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
    llm_model1: str = Field("llama-3.1-8b-instruct", env="LLM_MODEL")
    llm_model2: str = Field("Qwen2.5-Coder-7B-Instruct", env="LLM_MODEL")
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
    # clinical_synonyms: dict = {
    #     "symptom start": ["onset time", "onset date", "symptom onset"],
    #     "patient death": ["patient died", "date of death", "death causes"],
    #     "stroke type": ["stroke mechanism", "stroke type"],
    #     "hypodense": ["low density", "hypodensity", "decreased attenuation"],
    #     "blood pressure": ["systolic", "diastolic", "bp", "blood pressure"],
    #     "glucose": ["blood glucose", "glucose level", "bg"],
    # }
    clinical_synonyms: dict = {
        "Patient": ["patient id", "INSPIRE ID"],

        "Onset Age": ["age", "patient age", "years old"],
        "Gender": ["sex", "male", "female"],
        "Onset Date/Time": ["stroke onset", "symptom start time", "time symptoms began"],
        "Onset Date": ["stroke onset date", "date of onset"],
        "Onset Time": ["stroke onset time", "time of onset"],
        "Date Last Seen Well": ["last known well date", "lkw date"],
        "Time Last Seen Well": ["last known well time", "lkw time"],
        "Stroke Type": ["stroke", "type of stroke", "ischaemic stroke", "hemorrhagic stroke"],
        "Stroke Mechanism": ["stroke cause", "etiology", "stroke aetiology", "stroke origin"],
        "Occlusion Sites": ["blocked vessel", "artery blockage site", "clot location"],
        
        "Risk Factors_Smoking": ["smoker", "smoking history", "tobacco use"],
        "Risk Factors_Hypertension": ["high blood pressure", "hypertensive"],
        "Risk Factors_Atrial fibrillation": ["af", "a-fib", "irregular heartbeat"],
        "Risk Factors_Hypercholestermia": ["high cholesterol", "lipids", "cholesterol level"],
        "Risk Factors_Diabetes": ["diabetes mellitus", "high blood sugar", "hyperglycemia"],
        "Risk Factors_Previous diagnosis of TIA": ["history of tia", "mini stroke"],
        "Risk Factors_Previous diagnosis of stroke": ["past stroke", "stroke history"],
        "Risk Factors_Congestive heart failure": ["chf", "heart failure"],
        "Risk Factors_Ischaemic Heart Disease": ["ihd", "coronary artery disease", "cad", "heart disease"],
        
        "Drugs Taken_Aspirin": ["aspirin use", "asa"],
        "Drugs Taken_Clopidogrel": ["plavix", "clopidogrel use"],
        "Drugs Taken_Warfarin": ["coumadin", "warfarin use", "blood thinner"],
        "Drugs Taken_Statin": ["cholesterol medication", "statin therapy"],
        
        "Baseline Blood Glucose (mmol/L)": ["blood sugar", "glucose", "fasting sugar"],
        "Baseline Systolic Blood Pressure (mmHg)": ["sbp", "systolic bp", "upper blood pressure"],
        "Baseline Diastolic Blood Pressure (mmHg)": ["dbp", "diastolic bp", "lower blood pressure"],
        "Baseline total cholesterol level": ["cholesterol", "lipid profile", "total cholesterol"],
        "Baseline Cardiac Rhythm": ["heart rhythm", "ecg result", "cardiac rhythm"],
        "Baseline Body Temperature (ºC)": ["temperature", "fever", "body temp"],
        "Baseline Body Weight (kg)": ["weight", "body weight", "mass"],
        "Baseline NIHSS": ["nih stroke scale", "nihss", "stroke severity score"],
        "Pre-stroke mRS": ["modified rankin score", "mrs before stroke", "baseline mrs"],
        
        "Need for IV tPA": ["tpa given", "alteplase", "clot-busting drug", "thrombolysis"],
        "IV Treatment_Medication": ["iv drug given", "iv medication", "iv thrombolysis drug"],
        "IA Treatment_IA Treatment Delivered": ["mechanical thrombectomy", "endovascular treatment", "clot retrieval"],
        "IA Treatment_Date of groin puncture": ["groin puncture date", "arterial access date"],
        "IA Treatment_TICI Score": ["tici", "reperfusion score", "angiographic outcome"],
        
        "CTP Used": ["ct perfusion", "ct perfusion scan"],
        "Ischemic Core Volume (ml)": ["core volume", "infarct core"],
        "Penumbra Volume (ml)": ["penumbra", "penumbra size", "tissue at risk"],
        
        "Modified Rankin Score at discharge": ["mrs at discharge", "functional outcome", "rankin score"],
        "NIHSS at discharge": ["nihss discharge", "stroke severity at discharge"],
        "3 Months - mRS": ["3 month mrs", "90 day mrs", "functional outcome 3 months"],
        "12 Months - mRS": ["12 month mrs", "1 year mrs"],
        "Patient died before discharge": ["in-hospital death", "death before discharge"],
        "Date of Death": ["death date", "dod"],
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