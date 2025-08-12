"""Main application entry point for the clinical chatbot."""

import os
import logging
import asyncio
import uuid
from typing import Dict, Any, Optional
from pathlib import Path
# import openai

import streamlit as st
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_community.llms import Ollama, HuggingFacePipeline
from langsmith import Client as LangSmithClient

from .config.settings import settings
from .agents.primary_agent import PrimaryAgent
from .tools.data_loader import DataLoader

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Disable LangSmith to avoid warnings
os.environ["LANGCHAIN_TRACING_V2"] = "false"

# class OpenAILLM:
#     def __init__(self, api_key, model, max_tokens=256, temperature=0.1, top_p=0.9):
#         openai.api_key = api_key
#         self.model = model
#         self.max_tokens = max_tokens
#         self.temperature = temperature
#         self.top_p = top_p

#     def generate(self, prompt):
#         response = openai.ChatCompletion.create(
#             model=self.model,
#             messages=[{"role": "user", "content": prompt}],
#             max_tokens=self.max_tokens,
#             temperature=self.temperature,
#             top_p=self.top_p,
#         )
#         return response.choices[0].message.content

#     def __call__(self, prompt):
#         return self.generate(prompt)

@st.cache_resource
def initialize_llm():
    """Initialize the language model (cached across reruns)."""
    try:
        logger.info("Initializing LLM...")

        
        # Initialize the Hugging Face LLaMA-3.1 model.
        model_path = settings.model_dir / settings.llm_model
        tokenizer = AutoTokenizer.from_pretrained(str(model_path))

        model = AutoModelForCausalLM.from_pretrained(
            str(model_path),
            torch_dtype=torch.float16,
            device_map="auto",
            load_in_4bit=True   # Saves VRAM, optional
        )

        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=settings.max_tokens,
            temperature=settings.temperature,
            top_p=0.9
        )
        pipe2 = pipeline(
            "question-answering",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=settings.max_tokens,
            temperature=settings.temperature,
            top_p=0.9
        )

        llm = HuggingFacePipeline(pipeline=pipe)
        llm2 = HuggingFacePipeline(pipeline=pipe2)
        # llm = OpenAILLM(
        #     api_key=settings.openai_api_key,
        #     model=settings.llm_model,
        #     max_tokens=settings.max_tokens,
        #     temperature=settings.temperature,
        #     top_p=0.9,
        # )
        logger.info(f"LLM initialized successfully: {settings.llm_model}")
        return llm, llm2
        
    except Exception as e:
        logger.error(f"Failed to initialize LLM: {e}")
        st.error(f"Failed to initialize language model: {e}")
        st.stop()

@st.cache_resource
def _initialize_langsmith() -> None:
        """Initialize LangSmith for monitoring."""
        
        try:
            logger.info("Initializing langsmith...")
            if settings.langchain_api_key:
                os.environ["LANGCHAIN_TRACING_V2"] = "true"
                os.environ["LANGCHAIN_PROJECT"] = settings.langchain_project
                os.environ["LANGCHAIN_API_KEY"] = settings.langchain_api_key
                os.environ["LANGCHAIN_ENDPOINT"] = settings.langchain_endpoint
                logger.info("LangSmith monitoring initialized")
            else:
                logger.warning("LangSmith API key not provided - monitoring disabled")
        except Exception as e:
            logger.warning(f"Failed to initialize LangSmith: {e}")

@st.cache_resource
def get_primary_agent(_llm1, _llm2, session_id: str):
    """Get or create primary agent (cached across reruns)."""
    try:
        logger.info(f"Creating primary agent for session: {session_id}")
        agent = PrimaryAgent(_llm1, _llm2, session_id)
        return agent
    except Exception as e:
        logger.error(f"Failed to initialize agent: {e}")
        st.error(f"Failed to initialize agent: {e}")
        st.stop()


def initialize_session_state():
    """Initialize session state variables."""
    if 'session_id' not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    if 'query_input' not in st.session_state:
        st.session_state.query_input = ""


def process_query_sync(agent, query: str) -> Dict[str, Any]:
    """Process query synchronously (wrapper for Streamlit)."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(agent.process_query(query))
    except Exception as e:
        logger.error(f"Query processing failed: {e}")
        return {
            "query": query,
            "answer": f"I encountered an error processing your query: {str(e)}",
            "error": str(e),
            "session_id": st.session_state.session_id
        }
    finally:
        loop.close()


def run_streamlit_app():
    """Run the Streamlit web application."""
    st.set_page_config(
        page_title="Clinical Data Analysis Chatbot",
        page_icon="🏥",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize session state
    initialize_session_state()
    
    st.title("🏥 Clinical Data Analysis Chatbot")
    st.markdown("""
    Welcome to the Clinical Data Analysis Chatbot! I can help you analyze clinical data 
    and answer questions about patient records, medical conditions, and research findings.
    """)
    
    # Initialize LLM (cached)
    with st.spinner("Loading language model..."):
        llm1, llm2 = initialize_llm()
        _initialize_langsmith()
    
    # Get agent (cached per session)
    agent = get_primary_agent(llm1, llm2, st.session_state.session_id)
    
    # Sidebar
    with st.sidebar:
        st.header("Session Information")
        
        # New session button
        if st.button("New Session"):
            # Clear cache for the current session
            st.cache_resource.clear()
            # Reset session state
            st.session_state.session_id = str(uuid.uuid4())
            st.session_state.chat_history = []
            st.session_state.query_input = ""
            st.rerun()
        
        st.write(f"**Session ID:** `{st.session_state.session_id[:8]}...`")
        
        # Show session info
        try:
            session_info = agent.get_session_info()
            
            st.subheader("Data Status")
            st.write(f"✅ Structured Data: {'Loaded' if session_info['data_loaded']['structured'] else 'Not loaded'}")
            st.write(f"✅ Unstructured Data: {'Loaded' if session_info['data_loaded']['unstructured'] else 'Not loaded'}")
            
            st.subheader("Available Tools")
            for tool_name in session_info['tools_available']:
                st.write(f"• {tool_name.replace('_', ' ').title()}")
            
            # Memory stats
            memory_stats = session_info['memory_stats']
            if memory_stats['total_exchanges'] > 0:
                st.subheader("Conversation Stats")
                st.write(f"**Exchanges:** {memory_stats['total_exchanges']}")
                st.write(f"**Duration:** {memory_stats['session_duration']:.1f} minutes")
        
        except Exception as e:
            st.error(f"Error getting session info: {e}")
        
        # Data upload section
        st.subheader("Upload Data")
        uploaded_file = st.file_uploader(
            "Upload clinical data (CSV/Excel)",
            type=['csv', 'xlsx', 'xls'],
            help="Upload additional clinical data files"
        )
        
        if uploaded_file:
            try:
                # Save uploaded file temporarily
                temp_path = Path(f"/tmp/{uploaded_file.name}")
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.read())
                
                # Load the data
                agent.data_loader.load_structured_data(temp_path)
                st.success(f"Loaded {uploaded_file.name} successfully!")
                
            except Exception as e:
                st.error(f"Failed to load file: {e}")
    
    # Main chat interface
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Chat Interface")
        
        # Example queries
        with st.expander("Example Queries"):
            examples = [
                "How many patients are there in the sheet?",
                "How many patients died from stroke in the database?",
                "When did the symptoms start for patient INSP_AU010031?",
                "What are the ways to identify hypodense stroke?",
                "Show me patients with high blood pressure and their outcomes",
                "Explain the NIHSS scoring system",
                "What is the average age of patients with TACI stroke?"
            ]
            
            for i, example in enumerate(examples):
                if st.button(example, key=f"example_{i}"):
                    st.session_state.query_input = example
                    st.rerun()
        
        # Query input
        query = st.text_area(
            "Enter your query:",
            value=st.session_state.query_input,
            height=100,
            key="query_area"
        )
        
        col_send, col_clear = st.columns([1, 1])
        with col_send:
            send_button = st.button("Send Query", type="primary")
        with col_clear:
            if st.button("Clear Chat"):
                agent.clear_session()
                st.session_state.chat_history = []
                st.session_state.query_input = ""
                st.rerun()
        
        # Process query
        if send_button and query.strip():
            with st.spinner("Processing your query..."):
                result = process_query_sync(agent, query)
            
            # Store in session state
            st.session_state.chat_history.append({
                'query': query,
                'result': result,
                'timestamp': pd.Timestamp.now()
            })
            
            # Clear input
            st.session_state.query_input = ""
            st.rerun()
    
    with col2:
        st.subheader("Query Analysis")
        
        # Show analysis for the last query if available
        if st.session_state.chat_history:
            last_result = st.session_state.chat_history[-1]['result']
            
            if 'analysis' in last_result:
                analysis = last_result['analysis']
                
                st.write(f"**Query Type:** {analysis['query_type']}")
                st.write(f"**Confidence:** {analysis['confidence']:.2f}")
                st.write(f"**Processing Time:** {last_result.get('processing_time', 0):.2f}s")
                
                if 'tool_results' in last_result:
                    st.write("**Tools Used:**")
                    for tool_result in last_result['tool_results']:
                        st.write(f"• {tool_result['tool']}")
    
    # Chat history
    st.subheader("Chat History")
    
    if st.session_state.chat_history:
        for i, chat in enumerate(reversed(st.session_state.chat_history)):
            with st.expander(f"Query {len(st.session_state.chat_history) - i}: {chat['query'][:50]}..."):
                st.write(f"**Time:** {chat['timestamp']}")
                st.write(f"**Query:** {chat['query']}")
                st.write(f"**Answer:** {chat['result']['answer']}")
                
                if 'error' in chat['result']:
                    st.error(f"Error: {chat['result']['error']}")
                    
                # Show generated code if available
                if 'tool_results' in chat['result']:
                    for tool_result in chat['result']['tool_results']:
                        if 'Generated Pandas Code' in tool_result.get('result', ''):
                            with st.expander("View Generated Code"):
                                st.code(tool_result['result'], language='python')
    else:
        st.write("No queries yet. Ask something to get started!")


def run_cli():
    """Run command line interface."""
    print("🏥 Clinical Data Analysis Chatbot - CLI Mode")
    print("Type 'quit' to exit, 'help' for commands\n")
    
    # Initialize components
    session_id = str(uuid.uuid4())
    
    print("Loading language model...")
    llm1, llm2 = initialize_llm()
    
    print("Initializing langsmith monitoring")
    _initialize_langsmith()

    print("Initializing agent...")
    agent = get_primary_agent(llm1, llm2, session_id)
    
    print("Ready to chat!\n")
    
    while True:
        try:
            query = input("\n👤 You: ").strip()
            
            if query.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            
            if query.lower() == 'help':
                show_help()
                continue
            
            if query.lower() == 'session':
                show_session_info(agent, session_id)
                continue
            
            if query.lower() == 'clear':
                agent.clear_session()
                print("Session cleared!")
                continue
            
            if not query:
                continue
            
            print("🤖 Assistant: Processing...")
            result = process_query_sync(agent, query)
            print(f"🤖 Assistant: {result['answer']}")
            
            if 'error' in result:
                print(f"⚠️ Error: {result['error']}")
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")


def show_help():
    """Show CLI help."""
    help_text = """
Available commands:
• quit/exit/q - Exit the application
• help - Show this help message
• session - Show session information
• clear - Clear conversation memory

Example queries:
• "How many patients are there in the sheet?"
• "How many patients died from stroke?"
• "When did symptoms start for patient INSP_AU010031?"
• "What is hypodense stroke?"
• "Show me patients with high blood pressure"
"""
    print(help_text)


def show_session_info(agent, session_id):
    """Show session information in CLI."""
    try:
        session_info = agent.get_session_info()
        print(f"Session ID: {session_id}")
        print(f"Structured Data: {'✅' if session_info['data_loaded']['structured'] else '❌'}")
        print(f"Unstructured Data: {'✅' if session_info['data_loaded']['unstructured'] else '❌'}")
        print(f"Tools: {', '.join(session_info['tools_available'])}")
        
        stats = session_info['memory_stats']
        if stats['total_exchanges'] > 0:
            print(f"Exchanges: {stats['total_exchanges']}")
            print(f"Duration: {stats['session_duration']:.1f} minutes")
    except Exception as e:
        print(f"Error getting session info: {e}")


def main():
    """Main entry point."""
    import sys
    
    # Check command line arguments
    if len(sys.argv) > 1 and sys.argv[1] == "cli":
        run_cli()
    else:
        run_streamlit_app()


if __name__ == "__main__":
    main()