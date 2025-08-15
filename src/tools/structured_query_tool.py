"""Tool for querying structured clinical data."""

import logging
from typing import List, Dict, Any, Optional, Union, Tuple, ClassVar
import pandas as pd
from langchain.tools import BaseTool
from langchain.llms.base import BaseLLM
from langchain.prompts import PromptTemplate
from pydantic import Field
import re
import json
import io, sys, traceback
import textwrap
from rapidfuzz import fuzz
from sentence_transformers import SentenceTransformer, util


from .data_loader import DataLoader
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


class StructuredQueryTool(BaseTool):
    """Tool for querying structured clinical data using natural language."""
    
    name: str = "structured_data_query"
    description: str = """
    Query structured clinical data (CSV/Excel tables) using natural language.
    Use this tool when you need to:
    - Find specific patient information
    - Count patients with certain conditions
    - Calculate statistics on clinical measurements
    - Filter data based on clinical criteria
    - Analyze patient outcomes and demographics
    
    Input should be a natural language query about the clinical data.
    """
    
    data_loader: DataLoader = Field(exclude=True)
    llm: BaseLLM = Field(exclude=True)
    embedding_model: SentenceTransformer = Field(exclude=True)
    clinical_synonyms: dict = Field(exclude=True)
    # column_embeddings: Optional[dict] = Field(default=None, exclude=True)
    data: pd.DataFrame = Field(default=None, exclude=True)

    def __init__(self, data_loader: DataLoader, llm: BaseLLM, embedding_model: SentenceTransformer):
        super().__init__(
            data_loader=data_loader,
            llm=llm,
            embedding_model=embedding_model,
            clinical_synonyms=settings.clinical_synonyms
        )
        
        self.data = self.data_loader.structured_data
        print(self.data == None)
        # # Precompute embeddings for column names for faster semantic search
        # all_columns = list(self.data.columns)
        # self.column_embeddings = {
        #     col: embedding_model.encode(self._normalize(col), convert_to_tensor=True)
        #     for col in all_columns
        # }


    def _run(self, query: str) -> str:
        """Execute structured data query."""
        try:
            
            df = self.data
            relavent_column_info = self.find_query_column_matched(query, self.clinical_synonyms, df)

            # Generate pandas code to answer the query
            code_generation_prompt = PromptTemplate(
                input_variables=["query", "relavent_column_info", "data_shape"],
                template = """
                    <ROLE>
                    You are a data analyst. Write only Python pandas code to directly answer the query.
                    </ROLE>

                    <Query>
                    {query}
                    </Query>

                    <Relevant Columns Information>
                    {relavent_column_info}
                    </Relevant Columns Information>

                    <Data Shape>
                    {data_shape}
                    </Data Shape>

                    <INSTRUCTIONS>
                    - The DataFrame `df` is already provided and loaded.
                    - DO NOT import pandas or read any files (no `pd.read_excel()` or imports).
                    - Focus only on the relevant columns provided — use these columns for the code.
                    - If asked for patient in the query, use `INSPIRE ID` as the patient ID.
                    - Always store the final result in a variable called `result`.
                    - Use boolean indexing `df[df['column_name'] == value]` for filtering and only return the relevent column eg:`result = df[(df["relevant_column"] == value)][["INSPIRE ID", "relavent columns"]]`
                    - Always exclude missing values before ANY calculation or filtering:
                        * For numeric operations: use `pd.to_numeric(df['column_name'], errors='coerce')` before comparison and drop missing values.
                        * For string operations: use `df['column_name'].str.contains(..., na=False)` or `df['column_name'].notna()` before applying string methods.
                        * For boolean operations: use `df['column_name'].notna()` before comparison (e.g., `df[df['column_name'].notna() & (df['column_name'] == True)]`).
                        * For datetime operations: use `pd.to_datetime(df['column_name'], errors='coerce')` before comparison, then drop missing values.
                    - Write clean, error-free code with proper syntax.
                    - Do NOT write code that modifies the data (no write access).
                    - Generate only the Python code without any explanation, comments, or extra text.
                    - Do not repeat or rephrase the question, instructions, or context. Just output the code.
                    </INSTRUCTIONS>

                    <CODE_5348_TAG>
                    """
            )
            # Prepare sample data for context
            # sample_data = df.head(3).to_string()
            
            prompt_text = code_generation_prompt.format(
                query=query,
                relavent_column_info=relavent_column_info, 
                data_shape=f"{df.shape[0]} rows, {df.shape[1]} columns"
            )
            
            # Generate code
            generated_code = self.llm(prompt_text)
            generated_code = generated_code.split("</CODE_5348_TAG>")[0]
            
            clean_code = self.generate_clean_code(generated_code)

            # Log the generated code for debugging
            logger.info(f"Generated pandas code \n: {clean_code}")
            
            # Execute the code safely
            result = self._execute_pandas_code(clean_code, df)
            # Format the result with optional code display
            formatted_result = self._format_result(result)
            logger.info(f"Formatted result \n: {formatted_result}")
            
            logger.info(f"Structured query executed: {query[:50]}...")
            return formatted_result
            
        except Exception as e:
            logger.error(f"Structured query failed: {e}")
            # Include the generated code in error messages for debugging
            try:
                error_msg = f"I encountered an error while querying the data: {str(e)}"
                if hasattr(self, '_last_generated_code'):
                    error_msg += f"\n\nGenerated code that failed:\n```python\n{self._last_generated_code}\n```"
                return error_msg
            except:
                return f"I encountered an error while querying the data: {str(e)}"
            
    

    def generate_clean_code(self, generated_code: str) -> str:
        """Clean and normalize indentation of generated code."""
        
        # Remove markdown code fences and clean up
        lines = generated_code.strip().split('\n')
        clean_lines = []
        
        for line in lines:
            stripped = line.strip()
            
            # Skip markdown code fences
            if stripped.startswith("```"):
                continue
                
            # Skip empty lines and full-line comments
            if not stripped or stripped.startswith("#"):
                continue
                
            clean_lines.append(stripped)
        
        # Reconstruct with proper indentation
        if not clean_lines:
            return ""
        
        # Join lines and use textwrap.dedent to normalize
        code = '\n'.join(clean_lines)
        
        # Now rebuild with consistent 4-space indentation
        final_lines = []
        indent_level = 0
        
        for line in code.split('\n'):
            stripped = line.strip()
            if not stripped:
                continue
                
            # Decrease indent for except, else, finally, and closing blocks
            if any(stripped.startswith(keyword) for keyword in ['except', 'else', 'finally', 'elif']):
                indent_level = max(0, indent_level - 1)
            
            # Add line with proper indentation
            final_lines.append('    ' * indent_level + stripped)
            
            # Increase indent after colons (try, if, for, while, def, class, etc.)
            if stripped.endswith(':'):
                indent_level += 1
        
        return '\n'.join(final_lines)
    
        
    def find_query_column_matched(self, query: list[str], clinical_synonyms: dict, df: pd.DataFrame) -> str:
        """
        Analyzes DataFrame columns and identifies the most relevant ones for a given query.
        
        Uses data type inference and clinical synonyms to match query terms with appropriate
        columns, returning formatted column information strings for further processing.
        """
        try:
            column_info = []
            for col in df.columns:
                info = {"column_name": col}
                # Get data excluding 'Not Available' for type inference
                clean_data = df[col][(df[col] != '') & (df[col] != 'None')]
                # Determine actual data type from clean data
                if len(clean_data) > 0:
                    # Try numeric detection
                    numeric_vals = pd.to_numeric(clean_data, errors='coerce')
                    if numeric_vals.notna().mean() > 0.50:  # 95%+ numeric
                        info["type"] = "numeric"
                    else:
                        # Try datetime detection with explicit format guessing
                        try:
                            datetime_vals = pd.to_datetime(
                                clean_data, 
                                format="%Y-%m-%d",  # adjust to your known format
                                errors='coerce'
                            )
                        except Exception:
                            # fallback if format doesn't match
                            datetime_vals = pd.to_datetime(clean_data, errors='coerce')

                        if datetime_vals.notna().mean() > 0.50:  # >50% valid datetime
                            info["type"] = "datetime"
                        else:
                            # Boolean detection
                            unique_clean = pd.Series(clean_data.dropna().unique()).astype(str).str.lower()
                            if len(unique_clean) <= 3 and unique_clean.isin(['yes', 'no', 'true', 'false', '1', '0']).all():
                                info["type"] = "boolean"
                            else:
                                info["type"] = "categorical"
                else:
                    info["type"] = "unknown"

                
                info["Missing count"] = len(df[df[col] == ''])
                info["unique value"] = len(clean_data.unique()) if len(clean_data) > 0 else 0

                # Add sample values for categorical/string columns, excluding 'Not Available'
                if info["type"] in ['categorical', 'boolean'] or info["unique value"] < 15:
                    if len(clean_data) > 0:
                        sample_vals = clean_data.unique()[:5]
                        info["example"] = {tuple(sample_vals)}
                elif info["type"] == 'numeric' and len(clean_data) > 0:
                    # For numeric, show range instead of examples
                    try:
                        numeric_vals = pd.to_numeric(clean_data)
                        min_val = numeric_vals.min()
                        max_val = numeric_vals.max()
                        info["range"] = f"{min_val} to {max_val}"
                    except:
                        sample_vals = clean_data.unique()[:5]
                        info["example"] = {tuple(sample_vals)}
                elif info["type"] == 'datetime' and len(clean_data) > 0:
                    # For datetime, show date range
                    try:
                        date_vals = pd.to_datetime(clean_data)
                        min_date = date_vals.min().strftime('%Y-%m-%d') if pd.notna(date_vals.min()) else 'N/A'
                        max_date = date_vals.max().strftime('%Y-%m-%d') if pd.notna(date_vals.max()) else 'N/A'
                        info["range"] = f"{min_date} to {max_date}"
                    except:
                        sample_vals = clean_data.unique()[:3]
                        info["example"] = {tuple(sample_vals)}
                
                column_info.append(info)

            # Generate pandas code to answer the query
            relevent_column_generation_prompt = PromptTemplate(
                input_variables=["query", "column_info", "clinical_synonyms"],
                template = """
                    <ROLE>
                    You are an expert data analyst specializing in clinical data analysis. Your task is to identify the most relevant columns from a DataFrame that can answer the given query.
                    </ROLE>

                    <Query>
                    {query}
                    </Query>

                    <Available DataFrame Columns>
                    {column_info}
                    </Available DataFrame Columns>

                    <Clinical Terminology and Synonyms Reference>
                    {clinical_synonyms}
                    </Clinical Terminology and Synonyms Reference>

                    <INSTRUCTIONS>
                    - Carefully analyze each query term and match it against column names, considering:
                        * Exact matches and partial matches
                        * Clinical synonyms and medical terminology variations
                        * Common abbreviations and alternative spellings
                        * Semantic relationships between terms
                    - Consider data types, value ranges, and examples when assessing relevance.
                    - Prioritize columns that directly address the query's intent.
                    - Return ONLY the relevant column information exactly in the format shown above (list of dict).
                    - If the query is asking about a patient, then `INSPIRE ID` should also be considered relevant.
                    - Preserve all spaces, punctuation, and parentheses exactly as they appear.
                    - Do not include explanations, code, or additional text.
                    - If no columns are relevant, return an empty list.
                    - Do not repeat or rephrase the question, instructions, or context. Just output the relevant columns as requested.
                    </INSTRUCTIONS>

                    <RESPONSE_5348_TAG>
                    """
            )
            
            prompt_text = relevent_column_generation_prompt.format(
                    query=query,
                    column_info=column_info, 
                    clinical_synonyms=clinical_synonyms
                )
                
            # Generate relevant columns
            try:
                relevent_columns_info = self.llm(prompt_text)
                relevent_columns_info = relevent_columns_info.split("</RESPONSE_5348_TAG>")[0]
                
                logger.info(f"Relevant column information \n: {relevent_columns_info}")
                return relevent_columns_info
            
                    
            except Exception as llm_error:
                logger.error(f"Error calling LLM for column matching: {str(llm_error)}")
                # Fallback: return all column info if LLM fails
                return column_info
                
        except Exception as e:
            logger.error(f"Error in find_query_column_matched: {str(e)}")
            return ""
    
    def _execute_pandas_code(self, code: str, df: pd.DataFrame) -> Any:
        """Safely execute pandas code generated by the LLM."""
        try:
            self._last_generated_code = code

            # Whitelisted globals
            safe_globals = {
                'pd': pd,
                'df': df.copy(),  # Work on a copy to avoid destructive changes
                'len': len,
                'str': str,
                'int': int,
                'float': float,
                'list': list,
                'dict': dict,
                'set': set,
                'sum': sum,
                'max': max,
                'min': min,
                'abs': abs,
                'round': round
            }

            local_vars = {}

            # Capture stdout (for print statements)
            stdout_buffer = io.StringIO()
            sys_stdout_backup = sys.stdout
            sys.stdout = stdout_buffer

            try:
                exec(code, safe_globals, local_vars)
            finally:
                sys.stdout = sys_stdout_backup

            printed_output = stdout_buffer.getvalue().strip()

            # Determine result
            if local_vars:
                result = list(local_vars.values())[-1]
            else:
                try:
                    result = eval(code, safe_globals)
                except Exception:
                    result = None

            # If nothing is explicitly returned but there was print output, return that
            if result is None and printed_output:
                result = printed_output

            # If the code modified df but didn't return anything, return the modified df
            if result is None and not df.equals(safe_globals['df']):
                result = safe_globals['df']

            return result

        except Exception as e:
            tb_str = traceback.format_exc()
            return {
                "error": str(e),
                "traceback": tb_str,
                "generated_code": code
            }
    
    def _format_result(self, result: Any) -> str:
        """Format the query result for display."""
        try:
            if isinstance(result, pd.DataFrame):
                if len(result) == 0:
                    return "No data found matching your criteria."
                elif len(result) > 20:
                    # For large results, show summary
                    return (
                        f"Found {len(result)} records. Here are the first 10:\n\n"
                        f"{result.head(10).to_string(index=False)}\n\n"
                        f"... and {len(result) - 10} more records."
                    )
                else:
                    return result.to_string(index=False)
            
            elif isinstance(result, pd.Series):
                if len(result) == 0:
                    return "No data found matching your criteria."
                elif len(result) > 20:
                    return (
                        f"Found {len(result)} values. Summary:\n"
                        f"Count: {len(result)}\n"
                        f"First 10 values:\n{result.head(10).to_string()}"
                    )
                else:
                    return result.to_string()
            
            elif isinstance(result, (int, float)):
                return f"Result: {result}"
            
            elif isinstance(result, str):
                return result
            
            elif isinstance(result, (list, tuple)):
                if len(result) == 0:
                    return "No results found."
                elif len(result) > 10:
                    return f"Found {len(result)} items. First 10: {result[:10]}"
                else:
                    return f"Results: {result}"
            
            else:
                return str(result)
                
        except Exception as e:
            logger.error(f"Result formatting failed: {e}")
            return f"Query completed but result formatting failed: {str(result)[:500]}"
    
    
    def get_data_summary(self) -> str:
        """Get a summary of the available structured data."""
        if self.data_loader.structured_data is None:
            return "No structured data loaded."
        
        info = self.data_loader.get_structured_data_info()
        
        summary = f"""
Data Summary:
- Shape: {info['shape'][0]} rows, {info['shape'][1]} columns
- Key columns available:
"""
        
        # Group columns by category
        for category, columns in settings.column_mappings.items():
            available_cols = [col for col in columns if col in info['columns']]
            if available_cols:
                summary += f"\n  {category.title()}: {', '.join(available_cols)}"
        
        # Add data quality info
        null_counts = info['null_counts']
        high_null_cols = [col for col, count in null_counts.items() 
                         if count > info['shape'][0] * 0.5]
        
        if high_null_cols:
            summary += f"\n\nNote: Columns with >50% missing data: {', '.join(high_null_cols[:5])}"
        
        return summary
    
    async def _arun(self, query: str) -> str:
        """Async version of the tool."""
        return self._run(query)


