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
from datetime import datetime, date

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
                    - Use boolean indexing `df[df['column_name'] == value]` for filtering and only return the relevent column eg:`result = df[(df["relevant_column"] == value)][["INSPIRE ID", "relavent columns"]]`.
                    - All string values in the data except for the column names are lower case, so while filtering use the lower case in the value, but dont use lower case for column_name.
                    - Consider the 'example' values in the provided relevant column information while filtering. wg: If the example value is [nan, 'Yes', 'No'], Instead of filtering using df['column_name'] == True, use the example value as df['column_name'] == 'Yes'.
                    - Always exclude missing values before ANY calculation or filtering:
                        * For numeric operations: use `pd.to_numeric(df['column_name'], errors='coerce')` before comparison and drop missing values.
                        * For string operations: use `df['column_name'].str.contains(..., na=False)` or `df['column_name'].notna()` before applying string methods.
                        * For boolean operations: use `df['column_name'].notna()` before comparison (e.g., `df[df['column_name'].notna() & (df['column_name'] == True)]`). 
                        * For datetime operations: use `pd.to_datetime(df['column_name'], errors='coerce')` before comparison, then drop missing values.
                        * Note: While removing missing value, remove it first and then only use filtering or comparisions so that boolean mask keeps the same index.
                    - For comparisons between multiple columns: First filter for rows where ALL relevant columns have valid (non-null) values using `df[df['column1'].notna() & df['column2'].notna()]` before performing calculations or comparisons between those columns.
                    - Write clean, error-free code with proper syntax.
                    - Do NOT write code that modifies the data (no write access) and also donot save/write any intermediate file.
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
    
    def generate_column_info(self, df: pd.DataFrame) -> str:
        """
        Generate concise column information for preprocessed DataFrame.
        Returns: column_name, type, missing_percentage, unique_count, and range/examples
        """
        column_info = []
        
        for col in df.columns:
            info = {"column_name": col}
            
            # Get clean data excluding empty, None, and NaN values
            mask = (
                (df[col] != '') & 
                (df[col] != 'None') & 
                (df[col] != 'none') &
                (df[col].notna()) &
                (df[col] != 'nan') &
                (df[col] != 'null')
            )
            
            clean_data = df[col][mask]
            total_rows = len(df)
            clean_count = len(clean_data)
            
            # Basic info
            missing_count = total_rows - clean_count
            info["missing_percentage"] = round((missing_count / total_rows) * 100, 2) if total_rows > 0 else 0
            
            # Type detection
            if clean_count == 0:
                info["type"] = "empty"
                info["unique_count"] = 0
                info["examples"] = []
            else:
                unique_values = clean_data.unique()
                info["unique_count"] = len(unique_values)
                
                # 1. Check if already numeric (from preprocessing)
                if clean_data.dtype in ['int64', 'float64', 'int32', 'float32']:
                    info["type"] = "numeric"
                else:
                    # 2. Try numeric detection
                    numeric_vals = pd.to_numeric(clean_data, errors='coerce')
                    numeric_success_rate = numeric_vals.notna().mean()
                    
                    if numeric_success_rate > 0.80:
                        info["type"] = "numeric"
                    else:
                        # 3. DateTime detection - check for date objects first
                        if any(isinstance(val, date) and not isinstance(val, str) for val in clean_data.head(10) if pd.notna(val)):
                            info["type"] = "datetime"
                        else:
                            datetime_success_rate = 0
                            try:
                                # Only try string parsing if we don't have date objects
                                datetime_vals = pd.to_datetime(clean_data, errors='coerce')
                                datetime_success_rate = datetime_vals.notna().mean()
                                    
                            except Exception:
                                datetime_success_rate = 0
                            
                            if datetime_success_rate > 0.70:
                                info["type"] = "datetime"
                            else:
                                # 4. Boolean detection
                                unique_clean_lower = pd.Series([str(val).lower().strip() for val in unique_values if pd.notna(val)])
                                unique_clean_lower = unique_clean_lower.unique()
                                
                                boolean_values = {'yes', 'no', 'true', 'false', '1', '0', 'y', 'n', 't', 'f'}
                                
                                if (len(unique_clean_lower) <= 4 and 
                                    len(unique_clean_lower) > 0 and
                                    all(val in boolean_values for val in unique_clean_lower)):
                                    info["type"] = "boolean"
                                else:
                                    # 5. Categorical vs Text
                                    uniqueness_ratio = info["unique_count"] / clean_count
                                    if uniqueness_ratio > 0.70 and info["unique_count"] > 10:
                                        info["type"] = "text"
                                    else:
                                        info["type"] = "categorical"
                
                # Add type-specific range or examples
                if info["type"] == "numeric":
                    try:
                        if clean_data.dtype not in ['int64', 'float64', 'int32', 'float32']:
                            numeric_vals = pd.to_numeric(clean_data, errors='coerce').dropna()
                        else:
                            numeric_vals = clean_data.dropna()
                        
                        if len(numeric_vals) > 0:
                            min_val = float(numeric_vals.min())
                            max_val = float(numeric_vals.max())
                            info["range"] = f"{min_val} to {max_val}"
                        else:
                            info["range"] = "No valid values"
                    except Exception:
                        info["examples"] = list(unique_values[:5])
                
                elif info["type"] == "datetime":
                    try:
                        # Handle date objects directly
                        date_objects = [val for val in clean_data if isinstance(val, date) and not isinstance(val, str)]
                        if date_objects:
                            min_date = min(date_objects).strftime('%Y-%m-%d')
                            max_date = max(date_objects).strftime('%Y-%m-%d')
                            info["range"] = f"{min_date} to {max_date}"
                        else:
                            # Handle string dates as fallback
                            try:
                                date_vals = pd.to_datetime(clean_data, errors='coerce')
                                valid_dates = date_vals.dropna()
                                
                                if len(valid_dates) > 0:
                                    min_date = valid_dates.min().strftime('%Y-%m-%d')
                                    max_date = valid_dates.max().strftime('%Y-%m-%d')
                                    info["range"] = f"{min_date} to {max_date}"
                                else:
                                    info["range"] = "No valid dates"
                            except:
                                info["range"] = "Error parsing dates"
                    except Exception:
                        info["examples"] = list(unique_values[:5])
                
                else:
                    # For categorical, boolean, text - show examples
                    sample_size = min(10, info["unique_count"])
                    info["examples"] = list(unique_values[:sample_size])
            
            column_info.append(info)
        
        return column_info

    def find_query_column_matched(self, query: list[str], clinical_synonyms: dict, df: pd.DataFrame) -> str:
        """
        Analyzes DataFrame columns and identifies the most relevant ones for a given query.
        
        Uses data type inference and clinical synonyms to match query terms with appropriate
        columns, returning formatted column information strings for further processing.
        """
        try:
            column_info = self.generate_column_info(df)

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
                        * Matching terms to the column names
                    - Before finding relavent column, properly consider the types, range and examples and check weather it relate with the question. 
                    - Looking datatype for relavance Eg: if query is related to date of onset, use "Onset Date" column with type [datetime] instead of "Onset Date/Time" which is [categorical] 
                    - Prioritize columns that directly address the query's intent.
                    - Return ONLY the relevant column information exactly in the format shown above (list of dict). Dont miss out any relevant column to list.
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


