"""Tool for querying structured clinical data."""

import logging
from typing import List, Dict, Any, Optional, Union, Tuple
import pandas as pd
from langchain.tools import BaseTool
from langchain.llms.base import BaseLLM
from langchain.prompts import PromptTemplate
from pydantic import Field
import re
import io, sys, traceback
import textwrap
from rapidfuzz import fuzz
from sentence_transformers import SentenceTransformer, util


from .data_loader import DataLoader
from ..config.settings import settings

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
    show_code: bool = Field(default=True)  # New parameter to control code display
    embedding_model: SentenceTransformer = Field(exclude=True)
    column_mappings: dict = Field(exclude=True)
    clinical_synonyms: dict = Field(exclude=True)
    
    def __init__(self, data_loader: DataLoader, llm: BaseLLM, embedding_model: SentenceTransformer):
        super().__init__(data_loader=data_loader, llm=llm, embedding_model=embedding_model, column_mappings=settings.column_mappings,
        clinical_synonyms=settings.clinical_synonyms)


    def _run(self, query: str) -> str:
        """Execute structured data query."""
        try:
            if self.data_loader.structured_data is None:
                return "No structured data is currently loaded. Please load data first."
            
            df = self.data_loader.structured_data
            data_info = self.data_loader.get_structured_data_info()

            matched_columns, matched_domain_terms = self.preprocess_query(query)
            print(f"{matched_domain_terms} -> {matched_columns}")

            # Generate pandas code to answer the query
            code_generation_prompt = PromptTemplate(
                input_variables=["query", "matched_columns", "matched_domain_terms", "data_shape"],
                template="""
                <INSTRUCTION>
                You are a data analyst. Write only Python pandas code to directly answer the query.

                Query: {query}

                Relevant columns:
                {matched_domain_terms} -> {matched_columns}

                Data shape: {data_shape}

                Requirements:
                - The DataFrame `df` is already provided and loaded - DO NOT import pandas or read any files. So, DO NOT use pd.read_excel() and import
                - Use only the provided columns unless minimal preprocessing is required for the answer.
                - Output only accurate, valid, runnable Python pandas code. Ensure proper Python indentation structure
                - Handle missing values appropriately.
                - Do NOT perform unrelated preprocessing
                - Always compute and store the final result in a variable called `result`.
                - Print `result` as the last line of code.
                - Include basic error handling so the code will not fail if data is missing.
                </INSTRUCTION>

                <CODE>
                """
            )
            # Prepare sample data for context
            # sample_data = df.head(3).to_string()
            
            prompt_text = code_generation_prompt.format(
                query=query,
                matched_columns=matched_columns, 
                matched_domain_terms=matched_domain_terms,
                # sample_data=sample_data,
                data_shape=f"{df.shape[0]} rows, {df.shape[1]} columns"
            )
            
            # Generate code
            generated_code = self.llm(prompt_text)
            generated_code = generated_code.split("</CODE>")[0]
            
            clean_code = self.generate_clean_code(generated_code)
            print(f"Generated code:\n{clean_code}")

            
            # Log the generated code for debugging
            logger.info(f"Generated pandas code: {clean_code}")
            
            # Execute the code safely
            result = self._execute_pandas_code(clean_code, df)
            print(f"Result: {result}")
            # Format the result with optional code display
            formatted_result = self._format_result(result)
            
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
            
    # def generate_clean_code(self, generated_code: str) -> str:
    #     # Use textwrap.dedent to remove common leading whitespace
    #     # This preserves relative indentation (if statements, loops, etc.)
    #     generated_code = textwrap.dedent(generated_code).strip()
    #     code_lines = generated_code.splitlines()

    #     clean_code = []
    #     for line in code_lines:
    #         # Remove surrounding spaces from *check*, but don't strip indentation
    #         stripped_line = line.lstrip()

    #         # Skip markdown code fences
    #         if stripped_line.startswith("```"):
    #             continue

    #         # Skip entirely empty lines
    #         if not stripped_line:
    #             continue

    #         # Skip full-line comments (but keep inline comments inside code)
    #         if stripped_line.startswith("#"):
    #             continue

    #         # Keep original indentation, just strip trailing spaces
    #         clean_code.append(line.rstrip())

    #     return "\n".join(clean_code)

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

    def preprocess_query(self, query: str) -> Tuple[str, List[str], bool, List[str]]:
        """
        Map query terms to actual column names before analysis.
        Now supports:
            - Exact matches
            - Fuzzy typo tolerance
            - Plural handling
            - Synonym expansion
            - Semantic matching for unlisted synonyms
        """
        def normalize(text: str) -> str:
            return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", text.lower())).strip()

        query_norm = normalize(query)
        print(f"query_norm: {query_norm}")
        matched_columns = []
        matched_domain_terms = []

        def fuzzy_match(term: str, text: str, threshold: int = 75) -> bool:  # Lower threshold
            term_norm = normalize(term)
            score = fuzz.partial_ratio(term_norm, text)
            return score >= threshold

        def semantic_match(term: str, text: str, threshold: float = 0.75) -> bool:
            """Semantic similarity match using embeddings"""
            term_emb = self.embedding_model.encode(term, convert_to_tensor=True)
            text_emb = self.embedding_model.encode(text, convert_to_tensor=True)
            sim = util.cos_sim(term_emb, text_emb).item()
            return sim >= threshold

        # For each column, check if query contains terms that should match it
        for category, columns in self.column_mappings.items():
            for col in columns:
                col_norm = normalize(col)
                
                # Direct fuzzy match (with lower threshold)
                if fuzzy_match(col_norm, query_norm, threshold=65):
                    if col not in matched_columns:
                        matched_columns.append(col)
                        continue
                
                # NEW: Check if column contains any synonym keys
                for syn_key, syn_list in self.clinical_synonyms.items():
                    syn_key_norm = normalize(syn_key)
                    
                    # If column contains the synonym key (e.g., "systolic blood pressure" in column name)
                    print(f"syn_key_norm: {syn_key_norm}")
                    print(f"col_norm: {col_norm}")
                    if syn_key_norm in col_norm:
                        # Check if query contains any of the synonyms
                        for syn in syn_list:
                            print(f"syn: {syn}")
                            if fuzzy_match(syn, query_norm, threshold=75):  # Lower threshold
                                if col not in matched_columns:
                                    matched_columns.append(col)
                                    matched_domain_terms.append(syn)
                                    break
                        if col in matched_columns:
                            break
                
                # Semantic matching as fallback
                if col not in matched_columns:
                    try:
                        if semantic_match(col_norm, query_norm, threshold=0.70):  # Lower threshold
                            matched_columns.append(col)
                            matched_domain_terms.append(f"(semantic) {query_norm}")
                    except:
                        pass

        return matched_columns, matched_domain_terms
    
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