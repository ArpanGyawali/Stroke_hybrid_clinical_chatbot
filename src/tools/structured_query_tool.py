"""Tool for querying structured clinical data."""

import logging
from typing import List, Dict, Any, Optional, Union
import pandas as pd
from langchain.tools import BaseTool
from langchain.llms.base import BaseLLM
from langchain.prompts import PromptTemplate
from pydantic import Field

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
    
    def __init__(self, data_loader: DataLoader, llm: BaseLLM, show_code: bool = True):
        super().__init__(data_loader=data_loader, llm=llm, show_code=show_code)
    
    def _run(self, query: str) -> str:
        """Execute structured data query."""
        try:
            if self.data_loader.structured_data is None:
                return "No structured data is currently loaded. Please load data first."
            
            df = self.data_loader.structured_data
            data_info = self.data_loader.get_structured_data_info()
            
            # Generate pandas code to answer the query
            code_generation_prompt = PromptTemplate(
                input_variables=["query", "columns", "sample_data", "data_shape"],
                template="""
You are a data analyst. Generate Python pandas code to directly answer the following query about clinical data.

Query: {query}

Available columns: {columns}

Sample data:
{sample_data}

Data shape: {data_shape}

Instructions:
1. Use only the available columns unless minimal preprocessing is required for the answer.
2. Handle missing values appropriately.
3. Do NOT perform unrelated preprocessing or drop unrelated columns.
4. Always compute and return the final result in a variable called `result`.
5. Print `result` as the last line of code.
6. Include error handling so the code does not fail if data is missing.
7. Use descriptive variable names.
8. Return only the Python pandas code without prompt, explanations, comments, or additional text. The generated code must be syntactically correct and runnable as is.
"""
            )
            
            # Prepare sample data for context
            sample_data = df.head(3).to_string()
            
            prompt_text = code_generation_prompt.format(
                query=query,
                columns=", ".join(data_info["columns"]),
                sample_data=sample_data,
                data_shape=f"{df.shape[0]} rows, {df.shape[1]} columns"
            )
            
            # Generate code
            generated_code = self.llm(prompt_text)
            
            # Clean the generated code
            code_lines = generated_code.strip().split('\n')
            clean_code = []
            for line in code_lines:
                line = line.strip()
                if line and not line.startswith('#') and not line.startswith('```'):
                    clean_code.append(line)
            
            code = '\n'.join(clean_code)
            
            # Log the generated code for debugging
            logger.info(f"Generated pandas code: {code}")
            
            # Execute the code safely
            result = self._execute_pandas_code(code, df)
            
            # Format the result with optional code display
            formatted_result = self._format_result_with_code(result, query, code)
            
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
    
    def _execute_pandas_code(self, code: str, df: pd.DataFrame) -> Any:
        """Safely execute pandas code."""
        try:
            # Store the code for error reporting
            self._last_generated_code = code
            
            # Create a safe execution environment
            safe_globals = {
                'pd': pd,
                'df': df,
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
            
            # Execute the code
            local_vars = {}
            exec(code, safe_globals, local_vars)
            
            # Find the result (usually the last variable assigned)
            if local_vars:
                # Get the last assigned variable
                result = list(local_vars.values())[-1]
            else:
                # If no variables were assigned, try evaluating the code
                result = eval(code, safe_globals)
            
            return result
            
        except Exception as e:
            # Try alternative execution methods
            try:
                # Simple eval for single expressions
                result = eval(code, {'pd': pd, 'df': df})
                return result
            except:
                raise Exception(f"Code execution failed: {str(e)}")
    
    def _format_result_with_code(self, result: Any, original_query: str, generated_code: str) -> str:
        """Format the query result with optional code display."""
        try:
            # Format the main result
            main_result = self._format_result(result, original_query)
            
            # Add code section if enabled
            if self.show_code:
                code_section = f"\n\n📝 **Generated Pandas Code:**\n```python\n{generated_code}\n```"
                return main_result + code_section
            else:
                return main_result
                
        except Exception as e:
            logger.error(f"Result formatting failed: {e}")
            return f"Query completed but result formatting failed: {str(result)[:500]}"
    
    def _format_result(self, result: Any, original_query: str) -> str:
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
    
    def set_show_code(self, show_code: bool) -> None:
        """Enable or disable showing generated code in results."""
        self.show_code = show_code
        logger.info(f"Code display {'enabled' if show_code else 'disabled'}")
    
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