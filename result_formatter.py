"""Result formatting for different output types."""

import pandas as pd
from typing import List, Tuple, Dict


class ResultFormatter:
    """Formats results for different output targets."""

    @staticmethod
    def format_for_api(results: List[Tuple[str, str, str]]) -> List[Dict]:
        """Format results for API submission."""
        return [
            {"task_id": task_id, "submitted_answer": answer}
            for task_id, _, answer in results
        ]

    @staticmethod
    def format_for_display(results: List[Tuple[str, str, str]]) -> List[Dict]:
        """Format results for UI display."""
        return [
            {
                "Task ID": task_id,
                "Question": question_text,
                "Submitted Answer": answer
            }
            for task_id, question_text, answer in results
        ]

    @staticmethod
    def format_for_verification(results: List[Tuple[str, str, str]]) -> List[str]:
        """Format results for test verification output."""
        output = []
        for task_id, question_text, answer in results:
            output.append(f"\nTask ID: {task_id}")
            output.append(f"Question: {question_text}")
            output.append(f"Answer: {answer}")
        return output

    @staticmethod
    def print_dataframe(df: pd.DataFrame) -> None:
        """Print DataFrame with full content (no truncation)."""
        pd.set_option('display.max_colwidth', None)
        pd.set_option('display.max_rows', None)
        for col in df.columns:
            for val in df[col]:
                print(val, flush=True)
