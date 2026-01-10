"""Agent execution functionality for running questions through the GAIA agent."""

from typing import Optional, Tuple, List, Dict
from agents import MyLangGraphAgent


class AgentRunner:
    """Handles agent execution and question processing.
    """

    def __init__(self):
        self.agent = None

    def initialize_agent(self) -> bool:
        """Initialize the agent. Returns True if successful."""
        try:
            self.agent = MyLangGraphAgent()
            return True
        except Exception as e:
            print(f"Error instantiating agent: {e}")
            return False

    def run_on_questions(self, questions_data: List[Dict]) -> Optional[List[Tuple]]:
        """Run agent on a list of questions and return results."""
        if not self.initialize_agent():
            return None

        results = []
        total = len(questions_data)
        print(f"Running agent on {total} questions...")

        for idx, item in enumerate(questions_data, 1):
            task_id = item.get("task_id")
            question_text = item.get("question")
            file_name = item.get("file_name")

            if not task_id or question_text is None:
                print(f"\nSkipping item with missing task_id or question: {item}\n")
                continue

            print(f"\n{'#'*60}")
            print(f"Processing Question {idx}/{total} - Task ID: {task_id}")
            print(f"{'#'*60}")

            try:
                answer = self.agent(question_text, file_name=file_name)
                print(f"\n[RESULT] Task ID: {task_id}")
                print(f"Question: {question_text[:200]}{'...' if len(question_text) > 200 else ''}")
                print(f"Answer: {answer}")
                results.append((task_id, question_text, answer))
            except Exception as e:
                print(f"[ERROR] Exception running agent on task {task_id}: {e}")
                error_msg = f"AGENT ERROR: {str(e)[:100]}"
                results.append((task_id, question_text, error_msg))

        return results
