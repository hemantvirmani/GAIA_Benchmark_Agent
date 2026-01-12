"""Agent wrapper module for GAIA Benchmark."""

import config
from langgraphagent import LangGraphAgent


class MyGAIAAgents:
    """Wrapper class to manage multiple agent implementations.

    This class provides a unified interface for different agent types.
    The active agent is determined by the ACTIVE_AGENT configuration.
    """

    def __init__(self):
        """Initialize the wrapper with the active agent based on config."""
        active_agent = config.ACTIVE_AGENT

        if active_agent == "LangGraph":
            self.agent = LangGraphAgent()
        else:
            # Default to LangGraph if unknown agent type
            print(f"[WARNING] Unknown agent type '{active_agent}', defaulting to LangGraph")
            self.agent = LangGraphAgent()

    def __call__(self, question: str, file_name: str = None) -> str:
        """Invoke the active agent with the given question.

        Args:
            question: The question to answer
            file_name: Optional file name if the question references a file

        Returns:
            The agent's answer as a string
        """
        return self.agent(question, file_name)
