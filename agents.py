"""Agent wrapper module for GAIA Benchmark."""

import config
from langgraphagent import LangGraphAgent
from reactlanggraphagent import ReActLangGraphAgent


class MyGAIAAgents:
    """Wrapper class to manage multiple agent implementations.

    This class provides a unified interface for different agent types.
    The active agent is determined by the ACTIVE_AGENT configuration or constructor parameter.
    """

    def __init__(self, active_agent: str = None):
        """Initialize the wrapper with the active agent.

        Args:
            active_agent: The agent type to use. If None, uses config.ACTIVE_AGENT.
                         Valid values: config.AGENT_LANGGRAPH, config.AGENT_REACT_LANGGRAPH
        """
        if active_agent is None:
            active_agent = config.ACTIVE_AGENT

        if active_agent == config.AGENT_LANGGRAPH:
            self.agent = LangGraphAgent()
        elif active_agent == config.AGENT_REACT_LANGGRAPH:
            self.agent = ReActLangGraphAgent()
        else:
            # Default to LangGraph if unknown agent type
            print(f"[WARNING] Unknown agent type '{active_agent}', defaulting to {config.AGENT_LANGGRAPH}")
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
