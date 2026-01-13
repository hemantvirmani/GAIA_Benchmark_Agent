import os
import logging
import warnings
import time

# Suppress TensorFlow/Keras warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
logging.getLogger('tensorflow').setLevel(logging.ERROR)
warnings.filterwarnings('ignore', module='tensorflow')
warnings.filterwarnings('ignore', module='tf_keras')

from langchain import hub
from langchain.agents import create_react_agent, AgentExecutor
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate

from custom_tools import get_custom_tools_list
from system_prompt import SYSTEM_PROMPT
from utils import cleanup_answer
import config

# Suppress BeautifulSoup GuessedAtParserWarning
try:
    from bs4 import GuessedAtParserWarning
    warnings.filterwarnings('ignore', category=GuessedAtParserWarning)
except ImportError:
    pass


class ReActLangGraphAgent:
    """
    ReAct agent implementation using LangChain's create_react_agent function.

    This agent uses the ReAct (Reasoning + Acting) pattern where the agent
    reasons about what to do and then acts by calling tools iteratively.
    """

    def __init__(self):
        # Validate API keys
        if not os.getenv("GOOGLE_API_KEY"):
            print("WARNING: GOOGLE_API_KEY not found - analyze_youtube_video will fail")

        self.tools = get_custom_tools_list()
        self.llm = self._create_llm_client()
        self.agent_executor = self._build_agent()

    def _create_llm_client(self):
        """Create and return the LLM client."""
        apikey = os.getenv("GOOGLE_API_KEY")

        return ChatGoogleGenerativeAI(
            model=config.GEMINI_MODEL,
            temperature=config.GEMINI_TEMPERATURE,
            api_key=apikey,
            timeout=60
        )

    def _create_react_prompt(self) -> PromptTemplate:
        """
        Create a custom ReAct prompt template that incorporates the system prompt.

        Returns:
            PromptTemplate: A prompt template for the ReAct agent
        """
        # Create a ReAct-style prompt that includes our system prompt
        template = """
{system_prompt}

You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question (PLAIN TEXT ONLY, NO EXTRA TEXT)

CRITICAL: Your Final Answer must be PLAIN TEXT ONLY - just the answer itself with no additional text, markdown, or formatting.

Begin!

Question: {input}
Thought: {agent_scratchpad}
"""

        return PromptTemplate(
            template=template,
            input_variables=["input", "agent_scratchpad", "tools", "tool_names"],
            partial_variables={"system_prompt": SYSTEM_PROMPT}
        )

    def _build_agent(self) -> AgentExecutor:
        """Build and return the ReAct agent executor."""

        # Create custom prompt
        prompt = self._create_react_prompt()

        # Create the ReAct agent using create_react_agent
        agent = create_react_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=prompt
        )

        # Create agent executor with configuration
        agent_executor = AgentExecutor(
            agent=agent,
            tools=self.tools,
            verbose=True,
            max_iterations=40,  # Match the step limit from LangGraphAgent
            max_execution_time=None,
            handle_parsing_errors=True,
            return_intermediate_steps=False
        )

        return agent_executor

    def __call__(self, question: str, file_name: str = None) -> str:
        """
        Invoke the ReAct agent with the given question and return the final answer.

        Args:
            question: The question to answer
            file_name: Optional file name if the question references a file

        Returns:
            The agent's answer as a string
        """
        print(f"\n{'='*60}")
        print(f"[REACT AGENT START] Question: {question}")
        if file_name:
            print(f"[FILE] {file_name}")
        print(f"{'='*60}")

        start_time = time.time()

        try:
            # Build the question with file name if provided
            question_content = question
            if file_name:
                question_content += f'\n\nNote: This question references a file: {file_name}'

            # Invoke the agent executor with retry logic for 504 errors
            max_retries = config.MAX_RETRIES
            delay = config.INITIAL_RETRY_DELAY

            for attempt in range(max_retries + 1):
                try:
                    response = self.agent_executor.invoke({"input": question_content})
                    # Success - break out of retry loop
                    break
                except Exception as e:
                    error_msg = str(e)

                    # Check if this is a 504 DEADLINE_EXCEEDED error
                    if "504" in error_msg and "DEADLINE_EXCEEDED" in error_msg:
                        if attempt < max_retries:
                            print(f"[RETRY] Attempt {attempt + 1}/{max_retries} failed with 504 DEADLINE_EXCEEDED")
                            print(f"[RETRY] Retrying in {delay:.1f} seconds...")
                            time.sleep(delay)
                            delay *= config.RETRY_BACKOFF_FACTOR
                            continue
                        else:
                            print(f"[RETRY] All {max_retries} retries exhausted for 504 error")
                            print(f"[ERROR] Agent invocation failed after retries: {e}")
                            return f"Error: Agent failed after {max_retries} retries - {str(e)[:100]}"
                    else:
                        # Not a 504 error - fail immediately without retry
                        print(f"[ERROR] Agent invocation failed: {e}")
                        return f"Error: Agent failed - {str(e)[:100]}"

            elapsed_time = time.time() - start_time
            print(f"[REACT AGENT COMPLETE] Time: {elapsed_time:.2f}s")
            print(f"{'='*60}\n")

            # Extract the answer from the response
            answer = response.get("output")

            if answer is None:
                print("[WARNING] Agent completed but returned None as answer")
                return "Error: No answer generated"

            # Clean up the answer using utility function
            answer = cleanup_answer(answer)

            print(f"[FINAL ANSWER] {answer}")
            return answer

        except Exception as e:
            elapsed_time = time.time() - start_time
            print(f"[REACT AGENT ERROR] Failed after {elapsed_time:.2f}s: {e}")
            print(f"{'='*60}\n")
            return f"Error: {str(e)[:100]}"
