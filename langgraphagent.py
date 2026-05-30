import os
import logging
import warnings
import re
import time

# Suppress TensorFlow/Keras warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
logging.getLogger('tensorflow').setLevel(logging.ERROR)
warnings.filterwarnings('ignore', module='tensorflow')
warnings.filterwarnings('ignore', module='tf_keras')

from typing import TypedDict, Optional, List, Annotated
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import MessagesState, StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import tools_condition
from langgraph.prebuilt import ToolNode
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

from custom_tools import get_custom_tools_list, reset_tool_counters
from system_prompt import SYSTEM_PROMPT
from utils import cleanup_answer, extract_text_from_content
import config

# Suppress BeautifulSoup GuessedAtParserWarning
try:
    from bs4 import GuessedAtParserWarning
    warnings.filterwarnings('ignore', category=GuessedAtParserWarning)
except ImportError:
    pass


class AgentState(TypedDict):
    question: str
    messages: Annotated[list , add_messages]   # for LangGraph
    answer: str
    step_count: int  # Track number of iterations to prevent infinite loops
    file_name: str  # Optional file name for questions that reference files


class LangGraphAgent:

    def __init__(self):
        # Validate API keys
        if not config.GOOGLE_API_KEY:
            print("WARNING: GOOGLE_API_KEY not found - analyze_youtube_video will fail")

        self.tools = get_custom_tools_list()
        self.llm_client_with_tools = self._create_llm_client()
        self.graph = self._build_graph()

    def _create_llm_client(self, model_provider: str = "google"):
        """Create and return the LLM client with tools bound based on the model provider."""

        if model_provider == "google":
            apikey = config.GOOGLE_API_KEY

            return ChatGoogleGenerativeAI(
                model=config.ACTIVE_AGENT_LLM_MODEL,
                temperature=0,
                api_key=apikey,
                thinking_budget=0,
                timeout=120
                ).bind_tools(self.tools)

        elif model_provider == "huggingface":
            LLM_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
            apikey = os.getenv("HUGGINGFACEHUB_API_TOKEN")

            llmObject = HuggingFaceEndpoint(
                repo_id=LLM_MODEL,
                task="text-generation",
                max_new_tokens=512,
                temperature=0.7,
                do_sample=False,
                repetition_penalty=1.03,
                huggingfacehub_api_token=apikey
            )
            return ChatHuggingFace(llm=llmObject).bind_tools(self.tools)

    # Nodes
    def _init_questions(self, state: AgentState):
        """Initialize the messages in the state with system prompt and user question."""

        # Reset per-question tool counters (e.g., analyze_image call limit)
        reset_tool_counters()

        # Build the question message, including file name if available
        question_content = state["question"]
        if state.get("file_name"):
            question_content += f'\n\nNote: This question references a file: {state["file_name"]}'

        return {
            "messages": [
                    SystemMessage(content=SYSTEM_PROMPT),
                    HumanMessage(content=question_content)
                    ],
            "step_count": 0  # Initialize step counter
                }

    def _assistant(self, state: AgentState):
        """Assistant node which calls the LLM with tools"""

        # Track and log current step
        current_step = state.get("step_count", 0) + 1
        print(f"[STEP {current_step}] Calling assistant with {len(state['messages'])} messages")

        # Force termination at step limit — _should_continue cannot persist state changes
        # so we detect the near-limit here and force a final LLM call without tool binding
        if current_step >= 44:
            existing = state.get("answer")
            if existing:
                return {"messages": [], "answer": existing, "step_count": current_step}
            print(f"[WARNING] Near step limit at step {current_step} with no answer — forcing bare LLM call")
            from langchain_core.messages import SystemMessage as SM
            forced_suffix = SM(content="STOP ALL TOOL CALLS. Based only on information gathered so far, output ONLY the bare answer value — one word, number, or short phrase. No explanation.")

            def _extract_content(resp_content):
                if not resp_content:
                    return ""
                if isinstance(resp_content, str):
                    return resp_content.strip()
                if isinstance(resp_content, list):
                    parts = [item['text'] if isinstance(item, dict) and 'text' in item else str(item) for item in resp_content]
                    return " ".join(parts).strip()
                return str(resp_content).strip()

            llm_client = self.llm_client_with_tools
            if llm_client is None:
                return {"messages": [], "answer": "Error: Step limit reached", "step_count": current_step}

            # Attempt 1: full context
            try:
                forced_messages = list(state["messages"]) + [forced_suffix]
                forced_resp = llm_client.invoke(forced_messages)
                content = _extract_content(forced_resp.content)
                if content:
                    print(f"[FORCED FINAL] {content[:100]}")
                    return {"messages": [], "answer": content, "step_count": current_step}
                print("[FORCED FINAL] Empty content on attempt 1, retrying with reduced context")
            except Exception as fe:
                print(f"[WARNING] Forced final call attempt 1 failed: {fe}")

            # Attempt 2: reduced context (first 2 messages + last 10 messages) to avoid token overload
            try:
                msgs = state["messages"]
                reduced = msgs[:2] + (msgs[-10:] if len(msgs) > 12 else msgs[2:])
                reduced_messages = reduced + [forced_suffix]
                forced_resp2 = llm_client.invoke(reduced_messages)
                content2 = _extract_content(forced_resp2.content)
                if content2:
                    print(f"[FORCED FINAL REDUCED] {content2[:100]}")
                    return {"messages": [], "answer": content2, "step_count": current_step}
                print("[FORCED FINAL] Empty content on attempt 2 as well")
            except Exception as fe2:
                print(f"[WARNING] Forced final call attempt 2 failed: {fe2}")

            return {"messages": [], "answer": "Error: Step limit reached", "step_count": current_step}

        # Invoke LLM with tools enabled, with retry logic for 504 errors
        max_retries = config.MAX_RETRIES
        delay = config.INITIAL_RETRY_DELAY

        for attempt in range(max_retries + 1):
            try:
                response = self.llm_client_with_tools.invoke(state["messages"])
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
                        print(f"[ERROR] LLM invocation failed after retries: {e}")
                        return {
                            "messages": [],
                            "answer": f"Error: LLM failed after {max_retries} retries - {str(e)[:100]}",
                            "step_count": current_step
                        }
                else:
                    # Not a 504 error - fail immediately without retry
                    print(f"[ERROR] LLM invocation failed: {e}")
                    return {
                        "messages": [],
                        "answer": f"Error: LLM failed - {str(e)[:100]}",
                        "step_count": current_step
                    }

        # If no tool calls, set the final answer
        if not response.tool_calls:
            content = response.content
            print(f"[FINAL ANSWER] Agent produced answer (no tool calls)")

            # Handle case where content is a list (e.g. mixed content from Gemini)
            if isinstance(content, list):
                # Extract text from list of content parts
                text_parts = []
                for item in content:
                    if isinstance(item, dict) and 'text' in item:
                        text_parts.append(item['text'])
                    elif hasattr(item, 'text'):
                        text_parts.append(item.text)
                    else:
                        text_parts.append(str(item))
                content = " ".join(text_parts)
            elif isinstance(content, dict) and 'text' in content:
                # Handle single dict with 'text' field
                content = content['text']
            elif hasattr(content, 'text'):
                # Handle object with text attribute
                content = content.text
            else:
                # Fallback to string conversion
                content = str(content)

            # Clean up any remaining noise
            content = content.strip()
            print(f"[EXTRACTED TEXT] {content[:100]}{'...' if len(content) > 100 else ''}")

            # If content is empty (transient Gemini API issue), retry once with same messages
            if not content:
                print(f"[WARNING] Empty response from LLM at step {current_step} — retrying once")
                try:
                    retry_resp = self.llm_client_with_tools.invoke(state["messages"])  # type: ignore[union-attr]
                    retry_content = retry_resp.content
                    if isinstance(retry_content, str):
                        content = retry_content.strip()
                    elif isinstance(retry_content, list):
                        parts = [item['text'] if isinstance(item, dict) and 'text' in item else str(item) for item in retry_content]
                        content = " ".join(parts).strip()
                    if content:
                        print(f"[RETRY SUCCESS] Got content on retry: {content[:80]}")
                except Exception as re_err:
                    print(f"[WARNING] Retry also failed: {re_err}")

            return {
                "messages": [response],
                "answer": content,
                "step_count": current_step
            }

        # Has tool calls, log them
        print(f"[TOOL CALLS] Agent requesting {len(response.tool_calls)} tool(s):")
        for tc in response.tool_calls:
            print(f"  - {tc['name']}")

        return {
            "messages": [response],
            "step_count": current_step
        }


    def _should_continue(self, state: AgentState):
        """Check if we should continue or stop based on step count and other conditions."""

        step_count = state.get("step_count", 0)

        # Stop if we've exceeded maximum steps
        if step_count >= 45:  # Conservative: triggers before recursion_limit=120 (45*2=90 < 120)
            print(f"[WARNING] Max steps (45) reached, forcing termination")
            # Force a final answer if we don't have one
            if not state.get("answer"):
                state["answer"] = "Error: Maximum iteration limit reached"
            return END

        # Otherwise use the default tools_condition
        return tools_condition(state)


    def _build_graph(self):
        """Build and return the Compiled Graph for the agent."""

        graph = StateGraph(AgentState)

        # Build graph
        graph.add_node("init", self._init_questions)
        graph.add_node("assistant", self._assistant)
        graph.add_node("tools", ToolNode(self.tools))
        graph.add_edge(START, "init")
        graph.add_edge("init", "assistant")
        graph.add_conditional_edges(
            "assistant",
            # Use custom should_continue instead of tools_condition
            self._should_continue,
        )
        graph.add_edge("tools", "assistant")
        # Compile graph
        return graph.compile()

    def __call__(self, question: str, file_name: str = None) -> str:
        """Invoke the agent graph with the given question and return the final answer.

        Args:
            question: The question to answer
            file_name: Optional file name if the question references a file
        """

        print(f"\n{'='*60}")
        print(f"[LANGGRAPH AGENT START] Question: {question}")
        if file_name:
            print(f"[FILE] {file_name}")
        print(f"{'='*60}")

        start_time = time.time()

        try:
            response = self.graph.invoke(
                {"question": question, "messages": [], "answer": None, "step_count": 0, "file_name": file_name or ""},
                config={"recursion_limit": 120}  # Must be > 2x step limit (45 * 2 = 90 < 120)
            )

            elapsed_time = time.time() - start_time
            print(f"[LANGGRAPH AGENT COMPLETE] Time: {elapsed_time:.2f}s")
            print(f"{'='*60}\n")

            answer = response.get("answer")
            if not answer or answer is None:
                print("[WARNING] Agent completed but returned None as answer")
                return "Error: No answer generated"

            # Use utility function to extract text from various content formats
            answer = extract_text_from_content(answer)

            # Clean up the answer using utility function (includes stripping)
            answer = cleanup_answer(answer)

            print(f"[FINAL ANSWER] {answer}")
            return answer

        except Exception as e:
            elapsed_time = time.time() - start_time
            print(f"[LANGGRAPH AGENT ERROR] Failed after {elapsed_time:.2f}s: {e}")
            print(f"{'='*60}\n")
            return f"Error: {str(e)[:100]}"
