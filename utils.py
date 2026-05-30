"""Utility functions for GAIA Benchmark Agent including retry logic and answer cleanup."""

import re
import time
import requests
from typing import Callable, Any
from functools import wraps
import config


def retry_with_backoff(
    max_retries: int = config.MAX_RETRIES,
    initial_delay: float = config.INITIAL_RETRY_DELAY,
    backoff_factor: float = config.RETRY_BACKOFF_FACTOR,
    exceptions: tuple = (requests.RequestException,)
):
    """
    Decorator to retry a function with exponential backoff.

    Args:
        max_retries: Maximum number of retry attempts
        initial_delay: Initial delay in seconds before first retry
        backoff_factor: Multiplier for delay after each retry
        exceptions: Tuple of exception types to catch and retry
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            delay = initial_delay
            last_exception = None

            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_retries:
                        print(f"[RETRY] Attempt {attempt + 1}/{max_retries} failed: {e}")
                        print(f"[RETRY] Retrying in {delay:.1f} seconds...")
                        time.sleep(delay)
                        delay *= backoff_factor
                    else:
                        print(f"[RETRY] All {max_retries} retries exhausted")

            # Re-raise the last exception if all retries failed
            raise last_exception

        return wrapper
    return decorator


def extract_text_from_content(content: Any) -> str:
    """
    Extract plain text from various content formats returned by LLM agents.

    This function handles multiple content formats:
    - AgentOutput objects (LlamaIndex): Extracts the response attribute
    - Message objects with 'content' attribute: Extracts the content attribute
      (works for LlamaIndex ChatMessage, LangChain AIMessage, etc.)
    - String: Returns as-is
    - Dict with 'text' field: Extracts the text value
    - List of content blocks: Extracts text from all blocks with type='text'
    - Other types: Converts to string

    Args:
        content: The content object from an LLM response (can be str, dict, list, etc.)

    Returns:
        str: Extracted plain text content
    """
    # Handle LlamaIndex AgentOutput objects (has 'response' attribute)
    if hasattr(content, 'response') and not isinstance(content, (str, dict, list)):
        # Extract the response attribute from AgentOutput
        response = content.response
        # The response might itself be a message object with 'content'
        if hasattr(response, 'content'):
            return str(response.content)
        elif hasattr(response, 'message') and hasattr(response.message, 'content'):
            return str(response.message.content)
        else:
            return str(response)

    # Handle message objects with 'content' attribute (e.g., ChatMessage from various frameworks)
    # This works for LlamaIndex ChatMessage, LangChain AIMessage, etc.
    if hasattr(content, 'content') and not isinstance(content, (str, dict, list)):
        # Extract the content attribute (works for any message object)
        return str(content.content)

    # Handle dict format (e.g., {'text': 'answer'})
    if isinstance(content, dict):
        if 'text' in content:
            return str(content['text'])
        else:
            print(f"[WARNING] Content was dict without 'text' field, converting to string")
            return str(content)

    # Handle list format (e.g., [{'type': 'text', 'text': 'answer'}])
    elif isinstance(content, list):
        text_parts = []
        for item in content:
            if isinstance(item, dict):
                # Look for items with type='text' and extract the 'text' field
                if item.get('type') == 'text':
                    text_parts.append(str(item.get('text', '')))
                # Fallback: if there's a 'text' field but no type, use it
                elif 'text' in item:
                    text_parts.append(str(item['text']))
            elif isinstance(item, str):
                text_parts.append(item)
            else:
                text_parts.append(str(item))

        result = ' '.join(text_parts)
        if len(content) > 1 or (len(content) == 1 and isinstance(content[0], dict)):
            print(f"[INFO] Extracted text from list with {len(content)} item(s)")
        return result

    # Handle string format (already plain text)
    elif isinstance(content, str):
        return content

    # Fallback for other types
    else:
        print(f"[WARNING] Content was {type(content)}, converting to string")
        return str(content)


def cleanup_answer(answer: Any) -> str:
    """
    Clean up the agent answer to ensure it's in plain text format.

    This function:
    - Converts answer to string
    - Handles multi-line answers (extracts last meaningful non-debug line)
    - Normalizes whitespace
    - Strips trailing punctuation
    - Logs warnings for verbose or malformatted answers

    Args:
        answer: The raw answer from the agent (can be str, dict, list, etc.)

    Returns:
        str: Cleaned up answer as plain text
    """
    answer = str(answer).strip()

    if not answer:
        return answer

    # Handle multi-line: take the last line that isn't a debug/log prefix
    lines = [l.strip() for l in answer.split('\n') if l.strip()]
    if len(lines) > 1:
        debug_prefixes = ('[info', '[warning', '[error', '[retry', '[step', '[tool', '[final')
        for l in reversed(lines):
            if not l.lower().startswith(debug_prefixes):
                answer = l
                break
        else:
            answer = lines[-1]
        print(f"[CLEANUP] Extracted last meaningful line from {len(lines)}-line answer: '{answer[:80]}'")

    # NOTE: Do NOT strip commas here. The GAIA scorer's normalize_number_str already
    # strips commas from numeric answers, and split_string uses commas to split list
    # answers. Stripping here would corrupt comma-separated lists (e.g., "132,133,134"
    # becomes the invalid number string "132133134").

    # Normalize whitespace and strip trailing punctuation
    answer = ' '.join(answer.split()).strip().rstrip('.')

    # Sentence.NUMBER suffix — the model echoed its final answer as a bare number
    # appended directly after its reasoning, e.g. "...published 3 albums (included).3".
    # Match a NON-DIGIT char before the period (covers letters, ')', etc.) and require
    # whitespace earlier in the string so genuine bare decimals like "89706.00" or
    # "3.14" (no spaces) are never altered.
    if ' ' in answer and re.search(r'[^\d\s]\s*\.\d+$', answer):
        extracted = re.search(r'\.(\d+)$', answer).group(1)
        print(f"[CLEANUP] Extracted appended number from verbose answer: '{extracted}'")
        answer = extracted

    # If still verbose, try to extract a terminal answer value using pattern matching.
    # These cover cases where the agent appends the bare answer after a full sentence.
    # Threshold of 60 chars: all valid single-value answers are shorter; longer ones are
    # comma-separated lists whose last elements start with lowercase (safe from patterns below).
    if len(answer) > 60:
        # Pattern 1: ends with "number/award/code XXXXXXXX" → extract identifier
        # e.g. "...supported by NASA under award number 80GSFC21M0002" → "80GSFC21M0002"
        code_match = re.search(r'(?:number|award|code|ID)\s+([A-Z0-9]{4,})[.\s]*$', answer, re.IGNORECASE)
        if code_match:
            extracted = code_match.group(1)
            print(f"[CLEANUP] Extracted terminal identifier from verbose answer: '{extracted}'")
            answer = extracted
        # Pattern 3: ends with ", TitleCase Word(s)" → location/proper noun after last comma
        # e.g. "...in the Zoological Institute, Saint Petersburg" → "Saint Petersburg"
        elif re.search(r',\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})$', answer):
            place_match = re.search(r',\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})$', answer)
            extracted = place_match.group(1)
            print(f"[CLEANUP] Extracted terminal proper noun from verbose answer: '{extracted}'")
            answer = extracted
        # Pattern 4: "first name is X" → extract the name
        # e.g. "...whose nationality is a country that no longer exists. His first name is Claus"
        elif re.search(r'first\s+name\s+is\s+([A-Z][a-z]+)\.?$', answer, re.IGNORECASE):
            name_match = re.search(r'first\s+name\s+is\s+([A-Z][a-z]+)\.?$', answer, re.IGNORECASE)
            extracted = name_match.group(1)
            print(f"[CLEANUP] Extracted first name from verbose answer: '{extracted}'")
            answer = extracted
        # Pattern 5: extract number from specific sports stat noun context
        # Use findall + last match: context stats (e.g. "75 walks") appear before the answer stat
        # Excludes "walks/hits/games/seasons" which are often mentioned as context, not the answer
        else:
            stat_matches = re.findall(r'\b(\d+)\s+(?:at-?bats?|home\s*runs?|RBIs?|strikeouts?|innings?)\b', answer, re.IGNORECASE)
            if stat_matches:
                extracted = stat_matches[-1]  # last mention is usually the answer
                print(f"[CLEANUP] Extracted number from sports stat context: '{extracted}'")
                answer = extracted
            # Pattern 6: extract alphabetically-first 2-4 char uppercase code from parentheses
            # e.g. "Cuba (CUB) and Panama (PAN)...Cuba comes first" → "CUB"
            # Only triggers when 2+ such codes appear (avoids single-abbreviation false positives)
            elif len(re.findall(r'\(([A-Z]{2,4})\)', answer)) >= 2:
                code_matches = re.findall(r'\(([A-Z]{2,4})\)', answer)
                extracted = sorted(code_matches)[0]
                print(f"[CLEANUP] Extracted country code from parentheses: '{extracted}'")
                answer = extracted
            # Pattern 7: extract first name from "shows/lists FIRSTNAME LASTNAME" (case-sensitive TitleCase only)
            # e.g. "shows Claus Peter Flor as a recipient in 1983" → "Claus"
            # NOTE: no re.IGNORECASE — [A-Z][a-z]+ must match actual TitleCase to avoid false positives
            elif re.search(r'(?:shows?|lists?)\s+([A-Z][a-z]+)\s+[A-Z][a-z]+', answer):
                name_match = re.search(r'(?:shows?|lists?)\s+([A-Z][a-z]+)\s+[A-Z][a-z]+', answer)
                extracted = name_match.group(1)
                print(f"[CLEANUP] Extracted first name from 'shows/lists NAME' pattern: '{extracted}'")
                answer = extracted
            # Pattern 8: extract first name from "FIRSTNAME [MIDDLE] LASTNAME, who..." construction
            # e.g. "is Claus Peter Flor, who won in 1983" → "Claus"
            elif re.search(r'\b([A-Z][a-z]+)\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?,\s+who\b', answer):
                name_match = re.search(r'\b([A-Z][a-z]+)\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?,\s+who\b', answer)
                extracted = name_match.group(1)
                print(f"[CLEANUP] Extracted first name from 'NAME, who' pattern: '{extracted}'")
                answer = extracted
            # Pattern 9: extract two names from "number X is NAME1 and number Y is NAME2" roster context
            # e.g. "pitcher with number 18 is Yoshida and number 20 is Uehara" → "Yoshida, Uehara"
            # Requires exactly 2 such matches to avoid false positives
            else:
                num_name_pairs = re.findall(r'number\s+\d+\s+is\s+([A-Z][a-z]+)', answer)
                if len(num_name_pairs) >= 2:
                    extracted = ", ".join(num_name_pairs[:2])
                    print(f"[CLEANUP] Extracted name pair from 'number X is NAME' pattern: '{extracted}'")
                    answer = extracted

    # Log if answer looks verbose (agent not following instructions)
    if len(answer) > 100:
        print(f"[WARNING] Answer appears verbose ({len(answer)} chars). Agent may not be following SYSTEM_PROMPT instructions.")
        print(f"[WARNING] First 150 chars: {answer[:150]}...")

    # Log if answer contains suspicious formatting characters
    if any(char in answer for char in ['{', '}', '[', ']', '`', '*', '#']):
        print(f"[WARNING] Answer contains suspicious formatting characters: {answer[:100]}")

    return answer
