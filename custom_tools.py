import concurrent.futures
from ddgs import DDGS
from bs4 import BeautifulSoup
import requests
import re
import io
import os
import subprocess
import sys
from google import genai
from google.genai import types
import config

from langchain_community.document_loaders import WikipediaLoader
from langchain_community.document_loaders import ArxivLoader
from youtube_transcript_api import YouTubeTranscriptApi
from pytube import extract
from langchain_core.tools import tool

import pandas as pd
import speech_recognition as sr
from pydub import AudioSegment
from pypdf import PdfReader
from io import BytesIO
from markdownify import markdownify as md

# ============================================================================
# Per-question tool call counters (reset at start of each question)
# ============================================================================
_analyze_image_call_count = 0
MAX_ANALYZE_IMAGE_CALLS = 2


def reset_tool_counters():
    """Reset per-question tool counters. Call at the start of each new question."""
    global _analyze_image_call_count
    _analyze_image_call_count = 0


# ============================================================================
# Helper Functions (must be defined before tools that use them)
# ============================================================================

def _sanitize_file_path(file_name: str) -> tuple:
    """
    Sanitize file name to prevent path traversal attacks.

    Args:
        file_name: The file name to sanitize

    Returns:
        tuple: (is_valid: bool, sanitized_name_or_error: str)
    """
    # Check for path traversal attempts
    if '..' in file_name or file_name.startswith('/') or file_name.startswith('\\'):
        return False, "Invalid file name: path traversal not allowed"

    # Check for absolute paths (Windows and Unix)
    if os.path.isabs(file_name):
        return False, "Invalid file name: absolute paths not allowed"

    # Normalize the path and ensure it doesn't escape the files directory
    normalized = os.path.normpath(file_name)
    if normalized.startswith('..') or os.path.isabs(normalized):
        return False, "Invalid file name: path traversal detected"

    return True, normalized

def _get_file_content(file_name: str, mode: str = 'binary'):
    """
    Helper function to get file content from local filesystem or remote URL.

    Args:
        file_name: The file name (without 'files/' prefix)
        mode: 'binary' for bytes, 'text' for string

    Returns:
        tuple: (success: bool, data: bytes/str or error_message: str)

    NOTE — File source for GAIA benchmark question attachments:
    The question files (.png, .mp3, .py, .xlsx, etc.) are NOT served by the
    scoring API at agents-course-unit4-scoring.hf.space. A previous version of
    this code defaulted to that URL, which caused silent 404 failures for any
    question that referenced a file attachment.

    The correct source is the HuggingFace dataset:
        repo: gaia-benchmark/GAIA  (type: dataset)
        path: 2023/validation/<file_name>

    This function now tries sources in order:
        1. Local files/ directory (cache)
        2. HuggingFace dataset download (saves to files/ for future runs)
        3. SPACE_HOST env var (only when deployed on HF Spaces)

    To pre-download all question files manually, run:
        python -c "
        import json, os, shutil
        from huggingface_hub import hf_hub_download
        questions = json.load(open('files/questions.json', encoding='utf-8'))
        for q in questions:
            fn = q.get('file_name', '')
            if fn and not os.path.exists(f'files/{fn}'):
                src = hf_hub_download('gaia-benchmark/GAIA', f'2023/validation/{fn}', repo_type='dataset')
                shutil.copy(src, f'files/{fn}')
                print('Downloaded', fn)
        "
    """
    # Sanitize file name first
    is_valid, result = _sanitize_file_path(file_name)
    if not is_valid:
        return False, result

    file_name = result  # Use sanitized name
    file_path = f"files/{file_name}"

    def _read(path: str):
        if mode == 'binary':
            with open(path, 'rb') as f:
                return True, f.read()
        else:
            with open(path, 'r', encoding='utf-8') as f:
                return True, f.read()

    # 1. Local cache
    if os.path.exists(file_path):
        try:
            return _read(file_path)
        except Exception as e:
            return False, f"Error reading local file: {e}"

    # 2. HuggingFace GAIA dataset — downloads and caches locally
    try:
        import shutil
        from huggingface_hub import hf_hub_download
        print(f"[INFO] Downloading {file_name} from HuggingFace GAIA dataset...")
        hf_local = hf_hub_download(
            repo_id='gaia-benchmark/GAIA',
            filename=f'2023/validation/{file_name}',
            repo_type='dataset',
        )
        os.makedirs('files', exist_ok=True)
        shutil.copy(hf_local, file_path)
        print(f"[INFO] Cached to {file_path}")
        return _read(file_path)
    except Exception as e:
        print(f"[WARNING] HuggingFace download failed for {file_name}: {e}")

    # 3. SPACE_HOST fallback (only when explicitly deployed on a HF Space that serves files)
    space_host = os.getenv("SPACE_HOST")
    if space_host:
        try:
            if not space_host.startswith("http"):
                file_url = f"https://{space_host}/files/{file_name}"
            else:
                file_url = f"{space_host}/files/{file_name}"
            print(f"[INFO] Fetching {file_name} from {file_url}")
            response = requests.get(file_url, timeout=30)
            response.raise_for_status()
            if mode == 'binary':
                return True, response.content
            else:
                return True, response.text
        except Exception as e:
            print(f"[WARNING] SPACE_HOST fetch failed for {file_name}: {e}")

    return False, f"Could not retrieve file '{file_name}' from any source."

def _get_mime_type(file_name: str) -> str:
    """Helper function to determine MIME type from file extension."""
    ext = file_name.lower().split('.')[-1]
    mime_types = {
        'png': 'image/png',
        'jpg': 'image/jpeg',
        'jpeg': 'image/jpeg',
        'gif': 'image/gif',
        'webp': 'image/webp',
        'bmp': 'image/bmp'
    }
    return mime_types.get(ext, 'image/png')

# ============================================================================
# Tools
# ============================================================================

@tool
def calculate(operation: str, a: float, b: float) -> str:
    """Perform a basic arithmetic operation on two numbers.

    Args:
        operation (str): One of 'add', 'subtract', 'multiply', 'divide', 'power', 'modulus'.
        a (float): First number.
        b (float): Second number.
    """
    op = (operation or "").strip().lower()
    if op == "add":
        return str(a + b)
    elif op == "subtract":
        return str(a - b)
    elif op == "multiply":
        return str(a * b)
    elif op == "divide":
        if b == 0:
            return "Cannot divide by zero"
        return str(a / b)
    elif op == "power":
        return str(a ** b)
    elif op == "modulus":
        return str(int(a) % int(b))
    else:
        return f"Unsupported operation '{operation}'. Use: add, subtract, multiply, divide, power, modulus."

@tool
def string_reverse(input_string: str) -> str:
    """
    Reverses the input string. Useful whenever a string seems to be non-sensical or
    contains a lot of gibberish. This function can be used to reverse the string
    and check if it makes more sense when reversed.

    Args:
        input_string (str): The string to reverse.

    Returns:
        str: The reversed string.
    """
    return input_string[::-1]


@tool
def websearch(query: str) -> str:
    """This tool will search the web using DuckDuckGo.

    Args:
        query: The search query.
    """

    try:
        print(f"websearch called: {query}")
        with DDGS() as ddgs:
            results = ddgs.text(query, max_results=5, timelimit='y')  # Limit to past year for faster results
            if results:
                print(f"websearch results: {len(results)}")
                return "\n\n".join([f"Title: {r['title']}\nURL: {r['href']}\nSnippet: {r['body']}" for r in results])
            return "No results found. Try search with a different query."
    except Exception as e:
        return f"Search error (try again): {str(e)}"

@tool
def wiki_search(query: str) -> str:
    """Search Wikipedia for a query and return maximum 3 results.

    Args:
        query: The search query."""
    try:
        print(f"wiki_search called: {query}")

        search_docs = WikipediaLoader(query=query, load_max_docs=3).load()
        formatted_search_docs = "\n\n---\n\n".join(
            [
                f'<Document source="{doc.metadata["source"]}" page="{doc.metadata.get("page", "")}"/>\n{doc.page_content}\n</Document>'
                for doc in search_docs
            ])
        print(f"wiki_results: {len(formatted_search_docs)} characters")
        return {"wiki_results": formatted_search_docs}
    except Exception as e:
        return f"Error performing wikipedia search: {e}. try again."

@tool
def arvix_search(query: str) -> str:
    """Search Arxiv for a query and return maximum 3 result.

    Args:
        query: The search query."""
    try:
        print(f"arvix_search called: {query}")

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(lambda: ArxivLoader(query=query, load_max_docs=3).load())
            search_docs = future.result(timeout=config.ARXIV_TIMEOUT_SECONDS)

        formatted_search_docs = "\n\n---\n\n".join(
            [
                f'<Document source="{doc.metadata["source"]}" page="{doc.metadata.get("page", "")}"/>\n{doc.page_content[:1000]}\n</Document>'
                for doc in search_docs
            ])

        print(f"arvix_results: {len(formatted_search_docs)} characters")
        return {"arvix_results": formatted_search_docs}
    except concurrent.futures.TimeoutError:
        return f"ArXiv timed out after {config.ARXIV_TIMEOUT_SECONDS}s — try websearch instead"
    except Exception as e:
        return f"Error performing arxiv search: {e}. try again."

@tool
def youtube_tool(youtube_url: str, question: str = "") -> str:
    """Get the transcript of a YouTube video, or analyze it with AI to answer a question.

    If question is provided, uses a multimodal AI model to analyze the video (handles visual
    or audio content beyond just transcript). If question is empty, returns the raw transcript.

    Args:
        youtube_url (str): Full HTTPS URL of the YouTube video.
        question (str): Optional question to answer about the video. If empty, returns raw transcript.
    """
    print(f"youtube_tool called: {youtube_url} question={question!r}")

    if not question:
        # Transcript-only path — no API key needed
        try:
            video_id = extract.video_id(youtube_url)
            ytt_api = YouTubeTranscriptApi()
            transcript = ytt_api.fetch(video_id)
            txt = '\n'.join([s.text for s in transcript.snippets])
            print(f"youtube_transcript: {len(txt)} characters")
            return txt
        except Exception as e:
            msg = f"youtube_tool (transcript) failed: {e}"
            print(msg)
            return msg

    # AI analysis path
    try:
        api_key = config.GOOGLE_API_KEY
        if not api_key:
            return "Error: GOOGLE_API_KEY environment variable not set"

        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model=config.GEMINI_MODEL,
            contents=[types.Content(
                parts=[
                    types.Part(file_data=types.FileData(file_uri=youtube_url)),
                    types.Part(text=question)
                ]
            )],
            config=types.GenerateContentConfig(
                temperature=config.GEMINI_TEMPERATURE,
                max_output_tokens=config.GEMINI_MAX_TOKENS,
            )
        )
        return response.text or "(no response from model)"
    except Exception as e:
        error_msg = f"youtube_tool (AI analysis) failed: {str(e)[:config.QUESTION_PREVIEW_LENGTH]}"
        print(error_msg)
        return error_msg

@tool
def get_webpage_content(page_url: str) -> str:
    """Load a web page and return it as markdown if possible

    Args:
        page_url (str): the URL of web page to get

    Returns:
        str: The content of the page(s).
   """

    try:
        print(f"get_web_page_content called: with url {page_url}")
        r = requests.get(page_url, timeout=30)  # Add 30s timeout
        r.raise_for_status()
        text = ""
        # special case if page is a PDF file
        if r.headers.get('Content-Type', '') == 'application/pdf':
            pdf_file = BytesIO(r.content)
            reader = PdfReader(pdf_file)
            for page in reader.pages:
                text += page.extract_text()
        else:
            soup = BeautifulSoup((r.text), 'html.parser')
            if soup.body:
                # convert to markdown
                text = md(str(soup.body))
            else:
                # return the raw content
                text = r.text
        print(f"webpage_content: {len(text)} characters")
        return text
    except Exception as e:
        return f"get_webpage_content failed: {e}"

@tool
def read_file(file_name: str, sheet_name: str = "") -> str:
    """Read a file from the files directory and return its content.

    Supported formats:
    - .xlsx / .csv  → returned as a Markdown table
    - .py / .txt / .md / .json / .jsonl → returned as raw text

    Args:
        file_name (str): Name of the file (e.g., 'data.xlsx'). Do not include 'files/' prefix.
        sheet_name (str): For Excel files, the sheet name to read. Leave empty to read the first sheet.
    """
    print(f"read_file called: {file_name}")
    ext = file_name.rsplit(".", 1)[-1].lower() if "." in file_name else ""

    if ext in ("xlsx", "xls"):
        success, data = _get_file_content(file_name, mode='binary')
        if not success:
            return f"Error: {data}"
        assert isinstance(data, bytes)
        try:
            df = pd.read_excel(BytesIO(data), sheet_name=sheet_name or 0)
            return df.to_markdown(index=False)
        except Exception as e:
            return f"Error reading Excel file: {e}"

    if ext == "csv":
        success, data = _get_file_content(file_name, mode='binary')
        if not success:
            return f"Error: {data}"
        assert isinstance(data, bytes)
        try:
            df = pd.read_csv(BytesIO(data))
            return df.to_markdown(index=False)
        except Exception as e:
            return f"Error reading CSV file: {e}"

    # Text-based formats
    if ext in ("py", "txt", "md", "json", "jsonl", ""):
        success, data = _get_file_content(file_name, mode='text')
        if not success:
            return f"Error: {data}"
        return data

    return f"Unsupported file type '.{ext}'. Supported: xlsx, xls, csv, py, txt, md, json, jsonl."

@tool
def parse_audio_file(file_name: str) -> str:
    """
    Transcribes audio from an MP3 file into text.
    Use this tool to extract speech/text from audio files.

    Args:
        file_name (str): The name of the MP3 file (e.g., 'audio.mp3'). Do not include the 'files/' prefix.

    Returns:
        str: The transcribed text.
    """

    try:
        print(f"parse_audio_file called: with file {file_name}")

        # Get file content using helper function
        success, data = _get_file_content(file_name, mode='binary')
        if not success:
            return f"Error: Failed to read audio file. {data}"

        # Load audio from bytes
        audio = AudioSegment.from_file(io.BytesIO(data), format="mp3")
        # SpeechRecognition works best with WAV data so we to WAV format in memory
        wav_data = io.BytesIO()
        audio.export(wav_data, format="wav")
        wav_data.seek(0)  # Rewind the buffer to the beginning

        # Now we directly process the WAV data
        recognizer = sr.Recognizer()
        with sr.AudioFile(wav_data) as source:
            audio_data = recognizer.record(source)
        text = recognizer.recognize_google(audio_data)
        return text

    except sr.RequestError as e:
        return f"Error: Could not request results from Google Web Speech API; {e}"
    except Exception as e:
        if "ffmpeg" in str(e).lower() or "avlib" in str(e).lower():
            return f"Error: Failed to process audio. Reason: {e}. Ensure ffmpeg is installed and in your system's PATH."
        return f"Error: Failed to parse the audio file. Reason: {e}"

@tool
def analyze_image(question: str, file_name: str) -> str:
    """
    Analyzes an image file and answers a specific question about it using AI vision.
    Use this tool when you need to understand image content (e.g., chess positions, diagrams, photos).

    Args:
        question (str): The question you want answered about the image.
        file_name (str): The name of the image file (e.g., 'image.png'). Do not include the 'files/' prefix.

    Returns:
        str: The answer to the question based on the image analysis.
    """

    global _analyze_image_call_count
    _analyze_image_call_count += 1
    print(f"analyze_image called: {file_name} with question: {question}")
    if _analyze_image_call_count > MAX_ANALYZE_IMAGE_CALLS:
        return (
            f"ERROR: analyze_image has already been called {_analyze_image_call_count - 1} times. "
            f"MAXIMUM is {MAX_ANALYZE_IMAGE_CALLS}. "
            "Do NOT call analyze_image again. Commit to the chess position already described and use "
            "execute_python with the chess library to find the winning move."
        )

    try:
        api_key = config.GOOGLE_API_KEY
        if not api_key:
            return "Error: GOOGLE_API_KEY environment variable not set"

        # Get file content using helper function
        success, image_data = _get_file_content(file_name, mode='binary')
        if not success:
            return f"Error: Failed to read image file. {image_data}"

        client = genai.Client(api_key=api_key)

        # Use Gemini vision model with image data
        response = client.models.generate_content(
            model=config.GEMINI_MODEL,
            contents=[types.Content(
                parts=[
                    types.Part(inline_data=types.Blob(
                        mime_type=_get_mime_type(file_name),
                        data=image_data
                    )),
                    types.Part(text=question)
                ]
            )],
            config=types.GenerateContentConfig(
                temperature=config.GEMINI_TEMPERATURE,
                max_output_tokens=config.GEMINI_MAX_TOKENS,
            )
        )
        return response.text

    except Exception as e:
        error_msg = f"Error analyzing image: {str(e)[:config.QUESTION_PREVIEW_LENGTH]}"
        print(error_msg)
        return error_msg


@tool
def classical_cipher(cipher_type: str, mode: str, text: str, keyword: str = "", period: int = 5) -> str:
    """Encrypt or decrypt common classical ciphers.

    Supported ciphers: playfair, bifid.

    Args:
        cipher_type (str): Cipher family: 'playfair' or 'bifid'.
        mode (str): 'encrypt' or 'decrypt'.
        text (str): Input text (letters only; j is mapped to i).
        keyword (str): Key phrase used to build the 5x5 square.
        period (int): Bifid period (ignored for Playfair). Default is 5.
    """
    ctype = (cipher_type or "").strip().lower()
    op = (mode or "").strip().lower()
    if ctype not in {"playfair", "bifid"}:
        return "Unsupported cipher_type. Use 'playfair' or 'bifid'."
    if op not in {"encrypt", "decrypt"}:
        return "Unsupported mode. Use 'encrypt' or 'decrypt'."
    if period <= 0:
        return "Invalid period. Use a positive integer."

    alphabet = "abcdefghiklmnopqrstuvwxyz"

    def _normalize(s: str) -> str:
        return re.sub(r"[^a-z]", "", (s or "").lower().replace("j", "i"))

    def _build_square(key: str):
        seen = []
        for c in _normalize(key) + alphabet:
            if c not in seen:
                seen.append(c)
        sq = [seen[i * 5:(i + 1) * 5] for i in range(5)]
        pos = {c: (r, cidx) for r, row in enumerate(sq) for cidx, c in enumerate(row)}
        inv = {(r, cidx): ch for r, row in enumerate(sq) for cidx, ch in enumerate(row)}
        return sq, pos, inv

    sq, pos, inv = _build_square(keyword)
    normalized = _normalize(text)
    if not normalized:
        return ""

    if ctype == "playfair":
        if len(normalized) % 2 != 0:
            normalized = normalized + "x"
        d = -1 if op == "decrypt" else 1
        out = []
        for i in range(0, len(normalized), 2):
            a, b = normalized[i], normalized[i + 1]
            ra, ca = pos[a]
            rb, cb = pos[b]
            if ra == rb:
                out.append(sq[ra][(ca + d) % 5])
                out.append(sq[rb][(cb + d) % 5])
            elif ca == cb:
                out.append(sq[(ra + d) % 5][ca])
                out.append(sq[(rb + d) % 5][cb])
            else:
                out.append(sq[ra][cb])
                out.append(sq[rb][ca])
        return "".join(out)

    # bifid
    if op == "encrypt":
        out = []
        for i in range(0, len(normalized), period):
            block = normalized[i:i + period]
            rows, cols = [], []
            for ch in block:
                r, c = pos[ch]
                rows.append(r + 1)
                cols.append(c + 1)
            nums = rows + cols
            for j in range(0, len(nums), 2):
                out.append(inv[(nums[j] - 1, nums[j + 1] - 1)])
        return "".join(out)

    out = []
    for i in range(0, len(normalized), period):
        block = normalized[i:i + period]
        nums = []
        for ch in block:
            r, c = pos[ch]
            nums.extend([r + 1, c + 1])
        half = len(block)
        rows, cols = nums[:half], nums[half:]
        for rr, cc in zip(rows, cols):
            out.append(inv[(rr - 1, cc - 1)])
    return "".join(out)


@tool
def execute_python(code: str) -> str:
    """Execute a Python code snippet and return its stdout output.

    Use this for precise computations the LLM cannot do reliably:
    counting characters, implementing algorithms (ciphers, prime sieves),
    math calculations, data transformations, etc.

    Args:
        code (str): Valid Python 3 code. Use print() to produce output.
                    Do not read/write files or make network calls from within the code.
    """
    timeout = 30
    try:
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if result.returncode == 0:
            return result.stdout.strip() or "(no output)"
        return f"Exit {result.returncode}:\n{result.stderr.strip()}"
    except subprocess.TimeoutExpired:
        return f"Execution timed out after {timeout}s"
    except Exception as e:
        return f"execute_python failed: {e}"


@tool
def http_request(method: str, url: str, headers_json: str = "{}", body_json: str = "{}") -> str:
    """Make an HTTP request with a custom method, headers, and JSON body.

    Use this for POST, DELETE, or authenticated GET requests that require
    custom headers (e.g. Authorization: Bearer ...) or a request body.

    Args:
        method (str): HTTP method — 'GET', 'POST', or 'DELETE'.
        url (str): The full URL to call.
        headers_json (str): JSON object of request headers, e.g. '{"Authorization": "Bearer TOKEN"}'.
        body_json (str): JSON object for the request body (POST only). Use '{}' for empty body.

    Returns:
        str: Response body as text, prefixed with the HTTP status code.
    """
    import json
    method = method.upper()
    try:
        headers = json.loads(headers_json)
    except Exception as e:
        return f"Invalid headers_json: {e}"
    try:
        body = json.loads(body_json)
    except Exception as e:
        return f"Invalid body_json: {e}"

    try:
        if method == "GET":
            r = requests.get(url, headers=headers, timeout=30)
        elif method == "POST":
            r = requests.post(url, headers=headers, json=body, timeout=30)
        elif method == "DELETE":
            r = requests.delete(url, headers=headers, timeout=30)
        else:
            return f"Unsupported method '{method}'. Use GET, POST, or DELETE."
        try:
            content = json.dumps(r.json(), ensure_ascii=False)
        except ValueError:
            content = r.text
        return f"HTTP {r.status_code}\n{content}"
    except Exception as e:
        return f"http_request failed ({method} {url}): {e}"


@tool
def download_file(url: str, file_name: str) -> str:
    """Download a binary file from a URL and save it to the files directory.

    Use this before calling read_file, parse_audio_file,
    or analyze_image on files fetched from an API.
    After downloading, call the appropriate tool with the same file_name.

    Args:
        url (str): The full URL of the file to download.
        file_name (str): Local file name to save as (e.g. 'data.xlsx', 'audio.mp3').
                         Must not contain path separators or '..'.
    """
    if "/" in file_name or "\\" in file_name or ".." in file_name:
        return "Invalid file_name: path separators and '..' are not allowed."

    try:
        r = requests.get(url, timeout=60)
        r.raise_for_status()
    except Exception as e:
        return f"download_file failed (fetch): {e}"

    os.makedirs(config.FILES_DIR, exist_ok=True)
    dest = os.path.join(config.FILES_DIR, file_name)
    try:
        with open(dest, "wb") as f:
            f.write(r.content)
        return f"Downloaded {len(r.content)} bytes → {dest}"
    except Exception as e:
        return f"download_file failed (write): {e}"


@tool
def ask_advisor(question: str) -> str:
    """Consult a more powerful AI model when you are stuck or uncertain after 2+ failed attempts.

    Describe what you are trying to solve and what you have already tried.
    The advisor returns a concise recommendation (2-3 sentences) to guide your next step.
    Use sparingly — only for genuinely hard reasoning or planning problems, not for tool failures.

    Args:
        question (str): A clear description of the problem and what approaches you have already tried.
    """
    try:
        api_key = config.GOOGLE_API_KEY
        if not api_key:
            return "Error: GOOGLE_API_KEY not configured"
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model=config.GEMINI_MODEL,
            contents=question,
            config=types.GenerateContentConfig(
                system_instruction=(
                    "You are an expert advisor for an AI agent that is stuck on a search or reasoning problem. "
                    "Give a concise, actionable recommendation in 2-3 sentences about what to search for or how to reason. "
                    "Do NOT suggest installing Python packages or software. "
                    "Do NOT suggest writing code. "
                    "Only give search strategy or reasoning guidance."
                ),
                temperature=0,
            )
        )
        return response.text or "Advisor returned no response."
    except Exception as e:
        return f"Advisor unavailable: {e}"


# ============================================================================
# Tools List
# ============================================================================


def get_custom_tools_list() -> list:
    """Get list of all custom tools for the agent.

    Returns:
        list: List of tool functions
    """
    tools = [
        calculate,
        string_reverse,
        websearch,
        wiki_search,
        arvix_search,
        youtube_tool,
        get_webpage_content,
        read_file,
        parse_audio_file,
        analyze_image,
        classical_cipher,
        execute_python,
        ask_advisor,
        http_request,
        download_file,
    ]
    return tools
