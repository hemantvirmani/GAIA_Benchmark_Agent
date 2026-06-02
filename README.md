---
title: GAIA Benchmark Agent
emoji: 🕵🏻‍♂️
colorFrom: indigo
colorTo: indigo
sdk: gradio
sdk_version: 6.2.0
app_file: app.py
pinned: false
hf_oauth: true
hf_oauth_expiration_minutes: 480
---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference


# GAIA Benchmark Agent

A LangGraph-based AI agent designed to solve questions from the GAIA (General AI Assistants) benchmark. This agent uses Google's Gemini model with custom tools for web search, file processing, and multimodal analysis to answer complex questions requiring reasoning and information gathering.

**🔗 Live on Hugging Face Spaces:** [hemantvirmani/Final_Assignment_Template](https://huggingface.co/spaces/hemantvirmani/Final_Assignment_Template) — pushes to GitHub are synced to the Space by a GitHub Action and auto-deployed as a running Gradio app.

## 🏆 Results — and yes, I'm gloating

**18 out of 20. That's 90%.** On the GAIA benchmark. With a *flash-tier* model. Go ahead, re-read that.

GAIA is the benchmark that's supposed to humble "general AI assistants" — questions that chain web research, file parsing, audio transcription, image analysis, Excel math, and multi-step reasoning, all expecting one exact answer. My agent clears **90%** of it. Two misses out of twenty, and one of those is a chess puzzle that a general-purpose vision model has no honest business solving anyway.

No hardcoded answers. No question-specific cheats. Just tools, reasoning, and a system prompt that knows what it's doing. I built this, it works, and I'm absolutely taking the bow. 🎯

> **Why 90% and not 100%?** The official GAIA leaderboard on Hugging Face may show this agent at **100%** — but I'm only claiming **90%** here, on purpose. A couple of answers were effectively obtained by the agent crawling the web for (or recalling) the answer itself rather than genuinely reasoning its way there the intended way. Those don't count in my book, so I don't claim them. 90% is the number I'll stand behind honestly.

## Features

- **LangGraph Architecture**: Implements a state-graph agent workflow with tool calling capabilities
- **Multimodal Capabilities**:
  - Image analysis (PNG, JPG, JPEG, GIF, WebP, BMP)
  - YouTube video analysis and transcript extraction
  - Audio transcription (MP3)
  - PDF and Excel file processing
- **Web Research Tools**:
  - DuckDuckGo web search (with a duplicate-query guard that short-circuits repeated identical searches)
  - Wikipedia integration
  - ArXiv academic paper search
  - Web page content extraction (browser User-Agent so sources like Wikipedia don't return 403)
- **Mathematical Operations**: Basic arithmetic and modulus operations
- **Gradio Interface**: User-friendly web UI for testing and evaluation
- **Automated Evaluation**: Fetches questions from API, processes them, and submits answers
- **Observability**: Built-in integration with Langfuse for tracking traces and metrics

## Project Structure

```
GAIA_Benchmark_Agent/
├── app.py              # Main application entry point
├── agents.py           # LangGraph agent implementation
├── custom_tools.py     # Tool definitions for web search, files, etc.
├── system_prompt.py    # Agent system prompt and instructions
├── gradioapp.py        # Gradio UI components
├── requirements.txt    # Python dependencies
└── files/
    └── metadata.jsonl  # Ground truth data for local testing
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/GAIA_Benchmark_Agent.git
cd GAIA_Benchmark_Agent
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
export GOOGLE_API_KEY="your_google_api_key"

# Required for questions with attached files (image/audio/Excel/Python).
# The GAIA dataset is access-gated, so file downloads need an authenticated token.
# Request access to gaia-benchmark/GAIA, then set a read token here.
export HF_TOKEN="your_hf_token"

# Langfuse Observability (Optional)
export LANGFUSE_PUBLIC_KEY="pk-lf-..."
export LANGFUSE_SECRET_KEY="sk-lf-..."
export LANGFUSE_HOST="https://cloud.langfuse.com" # Optional
```

> **Note:** On Hugging Face Spaces, set `GOOGLE_API_KEY` and `HF_TOKEN` as Space **secrets**. Without `HF_TOKEN`, the gated dataset returns `401` and every file-based question fails.

## Requirements

- Python 3.8+
- Google API Key (for Gemini model)
- ffmpeg (optional, for audio processing)

### Key Dependencies

- `langchain-core`, `langgraph` - Agent framework
- `langchain-google-genai` - Google Gemini integration
- `gradio` - Web UI
- `requests`, `beautifulsoup4` - Web scraping
- `pypdf`, `pandas` - File processing
- `youtube-transcript-api` - YouTube integration
- `ddgs` - DuckDuckGo search

## Usage

### Running the Gradio Interface

Launch the web interface for interactive testing:

```bash
python app.py
```

This will start a Gradio app where you can:
- Log in with your Hugging Face account
- Run evaluation on all questions
- Test individual questions
- View results and scores

### Running Local Tests

Test the agent from the command line, scored locally against ground truth — nothing is submitted to the server:

```bash
# Run ALL 20 questions
python app.py --testall

# Run specific questions by index (0-based), passed directly on the command line
python app.py --test 0,4,9

# Run the default test subset
python app.py --test
```

Indices are passed as arguments — no need to edit the source. You can also pick the agent with `--agent` (`langgraph`, `reactlangg`, `llamaindex`).

### Using the Agent Programmatically

```python
from agents import MyGAIAAgents

# Initialize agent (automatically uses ACTIVE_AGENT from config)
agent = MyGAIAAgents()

# Ask a question
answer = agent("What is the capital of France?")
print(answer)

# Ask a question with a file reference
answer = agent(
    "What data is in this spreadsheet?",
    file_name="data.xlsx"
)
print(answer)
```

## How It Works

### Agent Architecture

The agent is built using LangGraph with the following workflow:

1. **Initialize**: Loads the question and system prompt
2. **Assistant Node**: Calls the LLM (Gemini) to decide on tool usage
3. **Tool Node**: Executes requested tools (search, file reading, etc.)
4. **Iteration**: Loops between assistant and tools until answer is found
5. **Termination**: Returns final answer or hits step limit (60 steps max)

### Available Tools

**Search & Research:**

- `websearch` - DuckDuckGo web search (with a duplicate-query guard)
- `wiki_search` - Wikipedia articles
- `arvix_search` - Academic papers
- `get_webpage_content` - Extract webpage/PDF text (browser User-Agent to avoid 403s)

**Files & Multimodal:**

- `read_file` - One tool for Excel/CSV → markdown tables and `.py`/`.txt`/`.md`/`.json` → raw text
- `parse_audio_file` - Transcribe MP3 files
- `analyze_image` - AI vision analysis of images
- `youtube_tool` - One tool for both raw transcripts and AI analysis of video (visual + audio)
- `download_file` - Fetch a remote file into the `files/` directory

**Computation & Utilities:**

- `calculate` - One tool for all arithmetic (`add`, `subtract`, `multiply`, `divide`, `power`, `modulus`)
- `execute_python` - Run Python for precise counting, algorithms, or file processing
- `string_reverse` - Reverse encoded/gibberish text
- `classical_cipher` - Encrypt/decrypt Playfair and Bifid ciphers
- `http_request` - Make GET/POST/DELETE requests
- `ask_advisor` - Consult a more capable model for search strategy when stuck

> Math, file-reading, and YouTube were previously several separate tools each — they've since been **consolidated** into single multi-purpose tools (`calculate`, `read_file`, `youtube_tool`).

### System Prompt

The agent follows strict output formatting rules defined in [system_prompt.py](system_prompt.py):
- Returns only the final answer (no conversational filler)
- No markdown formatting or JSON structures
- Uses tools instead of guessing
- Handles encoded/reversed text
- Verifies answers before output

## Configuration

### Change Agent Type

Edit the `ACTIVE_AGENT` variable in [config.py:32](config.py#L32):

```python
# Valid values: "LangGraph", "ReActLangGraph", "LLamaIndex", "SMOL"
ACTIVE_AGENT = "LangGraph"  # Currently only LangGraph is implemented
```

The `MyGAIAAgents` wrapper class will automatically instantiate the correct agent based on this configuration.

### Adjust Step Limits

Step limits live in [config.py](config.py) as a single source of truth — change `AGENT_STEP_LIMIT` and the derived values update automatically:

```python
AGENT_STEP_LIMIT = 60                          # max assistant iterations per question
AGENT_RECURSION_LIMIT = AGENT_STEP_LIMIT * 2 + 20  # LangGraph's guard, kept > 2x the step limit
```

The agent forces a final bare-answer call one step before the limit, and a duplicate-query guard on `websearch` prevents wasted iterations spent re-running identical searches.

### Customize Tools

Add or modify tools in [custom_tools.py](custom_tools.py) using the `@tool` decorator:

```python
from langchain_core.tools import tool

@tool
def my_custom_tool(param: str) -> str:
    """Tool description for the LLM."""
    # Implementation
    return result
```

## API Integration

The agent integrates with the GAIA benchmark API:

- **Questions Endpoint**: `https://agents-course-unit4-scoring.hf.space/questions`
- **Submit Endpoint**: `https://agents-course-unit4-scoring.hf.space/submit`

Questions may include attached files (image/audio/Excel/Python). These are **not** served by the scoring API; they are resolved in this order:

1. Local `files/` directory (cache)
2. The gated HuggingFace dataset `gaia-benchmark/GAIA` — requires `HF_TOKEN` (see [Installation](#installation))
3. `SPACE_HOST` fallback (only when deployed on a Space that serves the file)

## Testing

Local runs are scored against the GAIA ground truth using the official scorer and are **not** submitted to the server — so you can measure performance before a real submission. The submit path (`/submit`) is reached only through the Gradio button, never from the CLI.

For the commands, see [Running Local Tests](#running-local-tests).

## Performance Considerations

- **Timeout**: Agent has 180-second timeout per question
- **Step Limit**: Maximum 60 reasoning steps to prevent infinite loops
- **Tool Timeouts**: Individual tools have their own timeout settings
- **Cost**: Uses Google Gemini API (gemini-3.5-flash model)

## Deployment

### Hugging Face Spaces

This project runs on Hugging Face Spaces as a live Gradio app, deployed through an automated pipeline — you never push to Hugging Face directly:

1. **Push code to GitHub** (`main` branch).
2. A **GitHub Action** ([.github/workflows/sync-to-hf.yml](.github/workflows/sync-to-hf.yml)) copies the repository to the Hugging Face Space's git, as a single orphan commit (force-pushed), so the Space history stays clean of old binary files.
3. **Hugging Face builds and serves** the Gradio app from that commit.

So pushing to GitHub `main` is what triggers a redeploy. Doc-only changes are skipped — the workflow ignores `README.md`, `**.md`, `docs/**`, and `LICENSE` — so editing only docs won't rebuild the Space.

**Secrets involved:**

- **GitHub repo secret** — `HF_SYNC_TOKEN`: a Hugging Face write token the Action uses to push to the Space.
- **Hugging Face Space secrets** — `GOOGLE_API_KEY` and `HF_TOKEN`: used by the app at runtime (`SPACE_ID` / `SPACE_HOST` are injected automatically).

To replicate on your own Space: create a Gradio Space (SDK 6.2.0+), enable OAuth, add the two Space secrets above, add `HF_SYNC_TOKEN` to your GitHub repo, and point the workflow's remote at your Space. The app auto-detects the Hugging Face environment and configures URLs accordingly.

### Local Deployment

Simply run `python app.py` locally. The app will detect it's not in a Hugging Face Space and adjust behavior accordingly.

## Troubleshooting

### Common Issues

**"GOOGLE_API_KEY not found"**
- Set the environment variable: `export GOOGLE_API_KEY="your_key"`

**Audio parsing fails**
- Install ffmpeg: `apt-get install ffmpeg` (Linux) or `brew install ffmpeg` (macOS)

**Tool timeouts**
- Adjust timeout values in respective tool functions in [custom_tools.py](custom_tools.py)

**Agent exceeds step limit**
- Increase `AGENT_STEP_LIMIT` in [config.py](config.py) or optimize tool usage in the system prompt

## Contributing

Contributions are welcome! Areas for improvement:

- Add more tools (database access, code execution, etc.)
- Push the benchmark from 90% to 100% (the chess image question is the white whale)
- Improve error handling and retry logic
- Try with smaller LLMs
- Make it work with Ollama

## License

This project is open-source and available under the MIT License.

## Acknowledgments

- Built for the GAIA (General AI Assistants) benchmark
- Uses Google's Gemini model via LangChain
- LangGraph framework by LangChain
- Gradio for web interface

## Contact

For questions, issues, or suggestions, please open an issue on GitHub.
