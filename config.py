"""Configuration settings for GAIA Benchmark Agent."""

import os

# API Configuration
DEFAULT_API_URL = "https://agents-course-unit4-scoring.hf.space"
AGENT_TIMEOUT_SECONDS = 180  # 3 minutes max per question

# File Paths
QUESTIONS_FILE = "files/questions.json"
METADATA_FILE = "files/metadata.jsonl"
FILES_DIR = "files"

# API Timeouts (in seconds)
FETCH_TIMEOUT = 15
SUBMIT_TIMEOUT = 60
WEBPAGE_TIMEOUT = 30

# Test Configuration
DEFAULT_TEST_FILTER = (1, 4, 7, 15)  # Q2, Q5, Q8, Q16

# Environment Variables
SPACE_HOST = os.getenv("SPACE_HOST")
SPACE_ID = os.getenv("SPACE_ID")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Model Configuration
GEMINI_MODEL = "gemini-2.5-flash"
GEMINI_TEMPERATURE = 0
GEMINI_MAX_TOKENS = 1024
