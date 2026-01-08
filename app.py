import os
import sys
import argparse
import requests
import pandas as pd
import json

# Import agent-related code from agents module
from agents import MyLangGraphAgent
# Import Gradio UI creation function
from gradioapp import create_ui

# --- Constants ---
DEFAULT_API_URL = "https://agents-course-unit4-scoring.hf.space"
AGENT_TIMEOUT_SECONDS = 180  # 3 minutes max per question (enforced by agent's internal limits)


def FetchQuestions(api_url: str):
    """Fetch questions from the given API URL."""

    questions_url = f"{api_url}/questions"
    print(f"Fetching questions from: {questions_url}")
    try:
        response = requests.get(questions_url, timeout=15)
        response.raise_for_status()
        questions_data = response.json()

        if not questions_data:
             print("Fetched questions list is empty.")
             return "Fetched questions list is empty or invalid format.", None
        print(f"Fetched {len(questions_data)} questions.")
        return questions_data
    except requests.exceptions.RequestException as e:
        print(f"Error fetching questions: {e}")
        return f"Error fetching questions: {e}", None
    except requests.exceptions.JSONDecodeError as e:
         print(f"Error decoding JSON response from questions endpoint: {e}")
         print(f"Response text: {response.text[:500]}")
         return f"Error decoding server response for questions: {e}", None
    except Exception as e:
        print(f"An unexpected error occurred fetching questions: {e}")
        return f"An unexpected error occurred fetching questions: {e}", None

def run_and_submit_all(username: str):
    """
    Fetches all questions, runs the MyLangGraphAgent on them, submits all answers,
    and displays the results.
    """

    # --- Determine HF Space Runtime URL and Repo URL ---
    space_id = os.getenv("SPACE_ID") # Get the SPACE_ID for sending link to the code
    submit_url = f"{DEFAULT_API_URL}/submit"
    # In the case of an app running as a hugging Face space, this link points toward your codebase ( usefull for others so please keep it public)
    agent_code = f"https://huggingface.co/spaces/{space_id}/tree/main"

    # 1. Instantiate Agent ( modify this part to create your agent)
    try:
        agent = MyLangGraphAgent()
    except Exception as e:
        print(f"Error instantiating agent: {e}")
        return f"Error initializing agent: {e}", None

    # 2. Fetch Questions
    questions_data = FetchQuestions(DEFAULT_API_URL)

    # 3. Run your Agent
    results_log = []
    answers_payload = []
    print(f"Running agent on {len(questions_data)} questions...")
    for idx, item in enumerate(questions_data, 1):
        task_id = item.get("task_id")
        question_text = item.get("question")
        file_name = item.get("file_name")
        if not task_id or question_text is None:
            print(f"\nSkipping item with missing task_id or question: {item}\n")
            continue

        print(f"\n{'#'*60}")
        print(f"Processing Question {idx}/{len(questions_data)} - Task ID: {task_id}")
        print(f"{'#'*60}")

        try:
            # Run agent with question and optional file_name
            # The agent will handle file URLs internally if needed
            submitted_answer = agent(question_text, file_name=file_name)

            answers_payload.append({"task_id": task_id, "submitted_answer": submitted_answer})
            print(f"\n[RESULT] Task ID: {task_id}")
            print(f"Question: {question_text[:200]}{'...' if len(question_text) > 200 else ''}")
            print(f"Answer: {submitted_answer}")
            results_log.append({"Task ID": task_id, "Question": question_text, "Submitted Answer": submitted_answer})

        except Exception as e:
             print(f"[ERROR] Exception running agent on task {task_id}: {e}")
             error_msg = f"AGENT ERROR: {str(e)[:100]}"
             answers_payload.append({"task_id": task_id, "submitted_answer": error_msg})
             results_log.append({"Task ID": task_id, "Question": question_text, "Submitted Answer": error_msg})

    if not answers_payload:
        print("Agent did not produce any answers to submit.")
        return "Agent did not produce any answers to submit.", pd.DataFrame(results_log)

    # 4. Prepare Submission
    submission_data = {"username": username.strip(), "agent_code": agent_code, "answers": answers_payload}
    print(f"\n{'='*60}")
    print(f"Agent finished. Submitting {len(answers_payload)} answers for user '{username}'...")
    print(f"{'='*60}\n")

    # 5. Submit
    print(f"Submitting {len(answers_payload)} answers to: {submit_url}")
    try:
        response = requests.post(submit_url, json=submission_data, timeout=60)
        response.raise_for_status()
        result_data = response.json()
        final_status = (
            f"Submission Successful!\n"
            f"User: {result_data.get('username')}\n"
            f"Overall Score: {result_data.get('score', 'N/A')}% "
            f"({result_data.get('correct_count', '?')}/{result_data.get('total_attempted', '?')} correct)\n"
            f"Message: {result_data.get('message', 'No message received.')}"
        )
        print("Submission successful.")
        results_df = pd.DataFrame(results_log)
        return final_status, results_df
    except requests.exceptions.HTTPError as e:
        error_detail = f"Server responded with status {e.response.status_code}."
        try:
            error_json = e.response.json()
            error_detail += f" Detail: {error_json.get('detail', e.response.text)}"
        except requests.exceptions.JSONDecodeError:
            error_detail += f" Response: {e.response.text[:500]}"
        status_message = f"Submission Failed: {error_detail}"
        print(status_message)
        results_df = pd.DataFrame(results_log)
        return status_message, results_df
    except requests.exceptions.Timeout:
        status_message = "Submission Failed: The request timed out."
        print(status_message)
        results_df = pd.DataFrame(results_log)
        return status_message, results_df
    except requests.exceptions.RequestException as e:
        status_message = f"Submission Failed: Network error - {e}"
        print(status_message)
        results_df = pd.DataFrame(results_log)
        return status_message, results_df
    except Exception as e:
        status_message = f"An unexpected error occurred during submission: {e}"
        print(status_message)
        results_df = pd.DataFrame(results_log)
        return status_message, results_df

def load_ground_truth(file_path="files/metadata.jsonl"):
    truth_mapping = {}
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                question = data.get("Question")
                answer = data.get("Final answer")
                if question and answer:
                    truth_mapping[question] = answer
    except Exception as e:
        print(f"Error loading ground truth: {e}")
    return truth_mapping

def verify_answers(results, log_output):
    ground_truth = load_ground_truth()
    log_output.append("\n=== Verification Results ===")
    for question, answer in results:
        if question in ground_truth:
            correct_answer = ground_truth[question]
            is_correct = (str(answer).strip() == str(correct_answer).strip())
            log_output.append(f"Question: {question}...")
            log_output.append(f"Expected: {correct_answer}")
            log_output.append(f"Got: {answer}")
            log_output.append(f"Match: {'Correct' if is_correct else 'Incorrect'}\n")
        else:
            log_output.append(f"Question: {question[:50]}... - No ground truth found.\n")

def run_test_code():
    log_output = []
    results_to_verify = []
    log_output.append("=== Processing Example Questions One by One ===")

    my_questions = []
    my_questions_data = FetchQuestions(DEFAULT_API_URL)
    if isinstance(my_questions_data, list):
        # Extract both question and file_name for selected indices
        my_questions = [
            {
                "question": my_questions_data[i]["question"],
                "file_name": my_questions_data[i].get("file_name")
            }
            #for i in (0, 1, 3, 4, 5, 9, 11, 13, 14, 17, 18) if i < len(my_questions_data) - All 11 incorrect questions
            #for i in (0, 1, 4, 5, 14, 17) if i < len(my_questions_data) - All 6 incorrect except ones with files
            for i in (0, 5, 17) if i < len(my_questions_data)
        ]
        #print(f"Running these Questions:\n{chr(10).join(f'{i}. {q[\"question\"]}' for i, q in enumerate(my_questions, 1))}\n")

    # 1. Instantiate Agent ( modify this part to create your agent)
    try:
        my_agent = MyLangGraphAgent()
    except Exception as e:
        msg = f"Error instantiating agent: {e}"
        print(msg)
        return msg

    # Process each question separately
    try:
        for i, question_item in enumerate(my_questions, 1):
            # Use .get() for safe access (returns None if key doesn't exist)
            question_text = question_item.get("question")
            file_name = question_item.get("file_name")

            if not question_text:
                log_output.append(f"\nQuestion {i}: [ERROR] Missing question text")
                continue

            log_output.append(f"\nQuestion {i}: {question_text}")
            if file_name:
                log_output.append(f"File: {file_name}")

            # Run agent with question and optional file_name
            my_answer = my_agent(question_text, file_name=file_name)

            print(f"Question: {question_text} Answer: {my_answer}")
            log_output.append(f"Answer: {my_answer}")
            results_to_verify.append((question_text, my_answer))
    except Exception as e:
            error_msg = f"Error running agent on task: {e}"
            print(error_msg)
            log_output.append(error_msg)

    log_output.append("\n=== Completed Example Questions ===")
    verify_answers(results_to_verify, log_output)
    return pd.DataFrame(log_output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the agent application.")
    parser.add_argument("--test", action="store_true", help="Run local tests only and exit.")
    args = parser.parse_args()

    print("\n" + "-"*30 + " App Starting " + "-"*30)
    # Check for SPACE_HOST and SPACE_ID at startup for information
    space_host_startup = os.getenv("SPACE_HOST")
    space_id_startup = os.getenv("SPACE_ID") # Get SPACE_ID at startup

    if space_host_startup:
        print(f"[OK] SPACE_HOST found: {space_host_startup}")
        print(f"   Runtime URL should be: https://{space_host_startup}.hf.space")
    else:
        print("[INFO] SPACE_HOST environment variable not found (running locally?).")

    if space_id_startup: # Print repo URLs if SPACE_ID is found
        print(f"[OK] SPACE_ID found: {space_id_startup}")
        print(f"   Repo URL: https://huggingface.co/spaces/{space_id_startup}")
        print(f"   Repo Tree URL: https://huggingface.co/spaces/{space_id_startup}/tree/main")
    else:
        print("[INFO] SPACE_ID environment variable not found (running locally?). Repo URL cannot be determined.")

    print("-"*(60 + len(" App Starting ")) + "\n")

    if args.test and not space_id_startup:
        print("Running test code (CLI mode)...")
        result = run_test_code()
        if isinstance(result, pd.DataFrame):
            # Print DataFrame content without truncation
            pd.set_option('display.max_colwidth', None)
            pd.set_option('display.max_rows', None)
            # Iterate and print each row's content to ensure clean text output
            for col in result.columns:
                 for val in result[col]:
                     print(val)
        else:
            print(result)
        sys.exit(0)

    print("Launching Gradio Interface for Basic Agent Evaluation...")
    grTestApp = create_ui(run_and_submit_all, run_test_code)
    grTestApp.launch()
