import os
import time
import random
import logging
from openai import OpenAI
from google import genai

logger = logging.getLogger(__name__)

_openai_client = None
_gemini_client = None


def get_openai_client():
    global _openai_client
    if _openai_client is None:
        _openai_client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_BASE_URL"),
        )
    return _openai_client


def get_gemini_client():
    global _gemini_client
    if _gemini_client is None:
        _gemini_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    return _gemini_client


def is_retryable_error(exc: Exception) -> bool:
    msg = str(exc).lower()

    retry_signals = [
        "429",
        "rate limit",
        "quota",
        "resource exhausted",
        "temporarily unavailable",
        "timed out",
        "timeout",
        "connection reset",
        "internal error",
        "503",
        "500",
        "unavailable",
        "deadline exceeded",
    ]

    return any(sig in msg for sig in retry_signals)


def call_llm_with_retry(
    prompt: str,
    max_retries: int = 5,
    base_delay: float = 3.0,
    max_delay: float = 30.0,
):
    """
    Shared retry wrapper for Gemini/OpenAI-compatible calls.
    Uses exponential backoff with jitter.
    """
    use_gemini = os.getenv("USE_GEMINI", "false").lower() == "true"
    model_name = os.getenv("OPENAI_MODEL")

    last_error = None

    for attempt in range(max_retries + 1):
        try:
            logger.info(
                f"LLM call attempt {attempt + 1}/{max_retries + 1} | "
                f"use_gemini={use_gemini} | model={model_name}"
            )

            if use_gemini:
                client = get_gemini_client()
                response = client.models.generate_content(
                    model=model_name,
                    contents=prompt
                )
                content = getattr(response, "text", None)

            else:
                client = get_openai_client()
                response = client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0,
                    timeout=90
                )
                content = response.choices[0].message.content

            if not content or not str(content).strip():
                raise ValueError("Empty response from LLM")

            return content

        except Exception as e:
            last_error = e
            logger.warning(f"LLM call failed on attempt {attempt + 1}: {e}")

            if attempt == max_retries or not is_retryable_error(e):
                break

            delay = min(base_delay * (2 ** attempt), max_delay)
            jitter = random.uniform(0, 1.0)
            sleep_time = delay + jitter

            logger.info(f"Retrying after {sleep_time:.2f} seconds...")
            time.sleep(sleep_time)

    logger.error(f"LLM call failed after retries: {last_error}")
    return None