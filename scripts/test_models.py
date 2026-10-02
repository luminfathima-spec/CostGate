"""
Test script to make a tiny call to candidate models using the user's GEMINI_API_KEY.
Reports success/error and detailed token usage (input, output, thinking tokens).
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))

env_file = root_dir / ".env"
load_dotenv(dotenv_path=env_file)
api_key = os.getenv("GEMINI_API_KEY", "").strip()

if not api_key:
    print("Error: GEMINI_API_KEY is not set.")
    sys.exit(1)

from google import genai
from google.genai import types

client = genai.Client(api_key=api_key)

models_to_test = [
    ("small", "gemini-2.5-flash-lite", "generate"),
    ("medium", "gemini-2.5-flash", "generate"),
    ("large", "gemini-2.5-pro", "generate"),
    ("embedding", "gemini-embedding-001", "embed"),
]

results = []

for tier, model_id, action in models_to_test:
    print(f"\nTesting {tier} tier ({model_id})...")
    try:
        if action == "generate":
            response = client.models.generate_content(
                model=model_id,
                contents="Hello",
                config=types.GenerateContentConfig(
                    max_output_tokens=20,
                ),
            )
            usage = response.usage_metadata
            input_tokens = getattr(usage, "prompt_token_count", 0)
            output_tokens = getattr(usage, "candidates_token_count", 0)
            
            # Check thinking/reasoning tokens
            thinking_tokens = 0
            if hasattr(usage, "candidates_tokens_details") and usage.candidates_tokens_details:
                for detail in usage.candidates_tokens_details:
                    modality = getattr(detail, "modality", "")
                    token_count = getattr(detail, "token_count", 0)
                    # Some versions report reasoning / thinking tokens
                    if "THINKING" in str(modality).upper() or "REASONING" in str(modality).upper():
                        thinking_tokens += token_count

            # Also check if usage has thinking_token_count or similar
            if hasattr(usage, "thinking_token_count") and usage.thinking_token_count:
                thinking_tokens = usage.thinking_token_count

            print(f"SUCCESS: {model_id}")
            print(f"  Input Tokens: {input_tokens}")
            print(f"  Output Tokens: {output_tokens}")
            print(f"  Thinking Tokens: {thinking_tokens}")
            print(f"  Answer snippet: {repr(response.text.strip() if response.text else '')}")
            results.append({
                "tier": tier,
                "model": model_id,
                "status": "SUCCESS",
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "thinking_tokens": thinking_tokens,
                "error": None,
            })
        elif action == "embed":
            embed_resp = client.models.embed_content(
                model=model_id,
                contents="Hello",
            )
            # Embedding usage
            embedding_dim = len(embed_resp.embedding.values) if embed_resp.embedding else 0
            print(f"SUCCESS: {model_id}")
            print(f"  Embedding dimensions: {embedding_dim}")
            results.append({
                "tier": tier,
                "model": model_id,
                "status": "SUCCESS",
                "input_tokens": 1,
                "output_tokens": 0,
                "thinking_tokens": 0,
                "error": None,
            })
    except Exception as e:
        err_msg = str(e)
        if api_key in err_msg:
            err_msg = err_msg.replace(api_key, "***MASKED***")
        print(f"FAILED: {model_id} -> {err_msg}")
        results.append({
            "tier": tier,
            "model": model_id,
            "status": "FAILED",
            "input_tokens": 0,
            "output_tokens": 0,
            "thinking_tokens": 0,
            "error": err_msg,
        })
