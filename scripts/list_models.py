"""
Script to list accessible Gemini models for the configured GEMINI_API_KEY.

Security Notice:
This script NEVER prints or logs the GEMINI_API_KEY.
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Ensure costgate root is in sys.path
root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))

# Load .env file from root
env_file = root_dir / ".env"
if env_file.exists():
    load_dotenv(dotenv_path=env_file)
else:
    load_dotenv()

api_key = os.getenv("GEMINI_API_KEY", "").strip()

if not api_key or api_key == "your_gemini_api_key_here":
    print("=" * 60)
    print("ERROR: GEMINI_API_KEY is not set in your .env file.")
    print(f"Please create/update {env_file} with your API key:")
    print("GEMINI_API_KEY=AIzaSy...")
    print("=" * 60)
    sys.exit(1)

try:
    from google import genai
except ImportError:
    print("ERROR: 'google-genai' is not installed.")
    print("Run: pip install google-genai")
    sys.exit(1)

print("=" * 70)
print("Querying Google Gemini API for available models...")
print("=" * 70)

try:
    client = genai.Client(api_key=api_key)
    models = list(client.models.list())

    if not models:
        print("No models returned for this API key.")
        sys.exit(0)

    print(f"Found {len(models)} accessible models:\n")
    print(f"{'MODEL ID':<35} {'SUPPORTED ACTIONS':<30}")
    print("-" * 70)

    for m in models:
        # Strip 'models/' prefix for standard clean naming if present
        model_id = m.name.removeprefix("models/") if m.name else "unknown"
        actions = ", ".join(m.supported_actions or []) if m.supported_actions else "n/a"

        # Focus on content generation and embedding models
        print(f"{model_id:<35} {actions:<30}")

    print("\n" + "=" * 70)
    print("Done. Copy and paste the list above here.")
    print("=" * 70)

except Exception as err:
    # Clean error output without leaking sensitive data
    err_str = str(err)
    if api_key in err_str:
        err_str = err_str.replace(api_key, "***MASKED***")
    print("=" * 70)
    print(f"Failed to fetch models: {err_str}")
    print("=" * 70)
    sys.exit(1)
