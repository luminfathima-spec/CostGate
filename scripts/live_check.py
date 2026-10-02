"""
Execute exactly ONE live check against the running server for Phase 1 verification.
Never loops or runs batch calls. Protects free-tier quota.
"""

import subprocess
import time
import sys
import json
import sqlite3
import urllib.request
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))

# 1. Start uvicorn server in subprocess
print("Starting CostGate FastAPI server on 127.0.0.1:8000...")
proc = subprocess.Popen(
    [sys.executable, "-m", "uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", "8000"],
    cwd=str(root_dir),
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True,
)

try:
    # 2. Wait for server readiness via /health
    ready = False
    for _ in range(20):
        time.sleep(0.5)
        try:
            with urllib.request.urlopen("http://127.0.0.1:8000/health", timeout=2) as resp:
                if resp.status == 200:
                    ready = True
                    break
        except Exception:
            pass

    if not ready:
        print("ERROR: Server failed to start within 10 seconds.")
        stdout, stderr = proc.communicate(timeout=2)
        print("STDOUT:", stdout)
        print("STDERR:", stderr)
        sys.exit(1)

    print("Server is healthy! Sending single live question to POST /ask...")

    # 3. Send single live request
    req_data = json.dumps({"question": "What is the capital of France?"}).encode("utf-8")
    req = urllib.request.Request(
        "http://127.0.0.1:8000/ask",
        data=req_data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with urllib.request.urlopen(req, timeout=30) as resp:
        resp_body = resp.read().decode("utf-8")
        status_code = resp.status
        result_json = json.loads(resp_body)

    print("\n" + "=" * 70)
    print("LIVE HTTP RESPONSE (Status:", status_code, ")")
    print("=" * 70)
    print(json.dumps(result_json, indent=2))

    # 4. Read database row from costgate.db
    db_path = root_dir / "costgate.db"
    print("\n" + "=" * 70)
    print(f"DATABASE AUDIT ROW in {db_path.name}")
    print("=" * 70)
    if db_path.exists():
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        cur.execute("SELECT * FROM requests ORDER BY id DESC LIMIT 1")
        row = cur.fetchone()
        if row:
            for k in row.keys():
                print(f"  {k:<16}: {row[k]}")
        else:
            print("No rows found in requests table.")
        conn.close()
    else:
        print(f"Database file {db_path} not found.")

    print("=" * 70)

finally:
    # 5. Terminate server
    print("\nStopping uvicorn server...")
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
    print("Server stopped cleanly.")
