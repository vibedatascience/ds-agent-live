# Live data science agent

Upload a CSV, ask a question, and watch the agent work step by step: the code it writes, the output, advisor calls, memory, the chart, and token use.

Built from the Claude API building blocks: code execution + Files API, advisor, effort, memory + context editing.

## Run locally
    pip install -r requirements.txt
    export ANTHROPIC_API_KEY=...
    export APP_PASSCODE=...        # optional; the page asks for it
    python server.py               # http://localhost:8000

## Deploy (Render, Python runtime)
- Build: `pip install -r requirements.txt`
- Start: `gunicorn -w 1 --threads 16 -t 0 server:app`
- Env: `ANTHROPIC_API_KEY`, `APP_PASSCODE`

Use one worker: runs and sessions live in memory. Traces and memory files are on local disk and reset on redeploy.
