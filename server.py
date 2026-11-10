# Live frontend for the data science agent.  Local:  python server.py   then open http://localhost:8000
import os, glob, json, time, threading, uuid, hmac
from flask import Flask, request, jsonify, send_from_directory, abort
from werkzeug.utils import secure_filename
import agent

app = Flask(__name__, static_folder="static")
app.config["MAX_CONTENT_LENGTH"] = 25 * 1024 * 1024
DATA_DIR = os.environ.get("DATA_DIR", "./data")
PASSCODE = os.environ.get("APP_PASSCODE", "")
RUNS, SESSIONS = {}, {}

for p in sorted(glob.glob(os.path.join(agent.TRACE_DIR, "*.json"))):       # earlier runs show up as history
    r = json.load(open(p))
    r.setdefault("id", os.path.basename(p)[:-5]); r.setdefault("status", "done"); r.setdefault("session_id", None)
    r.setdefault("activity", ""); r.setdefault("error", None)
    r.setdefault("secs", round(sum(q["secs"] for q in r.get("requests", [])), 1))
    RUNS[r["id"]] = r


@app.before_request
def check_passcode():
    if PASSCODE and request.path.startswith("/api/"):
        if not hmac.compare_digest(request.headers.get("X-Passcode", ""), PASSCODE):
            return jsonify(error="passcode"), 401


def summary(r):
    return {"id": r["id"], "question": r["question"], "dataset": r["dataset"], "status": r["status"], "started": r.get("started")}


@app.get("/")
def index():
    return send_from_directory("static", "index.html")


@app.get("/api/state")
def state():
    files = sorted(f for f in os.listdir(DATA_DIR) if not f.startswith("."))
    return jsonify(datasets=files, runs=[summary(r) for r in RUNS.values()], model=agent.MODEL, advisor=agent.ADVISOR, effort=agent.EFFORT,
                   prices=agent.PRICES)


@app.post("/api/ask")
def ask():
    question = (request.form.get("question") or "").strip()
    if not question:
        return jsonify(error="Type a question first."), 400
    sid = request.form.get("session_id") or ""
    if sid in SESSIONS and not any(r["status"] == "running" and r["session_id"] == sid for r in RUNS.values()):
        session = SESSIONS[sid]                                             # follow-up: same conversation, same sandbox
    else:
        up = request.files.get("file")
        if up and up.filename:
            name = secure_filename(up.filename) or "upload.csv"
            up.save(os.path.join(DATA_DIR, name))
        else:
            name = secure_filename(request.form.get("dataset") or "")
        path = os.path.join(DATA_DIR, name)
        if not name or not os.path.isfile(path):
            return jsonify(error="Pick a dataset or upload a file."), 400
        try:
            session = agent.new_session(path)
        except Exception as e:
            return jsonify(error=f"Could not upload the dataset to the API: {e}"), 502
        sid = uuid.uuid4().hex[:12]
        SESSIONS[sid] = session
    run_id = time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
    run = agent.new_run(run_id, sid, session, question)
    RUNS[run_id] = run
    threading.Thread(target=agent.run_turn, args=(run, session, question), daemon=True).start()
    return jsonify(run_id=run_id)


@app.get("/api/runs/<run_id>")
def get_run(run_id):
    r = RUNS.get(run_id) or abort(404)
    after = int(request.args.get("after", 0))
    out = {k: v for k, v in r.items() if k not in ("events", "t0")}
    out["events"] = r["events"][after:]
    out["total"] = len(r["events"])
    out["elapsed"] = round(time.time() - r["t0"], 1) if r["status"] == "running" else r.get("secs")
    out["session_alive"] = r.get("session_id") in SESSIONS
    return jsonify(out)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8000)), threaded=True)
