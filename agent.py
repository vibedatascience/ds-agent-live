# Data science agent core. Built from the cwc-workshops building blocks:
# code execution + Files API (01), advisor (04), effort (05), memory + context editing (06).
import anthropic, os, time, base64, json, tempfile, mimetypes, traceback
from anthropic.tools import BetaLocalFilesystemMemoryTool

MODEL = os.environ.get("AGENT_MODEL", "claude-sonnet-5-5")
ADVISOR = os.environ.get("AGENT_ADVISOR", "claude-opus-5-5")
EFFORT = os.environ.get("AGENT_EFFORT", "high")
MEMORY_DIR = os.environ.get("MEMORY_DIR", "./ds_memories")
TRACE_DIR = os.environ.get("TRACE_DIR", "./traces")

SYSTEM = """You are a senior data scientist. The user's dataset is mounted in the sandbox at $INPUT_DIR.

How you work:
1. First view /memories. If there are notes on this dataset (schema, quirks, past findings), use them and skip re-profiling.
2. Otherwise profile the data before analysis: shape, dtypes, nulls, duplicates, value ranges, sample rows.
3. Answer the question with code, never by guessing. Check segments before trusting a top-line number (Simpson's paradox,
   sample ratio mismatch, novelty effects). Report effect sizes with confidence intervals, not only p-values.
4. Before the final answer on any causal or statistical claim, consult the advisor once on whether the method is sound.
5. Make exactly ONE chart per answer, the single most insightful one. Save it to $OUTPUT_DIR as a PNG. Style: white background,
   large figure (14x8), small fonts so labels never overlap, palette #E60023 (key series), #0a3069, #0d9488, #64748b,
   legend outside the plot area, value label at the last point of every line, generous space between title and subtitle.
6. Before finishing, save to /memories/<dataset_name>.md: the schema, data quirks, and each finding with its date. Update, don't duplicate.

Final answer format: the answer in one sentence first, then a markdown table of the key numbers, then a table of 5 sample
rows if the data was new to you, then caveats. Plain words. No filler."""

client = anthropic.Anthropic()
memory = BetaLocalFilesystemMemoryTool(base_path=MEMORY_DIR)

TOOLS = [{"type": "code_execution_20260521", "name": "code_execution"}, memory.to_dict()]
BETAS = ["files-api-2025-04-14", "context-management-2025-06-27"]
if ADVISOR:
    TOOLS.append({"type": "advisor_20260301", "name": "advisor", "model": ADVISOR, "max_uses": 2, "max_tokens": 1500})
    BETAS.append("advisor-tool-2026-03-01")

ACTIVITY = {"server_tool_use": "writing code", "bash_code_execution_tool_result": "running code in the sandbox",
            "text_editor_code_execution_tool_result": "editing a file in the sandbox", "code_execution_tool_result": "running code in the sandbox",
            "tool_use": "using memory", "text": "writing", "thinking": "thinking", "redacted_thinking": "thinking",
            "advisor_tool_result": "reading the advisor's reply"}


def memory_files():
    return {os.path.relpath(os.path.join(d, f), MEMORY_DIR): open(os.path.join(d, f), errors="replace").read()
            for d, _, fs in os.walk(MEMORY_DIR) for f in fs}


def new_session(path):
    """One session = one dataset, one conversation, one sandbox container."""
    name = os.path.basename(path)
    mime = mimetypes.guess_type(name)[0] or "text/csv"
    upload = client.beta.files.upload(file=(name, open(path, "rb").read(), mime))
    with open(path, errors="replace") as f:
        header = f.readline().strip()[:500]
        rows = sum(1 for _ in f)
    return {"path": path, "name": name, "file_id": upload.id, "messages": [], "container": None, "seen": set(),
            "rows": rows, "header": header, "bytes": os.path.getsize(path)}


def new_run(run_id, session_id, session, question):
    return {"id": run_id, "session_id": session_id, "dataset": session["name"], "question": question, "status": "running",
            "activity": "starting", "events": [], "requests": [], "model": MODEL, "advisor": ADVISOR, "effort": EFFORT,
            "system": SYSTEM, "started": time.strftime("%Y-%m-%d %H:%M:%S"), "t0": time.time(), "rows": session["rows"],
            "header": session["header"], "bytes": session["bytes"], "memory_files": {}, "error": None,
            "followup": bool(session["messages"])}


def _ev(run, kind, **kw):
    run["events"].append({"kind": kind, "req": len(run["requests"]) + 1, "t": round(time.time() - run["t0"], 1), **kw})


def _handle(block, run, session):
    t = block.type
    if t == "server_tool_use" and block.name == "advisor":
        _ev(run, "advisor_call", model=ADVISOR)
    elif t == "advisor_tool_result":
        _ev(run, "advisor_result", result_type=block.content.type, text=getattr(block.content, "text", None))
    elif t == "thinking":
        if getattr(block, "thinking", ""):
            _ev(run, "thinking", text=block.thinking)
    elif t == "server_tool_use":
        code = block.input.get("code") or block.input.get("command") or json.dumps(block.input, indent=1)
        _ev(run, "code", tool=block.name, code=code)
    elif t.endswith("code_execution_tool_result"):
        c, files = block.content, []
        for item in getattr(c, "content", None) or []:
            fid = getattr(item, "file_id", None)
            if fid and fid not in session["seen"]:
                session["seen"].add(fid)
                try:
                    name = os.path.basename(client.beta.files.retrieve_metadata(fid).filename)
                    with tempfile.TemporaryDirectory() as d:
                        p = os.path.join(d, "f")
                        client.beta.files.download(fid).write_to_file(p)
                        data = open(p, "rb").read()
                    if len(data) < 4_000_000:
                        files.append({"name": name, "b64": base64.b64encode(data).decode()})
                    else:
                        files.append({"name": name + " (too large to show)", "b64": ""})
                except Exception as e:
                    files.append({"name": f"file {fid} (download failed: {e})", "b64": ""})
        _ev(run, "output", stdout=(getattr(c, "stdout", "") or "")[:60000], stderr=(getattr(c, "stderr", "") or "")[:20000],
            rc=getattr(c, "return_code", None), error=getattr(c, "error_code", None), files=files)
    elif t == "tool_use":
        _ev(run, "memory", id=block.id, input=block.input)
    elif t == "text" and block.text.strip():
        _ev(run, "text", text=block.text.strip())


def run_turn(run, session, question):
    """Runs one question to completion, appending events to run["events"] as they happen."""
    keep = len(session["messages"])
    try:
        if not session["messages"]:
            session["messages"].append({"role": "user", "content": [
                {"type": "text", "text": f"Dataset: {session['name']}\n\n{question}"},
                {"type": "container_upload", "file_id": session["file_id"]}]})
        else:
            session["messages"].append({"role": "user", "content": question})
        while True:
            kw = {"container": session["container"]} if session["container"] else {}
            t0 = time.time()
            run["activity"] = "waiting for the model"
            with client.beta.messages.stream(
                    model=MODEL, max_tokens=16000, system=SYSTEM, messages=session["messages"], tools=TOOLS, betas=BETAS,
                    output_config={"effort": EFFORT},
                    context_management={"edits": [{"type": "clear_tool_uses_20250919",
                                                   "trigger": {"type": "input_tokens", "value": 60_000},
                                                   "keep": {"type": "tool_uses", "value": 4},
                                                   "exclude_tools": ["memory"]}]},
                    **kw) as stream:
                for event in stream:
                    if event.type == "content_block_start":
                        b = event.content_block
                        run["activity"] = ("asking the advisor" if getattr(b, "name", "") == "advisor"
                                           else ACTIVITY.get(b.type, b.type))
                    elif event.type == "content_block_stop":
                        _handle(event.content_block, run, session)
                        if event.content_block.type == "server_tool_use":
                            run["activity"] = ("waiting for the advisor" if event.content_block.name == "advisor"
                                               else "running code in the sandbox")
                resp = stream.get_final_message()
            if resp.container:
                session["container"] = resp.container.id
            run["requests"].append({
                "secs": round(time.time() - t0, 1), "in": resp.usage.input_tokens, "out": resp.usage.output_tokens,
                "stop": resp.stop_reason,
                "per_model": [{"model": getattr(i, "model", None) or resp.model, "in": i.input_tokens, "out": i.output_tokens}
                              for i in (getattr(resp.usage, "iterations", None) or [])]})
            session["messages"].append({"role": "assistant", "content": resp.content})
            if resp.stop_reason == "tool_use":                            # memory is client-side: run it here
                run["activity"] = "using memory"
                results = []
                for b in resp.content:
                    if b.type == "tool_use":
                        try:
                            out, bad = memory.call(b.input), False
                        except Exception as e:                            # e.g. create on a file that exists: tell the model, don't crash
                            out, bad = f"Error: {e}", True
                        results.append({"type": "tool_result", "tool_use_id": b.id, "content": out, "is_error": bad})
                        for e in run["events"]:
                            if e.get("id") == b.id:
                                e["result"] = out if isinstance(out, str) else str(out)
                session["messages"].append({"role": "user", "content": results})
                run["memory_files"] = memory_files()
            elif resp.stop_reason != "pause_turn":
                break
        run["status"] = "done"
    except Exception as e:
        run["status"], run["error"] = "error", f"{type(e).__name__}: {e}"
        del session["messages"][keep:]                                    # leave the conversation usable for the next question
        traceback.print_exc()
    run["activity"] = ""
    run["memory_files"] = memory_files()
    run["secs"] = round(time.time() - run["t0"], 1)
    os.makedirs(TRACE_DIR, exist_ok=True)
    json.dump({k: v for k, v in run.items() if k != "t0"}, open(os.path.join(TRACE_DIR, run["id"] + ".json"), "w"))
