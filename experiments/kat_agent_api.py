"""Floor test through an OpenAI-compatible chat API (DeepSeek's official API by default), in two modes:

  single : the floor image + the teaching set in one request; the model writes SONUC_<floor>.json directly.
  agent  : the model gets the tools 6.1 Sol built for itself on this test (kat_test/015_sol_sifir):
             view(x0, y0, x1, y1, grid)   enlarged crop of the floor with a coordinate grid (floor pixel frame)
             scan(axis, at, start, end)   dark-pixel runs along one row or column (wall faces, jambs)
             check(rooms)                 the draft polygons drawn over the floor + geometric problems
             submit(result)               the final answer (same JSON as GOREV_<floor>.md)
           Images are only allowed in user messages, so every image a tool produces follows its tool message
           as a user message.

The API key is read from the environment variable DEEPSEEK_API_KEY (set by the user). Nothing else is sent:
only the floor image, the teaching sheets and TEACH.md.

usage: python kat_agent_api.py single|agent [--floor 015] [--steps 60] [--effort high]
       -> kat_test/<floor>_deepseek_<mode>/SONUC_<floor>.json, transcript.json, then run kat_score.py on it
"""
import sys as _sys; from pathlib import Path as _P; _sys.path.insert(0, str(_P(__file__).resolve().parents[1]))
import config; config.setup()  # noqa: E402  repo paths: data/model roots come from config.local.json or env
import argparse
import base64
import io
import json
import os
import subprocess
import time
import urllib.request
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from shapely.geometry import Polygon

import ds_agent2 as A2

HERE = config.DATA
TEST = HERE / "kat_test"
URL = "https://api.deepseek.com/chat/completions"
MODEL = "deepseek-flash"


def b64(img, fmt="PNG"):
    b = io.BytesIO(); img.save(b, fmt)
    return f"data:image/{fmt.lower()};base64," + base64.b64encode(b.getvalue()).decode()


def img_part(img, detail="high"):
    return {"type": "image_url", "image_url": {"url": b64(img), "detail": detail}}


def api_key(var="DEEPSEEK_API_KEY"):
    """The key from the environment, or from the user's saved variables (what `setx` writes), so a key set
    after this program's parent process started is still found."""
    key = os.environ.get(var)
    if not key and os.name == "nt":
        try:
            import winreg
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as k:
                key = winreg.QueryValueEx(k, var)[0]
        except OSError:
            key = None
    return (key or "").strip() or None


BACKEND = {"name": "deepseek", "url": URL, "think": True}


def call(messages, effort, tools=None, max_tokens=16000, think_budget=None):
    if BACKEND["name"] == "local":
        # llama-server (Qwen3.8-27B): OpenAI format, no key; thinking through the chat template
        body = {"messages": messages, "max_tokens": max_tokens, "temperature": 0.6 if BACKEND["think"] else 0.2,
                "chat_template_kwargs": {"enable_thinking": BACKEND["think"]}}
        if BACKEND["think"] and think_budget:   # llama-server ends the thinking cleanly after this many tokens
            body["thinking_budget_tokens"] = int(think_budget)
            body["max_tokens"] = int(think_budget) + 2500
        headers = {"Content-Type": "application/json"}
    elif BACKEND["name"] == "dashscope":
        # Alibaba Cloud Model Studio, international (Singapore) OpenAI-compatible endpoint
        key = api_key("DASHSCOPE_API_KEY")
        if not key:
            raise SystemExit("DASHSCOPE_API_KEY is not set")
        body = {"model": BACKEND["model"], "messages": messages, "max_tokens": max_tokens,
                "enable_thinking": BACKEND["think"]}
        if BACKEND["model"].startswith("qwen"):
            body["thinking_budget"] = max(1024, max_tokens - 6000)   # leave room for the answer itself
        headers = {"Content-Type": "application/json", "Authorization": "Bearer " + key}
    else:
        key = api_key()
        if not key:
            raise SystemExit("DEEPSEEK_API_KEY is not set (run: setx DEEPSEEK_API_KEY \"...\" in your own terminal, then open a new one)")
        body = {"model": MODEL, "messages": messages, "max_tokens": max_tokens,
                "thinking": {"type": "enabled"}, "reasoning_effort": effort}
        headers = {"Content-Type": "application/json", "Authorization": "Bearer " + key}
    if tools:
        body["tools"] = tools
    if BACKEND["name"] == "dashscope":
        return _call_stream(body, headers)
    req = urllib.request.Request(BACKEND["url"], json.dumps(body).encode(), headers)
    for attempt in range(4):
        try:
            return json.load(urllib.request.urlopen(req, timeout=1800))
        except urllib.error.HTTPError as e:
            msg = e.read().decode(errors="replace")[:800]
            if e.code in (429, 500, 502, 503) and attempt < 3:
                time.sleep(10 * (attempt + 1)); continue
            raise SystemExit(f"API error {e.code}: {msg}")


def _call_stream(body, headers):
    """Streaming request (server-sent events), rebuilt into the non-streaming response shape. Long thinking
    answers (30+ minutes) otherwise hit the socket read timeout and are lost."""
    body = dict(body, stream=True, stream_options={"include_usage": True})
    req = urllib.request.Request(BACKEND["url"], json.dumps(body).encode(), headers)
    for attempt in range(3):
        try:
            resp = urllib.request.urlopen(req, timeout=300)   # timeout per read, not for the whole answer
            break
        except urllib.error.HTTPError as e:
            msg = e.read().decode(errors="replace")[:800]
            if e.code in (429, 500, 502, 503) and attempt < 2:
                time.sleep(15 * (attempt + 1)); continue
            raise SystemExit(f"API error {e.code}: {msg}")
    content, reasoning, calls, usage, finish = [], [], {}, {}, None
    for raw in resp:
        line = raw.decode("utf-8", errors="replace").strip()
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if data == "[DONE]":
            break
        try:
            ev = json.loads(data)
        except Exception:
            continue
        if ev.get("usage"):
            usage = ev["usage"]
        for ch in ev.get("choices") or []:
            d = ch.get("delta") or {}
            if d.get("content"):
                content.append(d["content"])
            if d.get("reasoning_content"):
                reasoning.append(d["reasoning_content"])
            for tc in d.get("tool_calls") or []:
                c = calls.setdefault(tc.get("index", 0), {"id": "", "type": "function", "function": {"name": "", "arguments": ""}})
                if tc.get("id"):
                    c["id"] = tc["id"]
                f = tc.get("function") or {}
                c["function"]["name"] += f.get("name") or ""
                c["function"]["arguments"] += f.get("arguments") or ""
            if ch.get("finish_reason"):
                finish = ch["finish_reason"]
    msg = {"role": "assistant", "content": "".join(content), "reasoning_content": "".join(reasoning)}
    if calls:
        msg["tool_calls"] = [calls[i] for i in sorted(calls)]
    return {"choices": [{"message": msg, "finish_reason": finish}], "usage": usage}


# ---------------- tools (same ideas as Sol's crop.ps1 / pixel scans / overlay check) ----------------
class Floor:
    def __init__(self, path):
        self.img = Image.open(path).convert("RGB")
        self.gray = self.img.convert("L")
        self.font = ImageFont.load_default()

    def view(self, x0, y0, x1, y1, grid=10):
        W, H = self.img.size
        x0, y0, x1, y1 = max(0, int(x0)), max(0, int(y0)), min(W, int(x1)), min(H, int(y1))
        if x1 - x0 < 8 or y1 - y0 < 8:
            return None, "region too small"
        reg = self.img.crop((x0, y0, x1, y1))
        s = min(1200 / max(reg.size), 8.0)
        reg = reg.resize((max(1, int(reg.width * s)), max(1, int(reg.height * s))),
                         Image.Resampling.NEAREST if s >= 2 else Image.Resampling.LANCZOS)
        L, T = 44, 18
        out = Image.new("RGB", (reg.width + L, reg.height + T), "white")
        out.paste(reg, (L, T))
        d = ImageDraw.Draw(out, "RGBA")
        g = max(1, int(grid or 0))
        if grid:
            major = g * 5
            for v in range((x0 // g + 1) * g, x1, g):
                X = L + (v - x0) * s
                d.line([(X, T), (X, out.height)], fill=(255, 0, 0, 110 if v % major == 0 else 45), width=1)
                if v % major == 0:
                    d.text((X + 1, 2), str(v), fill=(200, 0, 0), font=self.font)
            for v in range((y0 // g + 1) * g, y1, g):
                Y = T + (v - y0) * s
                d.line([(L, Y), (out.width, Y)], fill=(255, 0, 0, 110 if v % major == 0 else 45), width=1)
                if v % major == 0:
                    d.text((2, Y - 5), str(v), fill=(200, 0, 0), font=self.font)
        return out, f"view x {x0}-{x1}, y {y0}-{y1} (floor pixels), enlarged x{s:.2f}, grid every {g} px, labels every {g * 5} px"

    def scan(self, axis, at, start, end, dark=150):
        W, H = self.gray.size
        px = self.gray.load()
        runs, run = [], None
        rng = range(max(0, int(start)), min(W if axis == "row" else H, int(end)) + 1)
        for v in rng:
            x, y = (v, int(at)) if axis == "row" else (int(at), v)
            if not (0 <= x < W and 0 <= y < H):
                continue
            isd = px[x, y] < dark
            if isd and run is None:
                run = v
            if (not isd or v == rng[-1]) and run is not None:
                runs.append([run, v - 1 if not isd else v]); run = None
        what = f"y={at}, x {start}-{end}" if axis == "row" else f"x={at}, y {start}-{end}"
        return f"dark runs along {axis} {what} (start-end, floor pixels): " + json.dumps(runs)

    def check(self, rooms):
        ov = self.img.copy(); d = ImageDraw.Draw(ov)
        probs, polys = [], []
        for r in rooms:
            pts = r.get("boundary") or []
            if len(pts) < 3:
                continue
            q = [tuple(map(float, p)) for p in pts]
            d.line(q + q[:1], fill=(210, 0, 210), width=3)
            d.text(q[0], str(r.get("id", "")), fill=(0, 0, 255), font=self.font)
            P = Polygon(q)
            if not P.is_valid:
                probs.append(f"{r.get('id')}: polygon crosses itself")
            polys.append((r.get("id"), P.buffer(0)))
        for i in range(len(polys)):
            for j in range(i + 1, len(polys)):
                a, b = polys[i][1], polys[j][1]
                if a.intersects(b) and a.intersection(b).area > 30:
                    probs.append(f"{polys[i][0]} and {polys[j][0]} overlap by {a.intersection(b).area:.0f} px²")
        ov.thumbnail((1300, 1300))
        return ov, "draft drawn in magenta over the floor. problems: " + ("; ".join(probs) if probs else "none")


TOOLS = [
    {"type": "function", "function": {"name": "view", "description": "Enlarged crop of the floor image with a red coordinate grid in the floor's pixel frame (x right, y down). Use it to read exact corner, wall-face and jamb positions.",
     "parameters": {"type": "object", "properties": {"x0": {"type": "integer"}, "y0": {"type": "integer"}, "x1": {"type": "integer"}, "y1": {"type": "integer"},
                                                     "grid": {"type": "integer", "description": "grid spacing in floor pixels (10 for fine detail, 50 for overview, 0 for none)"}},
                    "required": ["x0", "y0", "x1", "y1"]}}},
    {"type": "function", "function": {"name": "scan", "description": "Dark-pixel runs along one row (axis=row, at=y) or one column (axis=col, at=x) between start and end. Each run is a drawn line; a wall shows as two runs (its two faces) or one thick run.",
     "parameters": {"type": "object", "properties": {"axis": {"type": "string", "enum": ["row", "col"]}, "at": {"type": "integer"}, "start": {"type": "integer"}, "end": {"type": "integer"}},
                    "required": ["axis", "at", "start", "end"]}}},
    {"type": "function", "function": {"name": "check", "description": "Draw draft room polygons over the floor and report self-crossings and overlaps.",
     "parameters": {"type": "object", "properties": {"rooms": {"type": "array", "items": {"type": "object", "properties": {"id": {"type": "string"}, "boundary": {"type": "array", "items": {"type": "array", "items": {"type": "number"}}}}}}},
                    "required": ["rooms"]}}},
    {"type": "function", "function": {"name": "submit", "description": "Submit the final answer: the JSON object described in the task (floor, image, rooms[]). Ends the task.",
     "parameters": {"type": "object", "properties": {"result": {"type": "object"}}, "required": ["result"]}}},
]

# --tools-from: the model's own workshop tools (scripts in tools/ + ARACLAR.md) and three tools to drive them.
WORK_TOOLS = [
    {"type": "function", "function": {"name": "run", "description": "Run one PowerShell command line in the test folder (your tools are in tools\\). `python` and `$py` are the Python with numpy/cv2/scipy/skimage/shapely/PIL. Returns stdout+stderr (truncated). Timeout 300 s.",
     "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}}},
    {"type": "function", "function": {"name": "look", "description": "Show an image file from the test folder (e.g. an overlay or crop your tools wrote), optionally only a region of it, downscaled to fit.",
     "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "x0": {"type": "integer"}, "y0": {"type": "integer"}, "x1": {"type": "integer"}, "y1": {"type": "integer"}},
                    "required": ["path"]}}},
    {"type": "function", "function": {"name": "write_file", "description": "Write a text file (e.g. a seeds JSON) inside the test folder.",
     "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]}}},
]
PY = r"C:\Users\Selim\Documents\Proje\.venv\Scripts\python.exe"


def inside(out, p):
    q = (out / p).resolve()
    if out.resolve() not in q.parents and q != out.resolve():
        raise ValueError("path outside the test folder")
    return q


def run_cmd(out, command):
    pre = f"$py = '{PY}'; function python {{ & $py @args }}; Set-Location '{out}'; "
    try:
        p = subprocess.run(["powershell", "-NoProfile", "-Command", pre + command], cwd=out, capture_output=True,
                           timeout=300, encoding="utf-8", errors="replace", env=dict(os.environ, PYTHONIOENCODING="utf-8"))
        s = (p.stdout or "") + (("\n[stderr]\n" + p.stderr) if p.stderr.strip() else "")
    except subprocess.TimeoutExpired:
        s = "timeout after 300 s"
    return s if len(s) <= 4000 else s[:2000] + f"\n… ({len(s) - 4000} chars cut) …\n" + s[-2000:]


def look(out, path, box):
    im = Image.open(inside(out, path)).convert("RGB")
    if box and all(v is not None for v in box):
        im = im.crop(tuple(box))
    im.thumbnail((1400, 1400))
    return im, f"{path} {'region ' + str(box) if box and all(v is not None for v in box) else ''} shown at {im.width}x{im.height}"


def teaching():
    intro, a, b, c, _how = A2.sections()
    return intro, a, b, c


def system_text(floor_task, mode):
    intro, a, b, c = teaching()
    how = ("HOW TO WORK: use the tools. Start with an overview, inventory every space of the floor, then go room by room: "
           "view each room enlarged with a fine grid, read every corner from the grid, confirm wall faces and door jambs "
           "with scan, draw drafts with check, fix them, and finally call submit once with all rooms. All coordinates are "
           "in the full floor image's pixel frame.") if mode == "agent" else (
           "HOW TO ANSWER: you have no tools. Read the floor image and answer ONLY with the JSON object described in the "
           "task. All coordinates are in the full floor image's pixel frame.")
    return "\n\n".join([intro, "SHEET A notes:\n" + a, "SHEET B notes:\n" + b, "SHEET C notes:\n" + c, how, "TASK:\n" + floor_task])


def first_user(floor_img):
    W, H = floor_img.size
    small = floor_img.copy(); small.thumbnail((1300, 1300))
    parts = [{"type": "text", "text": "Teaching sheet A (what the boundary follows):"}, img_part(Image.open(A2.TEACH / "SHEET_A.png").convert("RGB")),
             {"type": "text", "text": "Teaching sheet B (what is not boundary):"}, img_part(Image.open(A2.TEACH / "SHEET_B.png").convert("RGB")),
             {"type": "text", "text": "Teaching sheet C (solved rooms, green = correct boundary):"}, img_part(Image.open(A2.TEACH / "SHEET_C.png").convert("RGB")),
             {"type": "text", "text": f"The floor plan (shown downscaled; its real size is {W} x {H} px and ALL coordinates you give must be in that full frame):"},
             img_part(small)]
    return {"role": "user", "content": parts}


_CUT = [2]


def windowed_stable(msgs, jump=5, keep_steps=2):
    """Cache-friendly window for a local model: between cuts the conversation only grows at the end, so llama-server
    reuses its cache (a recurrent/hybrid model can only reuse an unchanged prefix). When more than `jump` steps have
    piled up after the cut, the cut moves forward in one go, keeping the last `keep_steps` steps. Everything before the
    cut is dropped - the platform's board in the latest tool result carries the full state."""
    asst = [i for i, m in enumerate(msgs) if m["role"] == "assistant" and i >= _CUT[0]]
    if len(asst) > jump:
        _CUT[0] = asst[-keep_steps]
    if _CUT[0] <= 2:
        return msgs
    note = {"role": "user", "content": "[Earlier steps were removed to save context. Your full state - plan, rooms, "
                                       "traces, recent changes - is on the BOARD at the end of the latest tool result.]"}
    return msgs[:2] + [note] + msgs[_CUT[0]:]


def windowed(msgs, keep):
    """Copy of the conversation in which only the last `keep` tool-image messages still carry their images
    (older ones become a short text note). The first user message (sheets + floor) is always kept whole.
    Needed for a local model whose context is far shorter than the API's."""
    if keep <= 0:
        return msgs
    idx = [i for i, m in enumerate(msgs) if i > 1 and m["role"] == "user" and isinstance(m["content"], list)
           and any(p.get("type") == "image_url" for p in m["content"])]
    old = set(idx[:-keep])
    tools_idx = [i for i, m in enumerate(msgs) if m["role"] == "tool"]
    old_tools = set(tools_idx[:-max(6, 2 * keep)])   # older tool results: keep only their first line
    asst_idx = [i for i, m in enumerate(msgs) if m["role"] == "assistant"]
    old_asst = set(asst_idx[:-3])                    # older assistant turns: drop their reasoning text
    out = []
    for i, m in enumerate(msgs):
        if i in old_tools and isinstance(m.get("content"), str) and len(m["content"]) > 300:
            out.append({**m, "content": m["content"].split("\n")[0][:300] + " [older tool output shortened]"})
        elif i in old_asst and m.get("reasoning_content"):
            out.append({k: v for k, v in m.items() if k != "reasoning_content"})
        elif i in old:
            out.append({"role": "user", "content": [p if p.get("type") != "image_url" else
                                                    {"type": "text", "text": "[older image removed to save context; view it again if needed]"}
                                                    for p in m["content"]]})
        else:
            out.append(m)
    return out


def parse_json(text):
    s, e = text.find("{"), text.rfind("}")
    return json.loads(text[s:e + 1]) if s >= 0 and e > s else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["single", "agent"])
    ap.add_argument("--floor", default="015")
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--effort", default="high")
    ap.add_argument("--max-tokens", dest="max_tokens", type=int, default=60000)
    ap.add_argument("--backend", choices=["deepseek", "local", "dashscope"], default="deepseek")
    ap.add_argument("--url", default="")
    ap.add_argument("--model", default="qwen3.8-max", help="dashscope: model id")
    ap.add_argument("--think", action="store_true", help="local: thinking on")
    ap.add_argument("--max-calls", dest="max_calls", type=int, default=0, help="tool calls executed per step (0 = all)")
    ap.add_argument("--max-images", dest="max_images", type=int, default=0, help="pictures sent back per step (0 = all)")
    ap.add_argument("--stable-window", dest="stable_window", action="store_true",
                    help="cache-friendly windowing (append-only, cut every few steps); for local models")
    ap.add_argument("--keep-images", dest="keep_images", type=int, default=0,
                    help="keep only the last N tool images in the context (0 = all)")
    ap.add_argument("--tag", default="")
    ap.add_argument("--tools-from", dest="tools_from", default="",
                    help="workshop folder: copy its tools\\*.py and ARACLAR.md in, add run/look/write_file tools")
    ap.add_argument("--token-budget", dest="token_budget", type=int, default=0,
                    help="stop before the next request once prompt+completion tokens reach this (0 = no limit)")
    ap.add_argument("--tool-module", dest="tool_module", default="",
                    help="python module with TOOLS, GUIDE and handle(name, args, ctx) -> (text, image|None, result|None); "
                         "replaces view/scan/check (submit stays)")
    ap.add_argument("--lessons", default="", help="text file of lessons from earlier runs, appended to the system prompt")
    ap.add_argument("--out", default="", help="output folder (default kat_test/<floor>_<tag>_<mode>)")
    a = ap.parse_args()
    if a.backend == "local":
        BACKEND.update({"name": "local", "url": a.url or "http://127.0.0.1:8091/v1/chat/completions", "think": a.think})
    elif a.backend == "dashscope":
        BACKEND.update({"name": "dashscope", "model": a.model, "think": True,
                        "url": a.url or "https://dashscope-intl.aliyuncs.com/compatible-mode/v1/chat/completions"})
    name = a.tag or {"deepseek": "deepseek", "dashscope": a.model.replace(".", "")}.get(a.backend, "qwen" + ("_think" if a.think else ""))
    out = Path(a.out) if a.out else TEST / f"{a.floor}_{name}_{a.mode}"; out.mkdir(parents=True, exist_ok=True)
    src = TEST / f"{a.floor}_sol_uzman" / f"kat_{a.floor}.png"
    if not src.exists():
        src = Path(r"C:\Users\Selim\Documents\cizim_projeler\egitim_verisi\sam_etiket_hassas_v3_20261022\floors") / a.floor / "clean.png"
    (out / f"kat_{a.floor}.png").write_bytes(src.read_bytes())
    fl = Floor(out / f"kat_{a.floor}.png")
    task = (TEST / f"GOREV_{a.floor}.md").read_text(encoding="utf-8")
    sysmsg = system_text(task, a.mode)
    tools = TOOLS
    plug = None
    if a.tool_module:
        import importlib
        plug = importlib.import_module(a.tool_module)
        tools = plug.TOOLS + [t for t in TOOLS if t["function"]["name"] == "submit"]
        sysmsg = sysmsg.split("HOW TO WORK:")[0] + plug.GUIDE + "\n\nTASK:\n" + task
    if a.lessons and Path(a.lessons).exists():
        sysmsg += "\n\nLESSONS FROM YOUR EARLIER RUNS (read carefully, apply them):\n" + Path(a.lessons).read_text(encoding="utf-8")
    ctx = {"out": out, "floor": fl, "image": out / f"kat_{a.floor}.png", "state": {}}
    if a.tools_from:
        src_w = Path(a.tools_from)
        (out / "tools").mkdir(exist_ok=True)
        for f in (src_w / "tools").glob("*.py"):
            (out / "tools" / f.name).write_bytes(f.read_bytes())
        araclar = (src_w / "ARACLAR.md").read_text(encoding="utf-8")
        (out / "ARACLAR.md").write_text(araclar, encoding="utf-8")
        tools = TOOLS + WORK_TOOLS
        sysmsg += ("\n\nYOUR OWN TOOLS: in an earlier workshop you wrote image-processing tools for exactly this job and a "
                   "guide for yourself (ARACLAR.md, below). They are in tools\\ in the test folder; run them with the run "
                   "tool, look at their output images with look, write seed files with write_file. This floor is NEW: no "
                   "approved drawing (GT) exists, so select candidates without GT (pick.py / wall stickiness / printed m²). "
                   f"This floor image is kat_{a.floor}.png ({fl.img.width} x {fl.img.height} px); its scale differs from "
                   "floor 019, calibrate it. Budget is tight: let the tools do the bulk work, do not trace rooms by hand "
                   "unless a tool fails.\n\n--- ARACLAR.md ---\n" + araclar)
    msgs = [{"role": "system", "content": sysmsg}, first_user(fl.img)]
    live = open(out / "SOHBET.md", "a", encoding="utf-8")
    log, t0, usage = [], time.time(), {"prompt_tokens": 0, "completion_tokens": 0}
    result = None
    steps = 1 if a.mode == "single" else a.steps
    for step in range(steps):
        if a.token_budget and usage["prompt_tokens"] + usage["completion_tokens"] >= a.token_budget:
            print(f"token budget {a.token_budget} reached: stopping", flush=True)
            break
        if (a.mode == "agent" and a.token_budget and not getattr(main, "_warned", False)
                and usage["prompt_tokens"] + usage["completion_tokens"] >= 0.8 * a.token_budget):
            main._warned = True
            msgs.append({"role": "user", "content": "Time is almost up: call submit NOW with every room you have drawn."})
        budget = plug.think_budget(ctx) if (plug is not None and hasattr(plug, "think_budget")) else None
        view = windowed_stable(msgs) if a.stable_window else windowed(msgs, a.keep_images)
        try:
            r = call(view, a.effort, tools if a.mode == "agent" else None, a.max_tokens, think_budget=budget)
        except SystemExit as e:
            if "exceed" not in str(e) or not a.stable_window:
                print(f"stopping: {e}", flush=True); break
            asst_i = [i for i, mm in enumerate(msgs) if mm["role"] == "assistant"]
            _CUT[0] = asst_i[-1] if asst_i else _CUT[0]   # context full: keep only the last step and retry once
            print(f"context full at step {step}: cut to the last step and retry", flush=True)
            try:
                r = call(windowed_stable(msgs), a.effort, tools, a.max_tokens, think_budget=budget)
            except SystemExit as e2:
                print(f"stopping: {e2}", flush=True); break
        for k in usage:
            usage[k] += (r.get("usage") or {}).get(k, 0)
        m = r["choices"][0]["message"]
        log.append({"step": step, "content": m.get("content"), "reasoning_chars": len(m.get("reasoning_content") or ""),
                    "finish_reason": r["choices"][0].get("finish_reason"),
                    "reasoning_tail": (m.get("reasoning_content") or "")[-1500:], "tool_calls": m.get("tool_calls")})
        print(f"step {step}: think budget {budget}, {len(m.get('tool_calls') or [])} tool calls, text {len(m.get('content') or '')} chars, "
              f"{time.time() - t0:.0f} s, tokens {usage}", flush=True)
        live.write(f"\n### Adım {step}  ({time.time() - t0:.0f} s, toplam token {usage['prompt_tokens'] + usage['completion_tokens']:,}, düşünme bütçesi {budget})\n")
        if m.get("reasoning_content"):
            live.write("> 💭 " + m["reasoning_content"][-700:].replace("\n", " ") + "\n\n")
        if m.get("content"):
            live.write(m["content"] + "\n")
        for tc in m.get("tool_calls") or []:
            live.write(f"🔧 **{tc['function']['name']}** `{tc['function']['arguments'][:400]}`\n")
        live.flush()
        if a.mode == "single":
            try:
                result = parse_json(m.get("content") or "")
            except Exception:
                result = None
            if not result and m.get("reasoning_content"):
                try:   # some models leave the JSON inside their reasoning
                    result = parse_json(m["reasoning_content"])
                except Exception:
                    result = None
            if result is not None and "rooms" not in result:
                result = None
            break
        assistant = {"role": "assistant", "content": m.get("content") or ""}
        if m.get("reasoning_content"):
            assistant["reasoning_content"] = m["reasoning_content"]
        if m.get("tool_calls"):
            assistant["tool_calls"] = m["tool_calls"]
        msgs.append(assistant)
        if not m.get("tool_calls"):
            if r["choices"][0].get("finish_reason") == "length":
                nudge = ("Your thinking ran out of room before you made a call. Do not re-plan everything: pick the next 1-3 "
                         "rooms, make ONE tool call now, and let the numbers it returns guide you.")
            else:
                nudge = "Continue with the tools; every reply must end in a tool call. When everything is done, call export."
            msgs.append({"role": "user", "content": nudge})
            continue
        images = []
        run_tcs = m["tool_calls"][:a.max_calls] if a.max_calls else m["tool_calls"]
        log[-1]["tool_calls"] = run_tcs   # the replay (rapor_gece.py) repeats exactly what was executed
        for tc in m["tool_calls"][len(run_tcs):]:
            msgs.append({"role": "tool", "tool_call_id": tc["id"],
                         "content": f"NOT RUN: at most {a.max_calls} tool calls per step. Call it again in a later step if still needed."})
            live.write(f"   ↳ NOT RUN (over the per-step limit): {tc['function']['name']}\n")
        for tc in run_tcs:
            name = tc["function"]["name"]
            try:
                args = json.loads(tc["function"]["arguments"] or "{}")
            except Exception:
                args = {}
            text = ""
            try:
                if plug is not None and name != "submit":
                    text, im, res = plug.handle(name, args, ctx)
                    for k, one in enumerate(im if isinstance(im, list) else [im]):   # a plugin may return several pictures
                        if one is not None:   # caption = first line only (the full text, e.g. a board, is in the tool message)
                            cap = text.split("\n")[0][:300] if k == 0 else getattr(one, "caption", "minimap")
                            images.append((tc["id"], cap, one))
                    if res is not None:
                        result = res
                elif name == "view":
                    im, text = fl.view(args.get("x0", 0), args.get("y0", 0), args.get("x1", 0), args.get("y1", 0), args.get("grid", 10))
                    if im is not None:
                        images.append((tc["id"], text, im))
                elif name == "scan":
                    text = fl.scan(args.get("axis", "row"), args.get("at", 0), args.get("start", 0), args.get("end", 0))
                elif name == "check":
                    im, text = fl.check(args.get("rooms", []))
                    images.append((tc["id"], text, im))
                elif name == "submit":
                    result = args.get("result"); text = "submitted"
                elif name == "run":
                    text = run_cmd(out, args.get("command", ""))
                elif name == "look":
                    im, text = look(out, args.get("path", ""), [args.get(k) for k in ("x0", "y0", "x1", "y1")])
                    images.append((tc["id"], text, im))
                elif name == "write_file":
                    q = inside(out, args.get("path", "x.txt")); q.parent.mkdir(parents=True, exist_ok=True)
                    q.write_text(args.get("content", ""), encoding="utf-8"); text = f"wrote {q.name} ({len(args.get('content', ''))} chars)"
            except Exception as e:
                text = f"tool error: {e}"
            msgs.append({"role": "tool", "tool_call_id": tc["id"], "content": text + (" (image follows in the next message)" if images and images[-1][0] == tc["id"] else "")})
            log[-1].setdefault("tool_results", []).append({"name": name, "args": args if name != "check" else {"n_rooms": len(args.get("rooms", []))}, "text": text[:400]})
            live.write(f"   ↳ {text[:300]}\n"); live.flush()
        json.dump(log, open(out / "transcript.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)   # every step
        json.dump({"mode": a.mode, "seconds": round(time.time() - t0), "steps": len(log), "usage": usage, "submitted": False},
                  open(out / "RUN.json", "w"), indent=1)
        if result is not None:
            break
        if a.max_images and len(images) > a.max_images:   # keep the newest pictures; a huge step must not overflow the context
            dropped = len(images) - a.max_images
            images = images[-a.max_images:]
            images[0] = (images[0][0], f"[{dropped} older picture(s) of this step dropped - look one region at a time] " + images[0][1], images[0][2])
        if images:
            parts = []
            for _, text, im in images:
                parts += [{"type": "text", "text": text}, img_part(im)]
            msgs.append({"role": "user", "content": parts})
    json.dump(log, open(out / "transcript.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    meta = {"mode": a.mode, "seconds": round(time.time() - t0), "steps": len(log), "usage": usage, "submitted": result is not None}
    json.dump(meta, open(out / "RUN.json", "w"), indent=1)
    if result:
        json.dump(result, open(out / f"SONUC_{a.floor}.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(json.dumps(meta))


if __name__ == "__main__":
    main()
