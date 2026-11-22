"""One conversation turn with DeepSeek Harness (headless) in the tool workshop, with a live readable chat log.

usage: python atolye_tur.py --dir WORKDIR --log LOGDIR --msg MESSAGE_FILE [--new]
- Sends MESSAGE_FILE as the user message (Claude speaking). Resumes the session in LOGDIR/SESSION.txt unless --new.
- Raw events -> LOGDIR/events_turN.jsonl; readable transcript appended to LOGDIR/SOHBET.md as it happens.
"""
import sys as _sys; from pathlib import Path as _P; _sys.path.insert(0, str(_P(__file__).resolve().parents[1]))
import config; config.setup()  # noqa: E402  repo paths: data/model roots come from config.local.json or env
import argparse
import datetime as dt
import json
import os
import subprocess
import time
import winreg
from pathlib import Path

DSH = r"C:\Users\Selim\AppData\Local\Programs\DeepSeek Harness\resources\runtime\cli\bin\dsh.cmd"


def user_env(name):
    with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as k:
        return winreg.QueryValueEx(k, name)[0]


def short(s, n):
    s = str(s)
    return s if len(s) <= n else s[:n] + f" … (+{len(s) - n} karakter)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--log", required=True)
    ap.add_argument("--msg", required=True)
    ap.add_argument("--new", action="store_true")
    a = ap.parse_args()
    log = Path(a.log); log.mkdir(parents=True, exist_ok=True)
    sess_file = log / "SESSION.txt"
    turn = len(list(log.glob("events_tur*.jsonl"))) + 1
    msg = Path(a.msg).read_text(encoding="utf-8")
    cmd = ["cmd", "/c", DSH, "--profile", "headless", "--json", "--permission-mode", "workspace-write",
           "--approval-policy", "never"]
    if not a.new and sess_file.exists():
        cmd += ["--session-id", sess_file.read_text().strip()]
    cmd += ["-"]
    env = dict(os.environ, DEEPSEEK_API_KEY=user_env("DEEPSEEK_API_KEY"), PYTHONIOENCODING="utf-8")
    md = open(log / "SOHBET.md", "a", encoding="utf-8")

    def w(s):
        md.write(s + "\n"); md.flush()
        print(s, flush=True)

    w(f"\n\n---\n## Tur {turn}  ({dt.datetime.now():%H:%M})\n\n### 🟣 Claude\n\n{msg.strip()}\n\n### 🔵 DeepSeek\n")
    raw = open(log / f"events_tur{turn}.jsonl", "w", encoding="utf-8")
    t0 = time.time()
    p = subprocess.Popen(cmd, cwd=a.dir, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=open(log / f"stderr_tur{turn}.txt", "w"),
                         env=env)
    p.stdin.write(msg.encode("utf-8")); p.stdin.close()
    usage_out = usage_in = usage_cache = 0
    steps = 0
    for line in p.stdout:
        line = line.decode("utf-8", errors="replace").strip()
        if not line:
            continue
        raw.write(line + "\n"); raw.flush()
        try:
            e = json.loads(line)
        except Exception:
            continue
        t = e.get("type")
        if t == "session":
            sess_file.write_text(e["sessionId"])
        elif t == "thinking":
            w(f"> 💭 {short(e.get('text', '').strip(), 600)}\n")
        elif t == "text":
            w(e.get("text", "").strip() + "\n")
        elif t == "tool_call":
            inp = e.get("input", {})
            body = inp.get("command") or inp.get("file_path") or json.dumps(inp, ensure_ascii=False)
            w(f"🔧 **{e.get('tool')}**: `{short(body, 400)}`")
        elif t == "tool_result":
            r = e.get("result") or e.get("output") or e.get("content") or ""
            if not isinstance(r, str):
                r = json.dumps(r, ensure_ascii=False)
            w(f"   ↳ {short(r.strip(), 300)}\n")
        elif t == "status" and e.get("phase") == "step_end":
            steps += 1
            u = e.get("usage", {})
            usage_in += u.get("inputTokens", 0); usage_out += u.get("outputTokens", 0); usage_cache += u.get("cacheReadTokens", 0)
        elif t == "final":
            pass
    p.wait()
    w(f"\n*Tur {turn} bitti: {steps} adım, {(time.time() - t0) / 60:.1f} dk, çıktı {usage_out:,} token, "
      f"yeni girdi {usage_in:,}, önbellek {usage_cache:,}, çıkış kodu {p.returncode}*")


if __name__ == "__main__":
    main()
