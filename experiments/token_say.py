"""Token use of one floor-test run, read from what the runner saved:
  Antigravity (Gemini):   events.jsonl step_update.usage, one per model step (last value per step_index), summed
  Codex (Sol / Terra):    events.jsonl turn.completed.usage (summed over turns)
  Claude Code:            events.jsonl result.usage (+ total_cost_usd)
  OpenAI-style API runs:  RUN.json usage.prompt_tokens / completion_tokens
Returns {kaynak, giris (fresh input), onbellek (cache read), cikti (output incl. thinking), dusunme, toplam, maliyet_usd}."""
import sys as _sys; from pathlib import Path as _P; _sys.path.insert(0, str(_P(__file__).resolve().parents[1]))
import config; config.setup()  # noqa: E402  repo paths: data/model roots come from config.local.json or env
import json
from pathlib import Path


def _text(p):
    b = Path(p).read_bytes()
    return b.decode("utf-16") if b[:2] in (b"\xff\xfe", b"\xfe\xff") else b.decode("utf-8-sig", "replace")


def tokens(run):
    run = Path(run); ev = run / "events.jsonl"; out = None
    if ev.exists():
        steps, turns, res = {}, [], None
        for line in _text(ev).splitlines():
            try: e = json.loads(line)
            except Exception: continue
            su = (e.get("step_update") or {})
            if su.get("usage"): steps[(su.get("conversation_id"), su.get("step_index"))] = su["usage"]
            if e.get("type") == "turn.completed" and e.get("usage"): turns.append(e["usage"])
            if e.get("type") == "result" and e.get("usage"): res = e
        if steps:
            u = list(steps.values()); g = sum(x.get("input_tokens", 0) for x in u); c = sum(x.get("cache_read_tokens", 0) for x in u)
            o = sum(x.get("output_tokens", 0) for x in u); th = sum(x.get("thinking_tokens", 0) for x in u)
            out = {"kaynak": "Antigravity", "giris": g, "onbellek": c, "cikti": o + th, "dusunme": th, "adim": len(u)}
        elif turns:
            g = sum(x.get("input_tokens", 0) - x.get("cached_input_tokens", 0) for x in turns); c = sum(x.get("cached_input_tokens", 0) for x in turns)
            o = sum(x.get("output_tokens", 0) for x in turns); th = sum(x.get("reasoning_output_tokens", 0) for x in turns)
            out = {"kaynak": "Codex", "giris": g, "onbellek": c, "cikti": o, "dusunme": th, "tur": len(turns)}
        elif res:
            u = res["usage"]; g = u.get("input_tokens", 0) + u.get("cache_creation_input_tokens", 0); c = u.get("cache_read_input_tokens", 0)
            o = u.get("output_tokens", 0); th = (u.get("output_tokens_details") or {}).get("thinking_tokens", 0)
            out = {"kaynak": "Claude Code", "giris": g, "onbellek": c, "cikti": o, "dusunme": th, "maliyet_usd": round(res.get("total_cost_usd", 0), 3)}
    if out is None and (run / "RUN.json").exists():
        R = json.loads(_text(run / "RUN.json")); u = R.get("usage") or R
        if "prompt_tokens" in u: out = {"kaynak": "API", "giris": u["prompt_tokens"], "onbellek": 0, "cikti": u.get("completion_tokens", 0), "dusunme": None}
    if out: out["toplam"] = out["giris"] + out["onbellek"] + out["cikti"]
    return out


if __name__ == "__main__":
    import sys
    for r in sys.argv[1:]: print(Path(r).name, tokens(r))
