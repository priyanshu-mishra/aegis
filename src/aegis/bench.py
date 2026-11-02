"""Latency benchmark: end-to-end backtest wall time, per-day scoring, and parser throughput."""

from __future__ import annotations

import json
import platform
import time
from pathlib import Path

import numpy as np

from aegis.backtest import run_backtest
from aegis.config import AegisConfig, ParserConfig
from aegis.data.synthetic import generate_world
from aegis.slm_parser import LexiconParser, OllamaParser, mask_entities


def _pcts(seconds: list[float]) -> dict[str, float]:
    ms = np.asarray(seconds) * 1000
    return {"n": len(ms), "p50_ms": float(np.percentile(ms, 50)),
            "p95_ms": float(np.percentile(ms, 95)), "max_ms": float(ms.max())}  # fmt: skip


def _time_parser(parse, texts: list[str]) -> dict[str, float]:
    times = []
    for text in texts:
        t0 = time.perf_counter()
        parse(text)
        times.append(time.perf_counter() - t0)
    return _pcts(times)


def run_bench(cfg: AegisConfig, slm_items: int = 10) -> dict:
    """Benchmark on the given (offline) config; the SLM is timed only if Ollama is reachable.

    Parser timings bypass the cache so they measure real parsing cost.
    """
    out: dict = {"machine": f"{platform.machine()} · {platform.processor() or platform.system()}"}

    t0 = time.perf_counter()
    res = run_backtest(cfg)
    out["backtest_wall_s"] = time.perf_counter() - t0
    out["test_days"] = len(res.vol)
    lat = res.info["latency"]
    out["score_one_day"] = lat.get("score_one_day")
    out["walk_forward_per_day_ms"] = (
        sum(v["total_s"] for k, v in lat.items() if k == "walk_forward") * 1000 / len(res.vol)
    )

    world = generate_world(cfg.data.tickers, n_days=600, seed=1)
    # Distinct texts only, so repeated templates don't flatter the numbers.
    texts = list(dict.fromkeys(mask_entities(i.text) for i in world.news))
    out["lexicon_parse"] = _time_parser(LexiconParser(cfg.data.tickers).parse, texts[:300])

    pc = ParserConfig(backend="ollama", model=cfg.parser.model, ollama_url=cfg.parser.ollama_url)
    slm = OllamaParser(model=pc.model, url=pc.ollama_url, tickers=cfg.data.tickers)
    if slm.ping():
        slm.parse(texts[0])  # warm-up: first call loads the model into memory
        out["slm_parse"] = {"model": pc.model, **_time_parser(slm.parse, texts[1 : 1 + slm_items])}
    else:
        out["slm_parse"] = None
    return out


def write_bench(result: dict, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "benchmark.json"
    path.write_text(json.dumps(result, indent=2))
    return path
