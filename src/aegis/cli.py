"""Command-line entry point: ``python main.py <command> --config configs/default.yaml``."""

from __future__ import annotations

import argparse
import sys

import yaml
from pydantic import ValidationError
from rich.console import Console

from aegis import __version__
from aegis.config import AegisConfig, load_config

console = Console(stderr=True)


def _cmd_config(cfg: AegisConfig, _args: argparse.Namespace) -> int:
    print(yaml.safe_dump(cfg.model_dump(mode="json"), sort_keys=False), end="")
    return 0


def _cmd_fetch(cfg: AegisConfig, args: argparse.Namespace) -> int:
    from rich.table import Table

    from aegis.data import load_dataset

    ds = load_dataset(cfg, refresh=args.refresh)
    m = ds.market
    table = Table(title=f"Dataset '{cfg.name}' ({m.source})")
    table.add_column("series")
    table.add_column("ann. vol", justify="right")
    for name, col in m.returns.items():
        table.add_row(name, f"{col.std() * 252**0.5:.1%}")
    table.add_row("portfolio", f"{m.portfolio.std() * 252**0.5:.1%}", style="bold")
    console.print(table)
    console.print(
        f"{len(m.trading_days)} trading days {m.trading_days[0]:%Y-%m-%d} → "
        f"{m.trading_days[-1]:%Y-%m-%d}; features {list(m.features.columns)}; "
        f"{len(ds.news)} news items ({cfg.data.news_source})"
    )
    return 0


def _out_dir(cfg: AegisConfig):
    return cfg.report.output_dir / cfg.name


def _cmd_backtest(cfg: AegisConfig, args: argparse.Namespace) -> int:
    import logging

    from aegis.backtest import run_backtest
    from aegis.report import write_report

    logging.getLogger("aegis").setLevel(logging.INFO)
    with console.status(f"Running backtest '{cfg.name}'..."):
        res = run_backtest(cfg)
        out = _out_dir(cfg)
        res.save(out)
        path = write_report(res, out, figures=cfg.report.figures)
    _print_summary(path.parent)
    console.print(f"[green]Report written to[/green] {path}")
    return 0


def _cmd_report(cfg: AegisConfig, args: argparse.Namespace) -> int:
    from aegis.backtest import BacktestResult
    from aegis.report import write_report

    out = _out_dir(cfg)
    if not (out / "run_info.json").exists():
        console.print(f"[red]No saved backtest in {out}; run 'backtest' first.[/red]")
        return 1
    path = write_report(BacktestResult.load(out), out, figures=cfg.report.figures)
    _print_summary(out)
    console.print(f"[green]Report written to[/green] {path}")
    return 0


def _print_summary(out_dir) -> None:
    import json

    from rich.table import Table

    from aegis.report import LABELS

    metrics = json.loads((out_dir / "metrics.json").read_text())
    table = Table(title=f"Target A coverage (target {1 - metrics['alpha']:.0%})")
    for col in ("method", "coverage", "avg width", "worst 63d"):
        table.add_column(col, justify="left" if col == "method" else "right")
    for m, v in metrics["vol"].items():
        width = "∞" if v["mean_width"] is None else f"{v['mean_width']:.3f}"
        table.add_row(LABELS.get(m, m), f"{v['coverage']:.1%}", width,
                      f"{v['worst_rolling_coverage']:.1%}")  # fmt: skip
    console.print(table)
    for a, mv in metrics["var"].items():
        t = Table(title=f"{(1 - float(a)) * 100:g}% VaR (target breaches {float(a):.0%})")
        for col in ("method", "breaches", "Kupiec p", "Christoffersen p"):
            t.add_column(col, justify="left" if col == "method" else "right")
        for m, v in mv.items():
            t.add_row(LABELS.get(m, m), f"{v['breach_rate']:.2%}", f"{v['kupiec_p']:.3f}",
                      f"{v['christoffersen_p']:.3f}")  # fmt: skip
        console.print(t)


def _cmd_parse(cfg: AegisConfig, args: argparse.Namespace) -> int:
    import numpy as np

    from aegis.data import load_dataset
    from aegis.slm_parser import build_parser, parse_items

    ds = load_dataset(cfg)
    try:
        parser = build_parser(cfg.parser, list(ds.market.returns.columns))
    except RuntimeError as exc:
        console.print(f"[red]{exc}[/red]")
        return 1
    with console.status(f"Parsing {len(ds.news)} items with {cfg.parser.backend}..."):
        parsed = parse_items(parser, ds.news, mask=cfg.parser.mask_entities)
    lat = np.array(parser.latencies_ms)
    console.print(
        f"{len(parsed)} items · {parser.hits} cached · {parser.misses} parsed now"
        + (
            f" · p50 {np.percentile(lat, 50):.1f} ms, p95 {np.percentile(lat, 95):.1f} ms"
            if len(lat)
            else ""
        )
        + f" · fallbacks {getattr(parser.inner, 'fallbacks', 0)}"
    )
    parser.close()
    if len(parsed):
        cols = [c for c in parsed.columns if c.startswith("shock_")] + ["sentiment"]
        console.print(parsed[cols].describe().loc[["mean", "std", "min", "max"]].round(2))
    return 0


def _cmd_ablation(cfg: AegisConfig, args: argparse.Namespace) -> int:
    from aegis.ablation import run_ablation, write_ablation

    with console.status(f"Running ablation '{cfg.name}'..."):
        df = run_ablation(cfg)
        path = write_ablation(df, cfg, _out_dir(cfg))
    console.print(path.read_text())
    console.print(f"[green]Ablation written to[/green] {path}")
    return 0


DEMO_OVERRIDES = [
    "name=demo",
    "data.synthetic.n_days=3000",
    "splits.train_end=2014-12-31",
    "splits.cal_end=2016-06-30",
]


def _cmd_demo(cfg: AegisConfig, args: argparse.Namespace) -> int:
    """Offline end-to-end run on a small synthetic world: no network, no model download."""
    import time
    from pathlib import Path

    demo_cfg = load_config(
        Path(__file__).resolve().parents[2] / "configs" / "synthetic.yaml",
        [*DEMO_OVERRIDES, f"report.output_dir={cfg.report.output_dir}", *args.overrides],
    )
    t0 = time.perf_counter()
    rc = _cmd_backtest(demo_cfg, args)
    console.print(f"Demo finished in {time.perf_counter() - t0:.1f}s (synthetic data, offline).")
    return rc


def _cmd_bench(cfg: AegisConfig, args: argparse.Namespace) -> int:
    from pathlib import Path

    from rich.table import Table

    from aegis.bench import run_bench, write_bench

    bench_cfg = load_config(
        Path(__file__).resolve().parents[2] / "configs" / "synthetic.yaml",
        [*DEMO_OVERRIDES, "name=bench", *args.overrides],
    )
    with console.status("Benchmarking..."):
        r = run_bench(bench_cfg)
    path = write_bench(r, cfg.report.output_dir)
    t = Table(title=f"Latency ({r['machine']})")
    for col in ("measure", "value"):
        t.add_column(col, justify="left" if col == "measure" else "right")
    t.add_row(f"Full backtest ({r['test_days']} test days)", f"{r['backtest_wall_s']:.2f} s")
    s = r["score_one_day"]
    t.add_row("Score one day (model + conformal), p50 / p95",
              f"{s['p50_ms']:.2f} / {s['p95_ms']:.2f} ms")  # fmt: skip
    t.add_row("Walk-forward, all methods, per test day", f"{r['walk_forward_per_day_ms']:.2f} ms")
    lx = r["lexicon_parse"]
    t.add_row(f"Lexicon parse, p50 / p95 (n={lx['n']})",
              f"{lx['p50_ms']:.2f} / {lx['p95_ms']:.2f} ms")  # fmt: skip
    if r["slm_parse"]:
        sl = r["slm_parse"]
        t.add_row(f"SLM parse ({sl['model']}), p50 / p95 (n={sl['n']})",
                  f"{sl['p50_ms']:.0f} / {sl['p95_ms']:.0f} ms")  # fmt: skip
    else:
        t.add_row("SLM parse", "skipped (Ollama not reachable)")
    console.print(t)
    console.print(f"[green]Benchmark written to[/green] {path}")
    return 0


def _cmd_sensitivity(cfg: AegisConfig, args: argparse.Namespace) -> int:
    from aegis.sensitivity import run_sensitivity

    out = _out_dir(cfg)
    if not (out / "run_info.json").exists():
        console.print(f"[red]No saved backtest in {out}; run 'backtest' first.[/red]")
        return 1
    with console.status("Running sensitivity grid..."):
        path = run_sensitivity(cfg, out, with_lambda=not args.quick)
    console.print(path.read_text())
    return 0


def _cmd_dashboard(cfg: AegisConfig, args: argparse.Namespace) -> int:
    import importlib.util
    import subprocess
    from pathlib import Path

    if importlib.util.find_spec("streamlit") is None:
        console.print('[red]Streamlit is not installed:[/red] pip install -e ".[dashboard]"')
        return 1
    app = Path(__file__).resolve().parents[2] / "dashboard" / "app.py"
    return subprocess.call([sys.executable, "-m", "streamlit", "run", str(app)])


def build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("-c", "--config", help="YAML config path (defaults if omitted)")
    common.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="override a config value, e.g. --set conformal.alpha=0.1 (repeatable)",
    )

    parser = argparse.ArgumentParser(prog="aegis", description=__doc__)
    parser.add_argument("--version", action="version", version=f"aegis {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("config", parents=[common], help="print the resolved configuration")
    fetch = sub.add_parser("fetch", parents=[common], help="download and cache prices and news")
    fetch.add_argument("--refresh", action="store_true", help="ignore caches and re-download")
    for name, help_text in [
        ("parse", "run the SLM parser over the news store"),
        ("ablation", "compare backtests with and without text signals"),
        ("demo", "offline end-to-end run on a small synthetic world"),
        ("bench", "latency benchmark (backtest, per-day scoring, parsers)"),
        ("dashboard", "launch the Streamlit dashboard over saved runs"),
        ("backtest", "run the walk-forward conformal backtest"),
        ("report", "regenerate the report from saved metrics"),
    ]:
        sub.add_parser(name, parents=[common], help=help_text)
    sens = sub.add_parser(
        "sensitivity", parents=[common], help="coverage and width across α, γ, window and λ"
    )
    sens.add_argument("--quick", action="store_true", help="skip the EWMA λ re-runs")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        cfg = load_config(args.config, args.overrides)
    except (OSError, ValueError, ValidationError) as exc:
        # pydantic's ValidationError subclasses ValueError; listed for clarity.
        console.print(f"[red]Invalid configuration:[/red] {exc}")
        return 1

    handlers = {
        "config": _cmd_config,
        "fetch": _cmd_fetch,
        "backtest": _cmd_backtest,
        "report": _cmd_report,
        "parse": _cmd_parse,
        "ablation": _cmd_ablation,
        "demo": _cmd_demo,
        "bench": _cmd_bench,
        "dashboard": _cmd_dashboard,
        "sensitivity": _cmd_sensitivity,
    }
    handler = handlers[args.command]
    return handler(cfg, args)


if __name__ == "__main__":
    sys.exit(main())
