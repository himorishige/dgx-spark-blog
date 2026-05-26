"""CLI entry point for the PLC-style simulator.

Generates `--hours` worth of 1 Hz observations and writes them to a CSV.
A short summary (row count, label distribution) is printed at the end.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from simulator import (
    EquipmentPhysics,
    FactoryState,
    FailureInjector,
    LoadGenerator,
    ROW_COLUMNS,
    SensorGenerator,
    Simulator,
    StreamPublisher,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate PLC-style sensor data for the Chronos-2 + LLM blog article."
    )
    p.add_argument("--hours", type=float, default=72.0,
                   help="Simulated duration in hours (default: 72).")
    p.add_argument("--dt-seconds", type=float, default=1.0,
                   help="Sampling interval in seconds (default: 1.0).")
    p.add_argument("--output", type=Path, default=Path("data/sim_72h.csv"),
                   help="Output CSV path (relative to this script).")
    p.add_argument("--seed", type=int, default=42,
                   help="RNG seed for reproducibility.")
    p.add_argument("--wear-per-step", type=float, default=5.0e-6,
                   help="Bearing wear increment per tick (linear regime).")
    p.add_argument("--spike-prob", type=float, default=0.0008,
                   help="Per-tick probability of injecting a vibration spike.")
    p.add_argument("--no-failures", action="store_true",
                   help="Disable wear + spike injection (baseline-only dataset).")
    p.add_argument("--quiet", action="store_true",
                   help="Suppress per-hour progress output.")
    return p.parse_args(argv)


def build_simulator(args: argparse.Namespace) -> Simulator:
    state = FactoryState()
    load = LoadGenerator()
    physics = EquipmentPhysics()
    sensors = SensorGenerator()
    failures = FailureInjector(
        wear_per_step=0.0 if args.no_failures else args.wear_per_step,
        spike_prob=0.0 if args.no_failures else args.spike_prob,
    )
    return Simulator(
        state=state,
        load=load,
        physics=physics,
        sensors=sensors,
        failures=failures,
        dt_s=args.dt_seconds,
        seed=args.seed,
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    output = (Path(__file__).parent / args.output).resolve() if not args.output.is_absolute() else args.output
    sim = build_simulator(args)
    n_steps = int(args.hours * 3600 / args.dt_seconds)
    progress_every = max(1, int(3600 / args.dt_seconds))  # once per simulated hour

    label_counts: dict[str, int] = {"normal": 0, "wear": 0, "spike": 0}
    started = time.monotonic()
    with StreamPublisher(output, columns=ROW_COLUMNS) as pub:
        for idx, row in enumerate(sim.run(n_steps), start=1):
            pub.append(row)
            label_counts[row["label_kind"]] = label_counts.get(row["label_kind"], 0) + 1
            if not args.quiet and idx % progress_every == 0:
                hours_done = idx * args.dt_seconds / 3600
                print(f"  [{hours_done:5.1f}h] wear={row['wear_truth']:.3f}  "
                      f"mode={row['operation_mode']}  product={row['product_type']}")

    wall = time.monotonic() - started
    total = sum(label_counts.values())
    print(f"\n[run_simulation] wrote {total:,} rows -> {output}")
    print(f"[run_simulation] elapsed: {wall:.1f}s")
    print("[run_simulation] label distribution:")
    for kind in ("normal", "wear", "spike"):
        pct = label_counts.get(kind, 0) / total * 100 if total else 0.0
        print(f"   {kind:>8s} : {label_counts.get(kind, 0):8,d}  ({pct:5.1f}%)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
