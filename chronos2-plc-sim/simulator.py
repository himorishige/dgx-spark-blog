"""PLC-style factory simulator.

Six building blocks:
  - FactoryState     : slowly varying mode/shift/ambient
  - LoadGenerator    : diurnal + sinusoidal + noise load profile
  - EquipmentPhysics : physical relations between load/current/temperature/vibration
  - SensorGenerator  : observable signals with measurement noise + dropouts
  - FailureInjector  : gradual wear + sudden spikes
  - StreamPublisher  : CSV writer + recent-window deque

Designed to feed Chronos-2 multivariate windows downstream. Sampling is 1 Hz
by default which matches typical PLC scan budgets in the few-hundred-ms to
1-second range.
"""

from __future__ import annotations

import csv
import math
from collections import deque
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np


OperationMode = Literal["normal", "changeover"]
ProductType = Literal["A", "B", "C"]

SENSOR_COLUMNS: tuple[str, ...] = (
    "motor_current_a",
    "bearing_temp_c",
    "vibration_mm_s",
    "ambient_temp_c",
)

ROW_COLUMNS: tuple[str, ...] = (
    "timestamp_s",
    *SENSOR_COLUMNS,
    "line_speed_pct",
    "operation_mode",
    "product_type",
    "shift_id",
    "wear_truth",
    "spike_active",
    "label_anomaly",
    "label_kind",
)


SECONDS_PER_DAY = 86_400
SHIFT_LENGTH_S = 8 * 3600
PRODUCT_CYCLE_S = 12 * 3600
CHANGEOVER_WINDOW_S = 300  # 5 min before & after a product change
PRODUCTS: tuple[ProductType, ...] = ("A", "B", "C")


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

@dataclass
class FactoryState:
    """Slowly varying factory-floor state derived from the global clock."""

    timestamp_s: float = 0.0
    operation_mode: OperationMode = "normal"
    product_type: ProductType = "A"
    line_speed_pct: float = 80.0
    ambient_temp_c: float = 26.0
    shift_id: int = 1

    def advance(self, dt_s: float) -> None:
        self.timestamp_s += dt_s
        t = self.timestamp_s

        # Diurnal ambient: min at 06:00, peak at ~15:00, swing ±4°C.
        self.ambient_temp_c = 26.0 + 4.0 * math.sin(
            2 * math.pi * (t - 6 * 3600) / SECONDS_PER_DAY
        )

        # Shift rotation: 8h day / 8h evening / 8h night.
        shift_block = int((t // SHIFT_LENGTH_S) % 3)
        self.shift_id = shift_block + 1

        # Product cycle: A -> B -> C every 12h.
        product_block = int((t // PRODUCT_CYCLE_S) % len(PRODUCTS))
        target_product: ProductType = PRODUCTS[product_block]

        # Changeover window: ±CHANGEOVER_WINDOW_S around each cycle boundary.
        into_cycle = t % PRODUCT_CYCLE_S
        in_early_window = into_cycle < CHANGEOVER_WINDOW_S
        in_late_window = into_cycle > (PRODUCT_CYCLE_S - CHANGEOVER_WINDOW_S)
        if in_early_window or in_late_window:
            self.operation_mode = "changeover"
            self.line_speed_pct = 40.0
        else:
            self.operation_mode = "normal"
            self.line_speed_pct = 80.0

        self.product_type = target_product


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

@dataclass
class LoadGenerator:
    """Diurnal + product-type offset + Gaussian noise. Clipped to 0-100 %."""

    base_load: float = 60.0
    diurnal_amp: float = 15.0
    noise_std: float = 1.5
    product_offset: dict[ProductType, float] = field(
        default_factory=lambda: {"A": 0.0, "B": 5.0, "C": -3.0}
    )
    changeover_drop: float = 30.0

    def sample(self, state: FactoryState, rng: np.random.Generator) -> float:
        t = state.timestamp_s
        diurnal = self.diurnal_amp * math.sin(2 * math.pi * t / SECONDS_PER_DAY)
        offset = self.product_offset.get(state.product_type, 0.0)
        noise = rng.normal(0.0, self.noise_std)
        load = self.base_load + diurnal + offset + noise
        if state.operation_mode == "changeover":
            load -= self.changeover_drop
        return float(np.clip(load, 0.0, 100.0))


# ---------------------------------------------------------------------------
# Equipment physics
# ---------------------------------------------------------------------------

@dataclass
class EquipmentPhysics:
    """Load -> current -> temperature -> vibration coupling.

    Coefficients are intentionally simple so that Chronos-2 picks up the
    multivariate correlation without memorising idiosyncratic dynamics.
    """

    current_coeff: float = 0.10       # line_speed contribution to current
    friction_coeff: float = 2.0       # wear-driven friction adds current
    temp_load_coeff: float = 0.30
    temp_current_coeff: float = 0.20
    vibration_base: float = 1.0
    vibration_wear_coeff: float = 0.5

    def step(
        self,
        load: float,
        state: FactoryState,
        wear: float,
    ) -> dict[str, float]:
        current = state.line_speed_pct * self.current_coeff + self.friction_coeff * wear
        bearing_temp = (
            state.ambient_temp_c
            + load * self.temp_load_coeff
            + current * self.temp_current_coeff
        )
        vibration = self.vibration_base + self.vibration_wear_coeff * wear
        return {
            "motor_current_a": float(current),
            "bearing_temp_c": float(bearing_temp),
            "vibration_mm_s": float(vibration),
            "ambient_temp_c": float(state.ambient_temp_c),
        }


# ---------------------------------------------------------------------------
# Sensor generator
# ---------------------------------------------------------------------------

@dataclass
class SensorGenerator:
    """Per-sensor Gaussian noise + sporadic dropouts (NaN)."""

    noise_std: dict[str, float] = field(
        default_factory=lambda: {
            "motor_current_a": 0.10,
            "bearing_temp_c": 0.20,
            "vibration_mm_s": 0.05,
            "ambient_temp_c": 0.05,
        }
    )
    dropout_prob: float = 0.0001

    def observe(
        self,
        truth: dict[str, float],
        rng: np.random.Generator,
    ) -> dict[str, float]:
        obs: dict[str, float] = {}
        for key, value in truth.items():
            std = self.noise_std.get(key, 0.0)
            if rng.random() < self.dropout_prob:
                obs[key] = float("nan")
                continue
            noisy = value + (rng.normal(0.0, std) if std > 0.0 else 0.0)
            obs[key] = float(noisy)
        return obs


# ---------------------------------------------------------------------------
# Failure injector
# ---------------------------------------------------------------------------

@dataclass
class FailureInjector:
    """Gradual wear (linear + optional nonlinear acceleration) + vibration spikes."""

    wear_per_step: float = 5.0e-6
    nonlinear_threshold: float = 1.0
    nonlinear_gain: float = 2.5
    spike_prob: float = 0.0008
    spike_amplitude: float = 5.0
    spike_duration_steps: int = 5

    wear: float = 0.0
    _spike_ttl: int = 0
    _spike_value: float = 0.0

    def step(self, rng: np.random.Generator) -> tuple[float, float]:
        # Wear progression.
        increment = self.wear_per_step
        if self.wear > self.nonlinear_threshold:
            increment *= self.nonlinear_gain
        self.wear += increment

        # Spike state machine.
        if self._spike_ttl > 0:
            offset = self._spike_value
            self._spike_ttl -= 1
        elif self.wear_per_step > 0.0 and rng.random() < self.spike_prob:
            self._spike_ttl = self.spike_duration_steps - 1
            self._spike_value = float(
                rng.uniform(self.spike_amplitude * 0.5, self.spike_amplitude)
            )
            offset = self._spike_value
        else:
            offset = 0.0
        return float(self.wear), float(offset)


# ---------------------------------------------------------------------------
# Stream publisher
# ---------------------------------------------------------------------------

class StreamPublisher:
    """Append rows to CSV and keep a deque of the most recent window."""

    def __init__(
        self,
        output_path: Path | str,
        window_size: int = 512,
        columns: tuple[str, ...] = ROW_COLUMNS,
    ) -> None:
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.window_size = window_size
        self.columns = columns
        self._window: deque[dict[str, float]] = deque(maxlen=window_size)
        self._file = self.output_path.open("w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._file, fieldnames=list(columns))
        self._writer.writeheader()

    def append(self, row: dict[str, float]) -> None:
        self._writer.writerow({k: row.get(k) for k in self.columns})
        self._window.append(row)

    def recent_window(self) -> list[dict[str, float]]:
        return list(self._window)

    def close(self) -> None:
        if not self._file.closed:
            self._file.close()

    def __enter__(self) -> "StreamPublisher":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


# ---------------------------------------------------------------------------
# Top-level simulator
# ---------------------------------------------------------------------------

@dataclass
class Simulator:
    """Coordinate the six blocks and yield observations one tick at a time."""

    state: FactoryState = field(default_factory=FactoryState)
    load: LoadGenerator = field(default_factory=LoadGenerator)
    physics: EquipmentPhysics = field(default_factory=EquipmentPhysics)
    sensors: SensorGenerator = field(default_factory=SensorGenerator)
    failures: FailureInjector = field(default_factory=FailureInjector)
    dt_s: float = 1.0
    seed: int = 42

    _rng: np.random.Generator = field(init=False, repr=False)
    label_wear_threshold: float = 0.5

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng(self.seed)

    def tick(self) -> dict[str, float]:
        timestamp_s = self.state.timestamp_s
        self.state.advance(self.dt_s)

        wear, spike_offset = self.failures.step(self._rng)
        load = self.load.sample(self.state, self._rng)
        truth = self.physics.step(load, self.state, wear)
        truth["vibration_mm_s"] += spike_offset

        obs = self.sensors.observe(truth, self._rng)

        spike_active = spike_offset > 0.0
        if spike_active:
            label_kind = "spike"
            label_anomaly = 1
        elif wear > self.label_wear_threshold:
            label_kind = "wear"
            label_anomaly = 1
        else:
            label_kind = "normal"
            label_anomaly = 0

        return {
            "timestamp_s": float(timestamp_s),
            **obs,
            "line_speed_pct": float(self.state.line_speed_pct),
            "operation_mode": self.state.operation_mode,
            "product_type": self.state.product_type,
            "shift_id": int(self.state.shift_id),
            "wear_truth": float(wear),
            "spike_active": int(spike_active),
            "label_anomaly": int(label_anomaly),
            "label_kind": label_kind,
        }

    def run(self, n_steps: int) -> Iterator[dict[str, float]]:
        for _ in range(n_steps):
            yield self.tick()


# ---------------------------------------------------------------------------
# Module sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    sim = Simulator()
    print("First 5 ticks:")
    for i, row in zip(range(5), sim.run(5)):
        print(f"  t={row['timestamp_s']:.0f}s  "
              f"I={row['motor_current_a']:.2f}A  "
              f"T={row['bearing_temp_c']:.2f}°C  "
              f"V={row['vibration_mm_s']:.3f}mm/s  "
              f"mode={row['operation_mode']}  "
              f"label={row['label_kind']}")
