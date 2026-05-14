"""RM1 step 2: small head-motion sanity test (relative-from-current).

Sequence:
  1. wake_up()
  2. capture neutral pose / joints as baseline
  3. lift head +10 mm in Z from current pose
  4. pitch +10 deg from current pose
  5. yaw +20 deg from current pose
  6. return to captured neutral
  7. goto_sleep()

Each step reads back both the head pose AND joint positions so we can tell
whether servos are moving even if the pose readout stays cached.
"""

from __future__ import annotations

import contextlib
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

DAEMON_PORT = 8765
DAEMON_STARTUP_TIMEOUT = 60.0
STEP_DURATION = 2.5  # seconds for each goto_target (commanded motion time)
SETTLE_TIME = 1.5    # seconds to let motion finish + give user time to see it


def section(title: str) -> None:
    print(f"\n{'=' * 8} {title} {'=' * 8}")


def wait_for_port(port: int, host: str = "127.0.0.1", timeout: float = 30.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        with contextlib.closing(socket.socket()) as s:
            s.settimeout(0.5)
            try:
                s.connect((host, port))
                return True
            except OSError:
                time.sleep(0.5)
    return False


def start_daemon(port: int) -> subprocess.Popen:
    section(f"Starting daemon on :{port}")
    cmd = [
        "reachy-mini-daemon",
        "--fastapi-host",
        "127.0.0.1",
        "--fastapi-port",
        str(port),
        "--localhost-only",
        "--no-media",
        "--headless",
        "--wake-up-on-start",
        "--goto-sleep-on-stop",
        "--log-level",
        "WARNING",
    ]
    print(f"  cmd : {' '.join(cmd)}")
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        preexec_fn=os.setsid,
    )
    print(f"  pid : {proc.pid}")
    return proc


def stop_daemon(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    section("Stopping daemon")
    with contextlib.suppress(ProcessLookupError):
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    try:
        proc.wait(timeout=10)
        print(f"  daemon exited with code {proc.returncode}")
    except subprocess.TimeoutExpired:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        proc.wait()


def report_state(label: str, mini) -> None:
    pose = mini.get_current_head_pose()
    xyz_mm = [round(v * 1000.0, 1) for v in pose[:3, 3].tolist()]
    head_joints, antenna_joints = mini.get_current_joint_positions()
    head_deg = [round(np.degrees(j), 2) for j in head_joints]
    antenna_deg = [round(np.degrees(j), 2) for j in antenna_joints]
    print(f"    [{label}] xyz(mm)={xyz_mm}")
    print(f"    [{label}] head_joints(deg)={head_deg}")
    print(f"    [{label}] antenna(deg)={antenna_deg}")


def run_motion_sequence(port: int) -> bool:
    from reachy_mini import ReachyMini
    from reachy_mini.utils import create_head_pose

    try:
        with ReachyMini(
            host="127.0.0.1",
            port=port,
            connection_mode="localhost_only",
            spawn_daemon=False,
            log_level="WARNING",
            media_backend="none",
        ) as mini:
            print("  client connected")

            # daemon already ran wake_up at startup (--wake-up-on-start);
            # we just sample state here for the baseline.
            section("baseline (daemon already woke up)")
            time.sleep(1.0)
            report_state("baseline", mini)

            # Capture baseline (neutral) so we can do relative moves.
            neutral_pose = mini.get_current_head_pose().copy()
            print("\n  captured neutral pose as baseline for relative deltas")

            def relative_target(dz_mm=0.0, dpitch_deg=0.0, dyaw_deg=0.0):
                """Build a target by applying a delta to the neutral baseline."""
                delta = create_head_pose(
                    z=dz_mm, pitch=dpitch_deg, yaw=dyaw_deg,
                    mm=True, degrees=True,
                )
                return neutral_pose @ delta

            section("Step 1: lift +10 mm in Z (relative)")
            mini.goto_target(head=relative_target(dz_mm=10), duration=STEP_DURATION)
            time.sleep(STEP_DURATION + SETTLE_TIME)
            report_state("after Z+10", mini)

            section("Step 2: pitch +10 deg (relative)")
            mini.goto_target(head=relative_target(dpitch_deg=10), duration=STEP_DURATION)
            time.sleep(STEP_DURATION + SETTLE_TIME)
            report_state("after pitch+10", mini)

            section("Step 3: yaw +20 deg (relative)")
            mini.goto_target(head=relative_target(dyaw_deg=20), duration=STEP_DURATION)
            time.sleep(STEP_DURATION + SETTLE_TIME)
            report_state("after yaw+20", mini)

            section("Step 4: return to neutral")
            mini.goto_target(head=neutral_pose, duration=STEP_DURATION)
            time.sleep(STEP_DURATION + SETTLE_TIME)
            report_state("after neutral", mini)

            # daemon will run goto_sleep at shutdown (--goto-sleep-on-stop)
            print("\n  (daemon will run goto_sleep on shutdown)")

        return True
    except Exception as e:  # noqa: BLE001
        print(f"  MOTION FAILED: {type(e).__name__}: {e}")
        return False


def main() -> int:
    print("Reachy Mini RM1 step 2 — head motion sanity test (relative-from-current)")
    daemon = start_daemon(DAEMON_PORT)
    try:
        if not wait_for_port(DAEMON_PORT, timeout=DAEMON_STARTUP_TIMEOUT):
            print(f"  daemon did not open :{DAEMON_PORT} within {DAEMON_STARTUP_TIMEOUT}s")
            return 1
        print(f"  daemon listening on :{DAEMON_PORT}")
        ok = run_motion_sequence(DAEMON_PORT)
    finally:
        stop_daemon(daemon)

    section("Summary")
    print(f"  {'OK ' if ok else 'NG '} motion sequence")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
