"""RM1 step 3: expressive motion demo (still safe envelope).

Sequence (all absolute from INIT_HEAD_POSE = identity / xyz=0):
  1. nod    : pitch +15 / -15 / 0
  2. shake  : yaw +25 / -25 / 0
  3. tilt   : roll +15 / -15 / 0
  4. antennas: both up / both down / neutral
  5. return to INIT_HEAD_POSE
"""

from __future__ import annotations

import contextlib
import math
import os
import signal
import socket
import subprocess
import sys
import time

import numpy as np

DAEMON_PORT = 8765
DAEMON_STARTUP_TIMEOUT = 60.0
STEP_DURATION = 1.5
SETTLE = 0.6


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
        "--fastapi-host", "127.0.0.1",
        "--fastapi-port", str(port),
        "--localhost-only",
        "--no-media",
        "--headless",
        "--wake-up-on-start",
        "--goto-sleep-on-stop",
        "--log-level", "WARNING",
    ]
    print(f"  cmd : {' '.join(cmd)}")
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, preexec_fn=os.setsid,
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
        proc.wait(timeout=15)
        print(f"  daemon exited with code {proc.returncode}")
    except subprocess.TimeoutExpired:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        proc.wait()


def report(label: str, mini) -> None:
    pose = mini.get_current_head_pose()
    xyz_mm = [round(v * 1000.0, 1) for v in pose[:3, 3].tolist()]
    head_joints, antenna_joints = mini.get_current_joint_positions()
    head_deg = [round(np.degrees(j), 1) for j in head_joints]
    antenna_deg = [round(np.degrees(j), 1) for j in antenna_joints]
    print(f"    [{label}] xyz(mm)={xyz_mm}  antenna(deg)={antenna_deg}")
    print(f"    [{label}] head_joints(deg)={head_deg}")


def goto(mini, *, roll=0.0, pitch=0.0, yaw=0.0, antennas=None, dur=STEP_DURATION):
    """Move to an absolute pose described by Euler RPY (deg) plus optional antennas."""
    from reachy_mini.utils import create_head_pose

    head = create_head_pose(roll=roll, pitch=pitch, yaw=yaw, mm=True, degrees=True)
    if antennas is None:
        mini.goto_target(head=head, duration=dur)
    else:
        mini.goto_target(head=head, antennas=antennas, duration=dur)
    time.sleep(dur + SETTLE)


def run_demo(port: int) -> bool:
    from reachy_mini import ReachyMini

    try:
        with ReachyMini(
            host="127.0.0.1", port=port,
            connection_mode="localhost_only", spawn_daemon=False,
            log_level="WARNING", media_backend="none",
        ) as mini:
            print("  client connected")
            time.sleep(0.8)
            report("baseline (post wake_up)", mini)

            section("Nod (pitch +15 / -15 / 0)")
            goto(mini, pitch=15)
            report("pitch+15", mini)
            goto(mini, pitch=-15)
            report("pitch-15", mini)
            goto(mini)  # back to identity
            report("pitch 0", mini)

            section("Shake (yaw +25 / -25 / 0)")
            goto(mini, yaw=25)
            report("yaw+25", mini)
            goto(mini, yaw=-25)
            report("yaw-25", mini)
            goto(mini)
            report("yaw 0", mini)

            section("Tilt (roll +15 / -15 / 0)")
            goto(mini, roll=15)
            report("roll+15", mini)
            goto(mini, roll=-15)
            report("roll-15", mini)
            goto(mini)
            report("roll 0", mini)

            section("Antennas (both up / both down / neutral)")
            # Both forward
            goto(mini, antennas=[math.radians(60), math.radians(-60)])
            report("antennas forward", mini)
            # Both back
            goto(mini, antennas=[math.radians(-60), math.radians(60)])
            report("antennas back", mini)
            # Neutral (matches INIT_ANTENNAS ≈ ±10 deg)
            goto(mini, antennas=[math.radians(-10), math.radians(10)])
            report("antennas neutral", mini)

            section("Final settle to INIT")
            goto(mini)
            report("final", mini)

            print("\n  (daemon will run goto_sleep on shutdown)")
        return True
    except Exception as e:  # noqa: BLE001
        print(f"  MOTION FAILED: {type(e).__name__}: {e}")
        return False


def main() -> int:
    print("Reachy Mini RM1 step 3 — expressive motion demo")
    daemon = start_daemon(DAEMON_PORT)
    try:
        if not wait_for_port(DAEMON_PORT, timeout=DAEMON_STARTUP_TIMEOUT):
            print(f"  daemon did not open :{DAEMON_PORT} within {DAEMON_STARTUP_TIMEOUT}s")
            return 1
        print(f"  daemon listening on :{DAEMON_PORT}")
        ok = run_demo(DAEMON_PORT)
    finally:
        stop_daemon(daemon)

    section("Summary")
    print(f"  {'OK ' if ok else 'NG '} expressive motion demo")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
