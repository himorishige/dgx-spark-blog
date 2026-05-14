"""RM1 step 1: Reachy Mini Wired connectivity smoke test (read-only).

The daemon is started explicitly as a subprocess on a configurable port so we
can avoid clashing with anything that may already be holding :8000. No motor
movement is performed.
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

DAEMON_PORT = 8765
DAEMON_STARTUP_TIMEOUT = 60.0  # daemon may need ~30s for motor handshake


def section(title: str) -> None:
    print(f"\n{'=' * 8} {title} {'=' * 8}")


def check_os_level() -> bool:
    """Verify /dev nodes exist and have the expected permissions."""
    section("OS-level device check")
    ok = True

    serial = Path("/dev/ttyACM0")
    if serial.exists():
        writable = os.access(serial, os.W_OK)
        print(f"  serial : {serial} exists, writable={writable}")
        ok = ok and writable
    else:
        print(f"  serial : {serial} NOT FOUND")
        ok = False

    for vid in ("/dev/video0", "/dev/video1"):
        p = Path(vid)
        print(f"  camera : {p} {'exists' if p.exists() else 'NOT FOUND'}")

    return ok


def check_sdk_import() -> bool:
    section("SDK import")
    try:
        import reachy_mini

        print(f"  reachy_mini version : {reachy_mini.__version__}")
        return True
    except Exception as e:  # noqa: BLE001
        print(f"  IMPORT FAILED: {e!r}")
        return False


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
        "--no-media",  # skip webrtcsink (GStreamer rust plugin not installed)
        "--headless",
        "--log-level",
        "WARNING",
    ]
    print(f"  cmd : {' '.join(cmd)}")
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        preexec_fn=os.setsid,  # so we can kill the whole process group
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
        print("  daemon did not exit in 10s, sending SIGKILL")
        with contextlib.suppress(ProcessLookupError):
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        proc.wait()


def check_connection(port: int) -> bool:
    section("SDK live connection (read-only)")

    from reachy_mini import ReachyMini

    try:
        with ReachyMini(
            host="127.0.0.1",
            port=port,
            connection_mode="localhost_only",
            spawn_daemon=False,
            log_level="WARNING",
            media_backend="none",
        ) as mini:
            print("  client connected to daemon")
            time.sleep(0.5)

            pose = mini.get_current_head_pose()
            print(f"  head pose shape    : {pose.shape}")
            print(f"  head xyz (raw)     : {pose[:3, 3].tolist()}")

            head_joints, antenna_joints = mini.get_current_joint_positions()
            print(f"  head joints (6)    : {[round(j, 3) for j in head_joints]}")
            print(f"  antenna joints (2) : {[round(j, 3) for j in antenna_joints]}")

            try:
                imu_obj = mini.imu
                print(f"  imu accessor       : {type(imu_obj).__name__}")
            except Exception as e:  # noqa: BLE001
                print(f"  imu read skipped   : {e!r}")

        print("  client disconnected cleanly")
        return True
    except Exception as e:  # noqa: BLE001
        print(f"  CONNECTION FAILED: {type(e).__name__}: {e}")
        return False


def main() -> int:
    print("Reachy Mini RM1 step 1 — connectivity smoke test")

    results = {"os_level": check_os_level(), "sdk_import": check_sdk_import()}
    if not results["sdk_import"]:
        results["daemon"] = False
        results["connection"] = False
    else:
        daemon = start_daemon(DAEMON_PORT)
        try:
            if wait_for_port(DAEMON_PORT, timeout=DAEMON_STARTUP_TIMEOUT):
                results["daemon"] = True
                print(f"  daemon listening on :{DAEMON_PORT}")
                results["connection"] = check_connection(DAEMON_PORT)
            else:
                results["daemon"] = False
                results["connection"] = False
                print(f"  daemon did not open :{DAEMON_PORT} within {DAEMON_STARTUP_TIMEOUT}s")
                # Drain captured output for debugging.
                try:
                    out, _ = daemon.communicate(timeout=2)
                    if out:
                        print("  --- daemon stdout/stderr (tail) ---")
                        for line in out.splitlines()[-30:]:
                            print(f"  {line}")
                except subprocess.TimeoutExpired:
                    pass
        finally:
            stop_daemon(daemon)

    section("Summary")
    for key, value in results.items():
        marker = "OK " if value else "NG "
        print(f"  {marker} {key}")

    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
