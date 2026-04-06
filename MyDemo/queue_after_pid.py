from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Wait for a PID to exit, then run a command.")
    parser.add_argument("--wait-pid", type=int, required=True, help="PID to wait for.")
    parser.add_argument("--cwd", type=Path, default=Path.cwd(), help="Working directory for the queued command.")
    parser.add_argument("--log", type=Path, required=True, help="Log file for stdout/stderr.")
    parser.add_argument("--poll-seconds", type=float, default=60.0, help="Polling interval in seconds.")
    parser.add_argument("command", nargs=argparse.REMAINDER, help="Command to run after '--'.")
    return parser.parse_args()


def pid_exists(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def main() -> int:
    args = parse_args()
    command = args.command
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        raise SystemExit("No command provided. Use: queue_after_pid.py --wait-pid PID --log path -- <command>")

    args.log.parent.mkdir(parents=True, exist_ok=True)
    with args.log.open("a", encoding="utf-8") as f:
        f.write(f"[queue] waiting for pid {args.wait_pid}\n")
        f.flush()

        while pid_exists(args.wait_pid):
            time.sleep(args.poll_seconds)

        f.write(f"[queue] pid {args.wait_pid} exited, starting command:\n")
        f.write("[queue] " + " ".join(command) + "\n")
        f.flush()

        proc = subprocess.Popen(
            command,
            cwd=str(args.cwd),
            stdout=f,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        return proc.wait()


if __name__ == "__main__":
    sys.exit(main())
