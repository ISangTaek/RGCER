"""Durable command/exit evidence for the V9-C0 WSL/server handoff."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time


def write(path, value):
    with path.open('x', encoding='utf8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)


def capture(prefix, command):
    prefix = Path(prefix).resolve()
    if not command:
        raise ValueError('command required')
    prefix.parent.mkdir(parents=True, exist_ok=True)
    started = Path(str(prefix) + '.started.json')
    write(started, dict(command=command, cwd=str(Path.cwd()),
                        utc=datetime.now(timezone.utc).isoformat()))
    clock = time.monotonic()
    code = None
    error = None
    try:
        with Path(str(prefix)+'.stdout.log').open('xb') as out, Path(str(prefix)+'.stderr.log').open('xb') as err:
            code = subprocess.run(command, stdout=out, stderr=err).returncode
    except BaseException as exc:
        error = dict(type=type(exc).__name__, message=str(exc))
        raise
    finally:
        write(Path(str(prefix)+'.exit.json'), dict(exit_code=code, error=error,
            elapsed_seconds=time.monotonic()-clock, utc=datetime.now(timezone.utc).isoformat()))
    return code


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prefix', type=Path, required=True)
    p.add_argument('command', nargs=argparse.REMAINDER)
    a = p.parse_args()
    command = a.command[1:] if a.command[:1] == ['--'] else a.command
    sys.exit(capture(a.prefix, command))
