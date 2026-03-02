import os
import sys


def _force_utf8_io() -> None:
    # Ensure Python and child processes default to UTF-8.
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    os.environ.setdefault("PYTHONUTF8", "1")

    for name in ("stdout", "stderr"):
        stream = getattr(sys, name, None)
        if stream is None:
            continue
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            # Some environments don't support reconfigure or stream is closed.
            pass


_force_utf8_io()
