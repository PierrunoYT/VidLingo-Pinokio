"""Regenerate `app/requirements.lock.txt` from `app/requirements.txt`.

Run from anywhere:

    python tools/relock.py

Two passes are needed. PyTorch and its CUDA/Triton stack must not appear in
the lock — `torch.js` installs the platform-specific build (CUDA, ROCm,
DirectML, MPS, or CPU) before the lock is installed, and a generic pin here
would force-replace it with a redundant multi-gigabyte download. But their
names are not known up front, so the first pass resolves everything and the
second pass re-resolves while excluding whatever Torch dragged in. Both passes
are constrained to the Torch version `torch.js` installs, so the locked
versions are consistent with the Torch that will actually be present.
"""

from __future__ import annotations

import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REQUIREMENTS = ROOT / "app" / "requirements.txt"
CONSTRAINT = ROOT / "app" / "torch-constraint.txt"
LOCKFILE = ROOT / "app" / "requirements.lock.txt"

PYTHON_VERSION = "3.10"
EXCLUDE_RE = re.compile(r"^(torch|triton|nvidia-[a-z0-9-]+)", re.IGNORECASE)

HEADER = """\
# Resolved dependency lock for VidLingo — do NOT edit by hand.
#
# `app/requirements.txt` is the input spec; this file is the resolved set that
# install.js and update.js actually install, so two machines installing on
# different days get the same packages.
#
# Regenerate after changing requirements.txt:
#
#     python tools/relock.py
#
# PyTorch and its CUDA/Triton stack are deliberately excluded: torch.js
# installs the platform-specific build (CUDA, ROCm, DirectML, MPS, or CPU)
# before this file is installed, and letting the resolver pin a generic torch
# here would force-replace it with a redundant multi-gigabyte download. The
# resolution is still constrained to the torch version torch.js installs
# (app/torch-constraint.txt), so everything below is consistent with it.
#
# Universal resolution: markers cover Python {python_version}+ on all supported
# platforms.
"""


def _compile(out_path: Path, exclude: list[str]) -> str:
    cmd = [
        "uv",
        "pip",
        "compile",
        "--universal",
        "--no-header",
        "--python-version",
        PYTHON_VERSION,
        "-c",
        str(CONSTRAINT),
    ]
    for name in exclude:
        cmd += ["--no-emit-package", name]
    cmd += [str(REQUIREMENTS), "-o", str(out_path)]
    result = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    if result.returncode != 0:
        sys.stderr.write(result.stderr)
        raise SystemExit(result.returncode)
    return out_path.read_text(encoding="utf-8")


def main() -> int:
    if not REQUIREMENTS.is_file():
        print(f"missing {REQUIREMENTS}", file=sys.stderr)
        return 1

    with tempfile.TemporaryDirectory() as tmp:
        probe = Path(tmp) / "full.lock"
        resolved = _compile(probe, exclude=[])
        excluded = sorted(
            {
                match.group(1).lower()
                for line in resolved.splitlines()
                if (match := EXCLUDE_RE.match(line))
            }
        )
        print(f"excluding {len(excluded)} torch/CUDA packages: {' '.join(excluded)}")

        final = Path(tmp) / "app.lock"
        body = _compile(final, exclude=excluded)

    header = HEADER.format(python_version=PYTHON_VERSION)
    LOCKFILE.write_text(header + body, encoding="utf-8")
    print(f"wrote {LOCKFILE.relative_to(ROOT)} ({len(body.splitlines())} entries)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
