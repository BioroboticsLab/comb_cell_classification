"""Parses camera id and capture timestamp out of annotation filenames (no correction happens here)."""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import Any

_CAM_RE = re.compile(r"cam-(\d+)")
# Filenames carry one of these stamp forms:
#   06_..._cam-0_20250701T112749.804969.329Z.json   ISO-ish, possibly several .<digits> fractional groups
#   ..._<start>Z--<end>Z.json                       time range (the first ISO stamp is the start)
#   02_20240606_cam-3.json                          older: bare date
_ISO_RE = re.compile(r"(\d{8})T(\d{6})((?:\.\d+)*)Z")
# Bare date = 8 digits not followed by 'T' (which would make it an ISO stamp).
_DATE_RE = re.compile(r"(?<!\d)(\d{8})(?!\dT)(?!T)")


def _parse_iso(date: str, time: str, frac: str) -> tuple[datetime, int]:
    """Build a datetime from the YYYYMMDD / HHMMSS / '.f1.f2' filename parts."""
    dt = datetime.strptime(date + time, "%Y%m%d%H%M%S")
    groups = [g for g in frac.split(".") if g]  # e.g. ['804969', '329']
    micro = 0
    extra = 0
    if groups:
        micro = int(groups[0][:6].ljust(6, "0"))
        extra_digits = "".join(groups[1:]) + groups[0][6:]
        extra = int(extra_digits) if extra_digits else 0
    return dt.replace(microsecond=micro), extra


def parse_timestamp(name: str) -> tuple[datetime, tuple[datetime, int]]:
    """Extract ``(timestamp, sort_key)`` from a filename, raising ValueError if no timestamp is found."""
    m = _ISO_RE.search(name)
    if m:
        dt, extra = _parse_iso(m.group(1), m.group(2), m.group(3))
        return dt, (dt, extra)

    m = _DATE_RE.search(name)
    if m:
        dt = datetime.strptime(m.group(1), "%Y%m%d")
        return dt, (dt, 0)

    raise ValueError(f"No timestamp found in filename {name!r}")


def parse_annotation_filename(name: str) -> dict[str, Any]:
    """Parse a filename or path into ``{cam, timestamp, sort_key}``."""
    name = Path(name).name

    cam_m = _CAM_RE.search(name)
    if not cam_m:
        raise ValueError(f"No 'cam-N' token in filename {name!r}")
    cam = f"cam-{cam_m.group(1)}"

    timestamp, sort_key = parse_timestamp(name)
    return {
        "cam": cam,
        "timestamp": timestamp,
        "sort_key": sort_key,
    }
