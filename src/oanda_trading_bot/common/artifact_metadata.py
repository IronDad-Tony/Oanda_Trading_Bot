"""
Helpers for reading and writing sidecar metadata files for trained artifacts.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional
from datetime import datetime, timezone


def metadata_path_for(artifact_path: Path) -> Path:
    return artifact_path.with_suffix(artifact_path.suffix + ".meta.json")


def write_metadata(artifact_path: Path, metadata: Dict[str, Any]) -> None:
    artifact_path = Path(artifact_path)
    meta_path = metadata_path_for(artifact_path)
    payload = {
        "artifact": str(artifact_path),
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        **metadata,
    }
    meta_path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def read_metadata(artifact_path: Path) -> Optional[Dict[str, Any]]:
    meta_path = metadata_path_for(Path(artifact_path))
    if not meta_path.exists():
        return None
    try:
        return json.loads(meta_path.read_text())
    except json.JSONDecodeError:
        return None
