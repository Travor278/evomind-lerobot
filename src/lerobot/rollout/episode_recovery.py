"""Emergency numeric salvage; never advertised as a complete trainable episode."""
from __future__ import annotations

import json
from pathlib import Path
import uuid

import numpy as np


def preserve_failed_episode(root, buffer: dict, error: Exception) -> Path:
    root = Path(root)
    folder = root / "recovery" / uuid.uuid4().hex
    folder.mkdir(parents=True, exist_ok=False)
    arrays, unavailable = {}, []
    for key, value in buffer.items():
        try:
            array = np.asarray(value)
            if array.dtype.kind == "O":
                unavailable.append(key)
            else:
                arrays[key] = array
        except (TypeError, ValueError):
            unavailable.append(key)
    np.savez(folder / "frames.npz", **arrays)
    manifest = {
        "format": "evomind-emergency-episode-v1", "trainable": False,
        "error": repr(error), "frames": int(buffer.get("size", 0)),
        "unavailable_fields": unavailable,
        "video_files": [str(p.relative_to(root)) for p in root.rglob("*.mp4")],
        "note": "Verify video alignment before recovery; missing frames are not fabricated.",
    }
    (folder / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    return folder
