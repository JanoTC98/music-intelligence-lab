from __future__ import annotations

import json

from spotify_intelligence.data.pipeline import prepare_data

if __name__ == "__main__":
    manifest = prepare_data()
    print(json.dumps(manifest, indent=2))
