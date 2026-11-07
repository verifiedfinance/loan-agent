"""Put src/ and its sub-folders on the import path, as the repository's modules do."""
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / 'src'
sys.path[:0] = [p for p in [str(SRC)] + [str(d) for d in sorted(SRC.iterdir()) if d.is_dir() and not d.name.startswith(('_', '.'))]
                if p not in sys.path]
