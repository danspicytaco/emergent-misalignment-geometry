#!/usr/bin/env python3
"""Fresh independent all-file readback in bounded disposable batches."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from em_geometry.data import main

if __name__ == "__main__":
    main("verify")
