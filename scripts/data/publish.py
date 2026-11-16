#!/usr/bin/env python3
"""Dry-run by default; --upload explicitly enables HF dataset writes."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from em_geometry.data import main

if __name__ == "__main__":
    main("publish")
