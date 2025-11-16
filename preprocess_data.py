#!/usr/bin/env python3
"""
Preprocess TuSimple dataset to Bezier format
"""
import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from src.data.preprocess_tusimple_bezier import main

if __name__ == "__main__":
    main()
