#!/usr/bin/env python3
"""
Main training entry point for Lane Detection Model
"""
import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from src.training.train import main

if __name__ == "__main__":
    main()
