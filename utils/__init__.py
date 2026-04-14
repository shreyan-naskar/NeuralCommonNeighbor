"""
Shared utilities for CNT experiment notebooks.
Adds the CNT project root to sys.path so model.py / utils.py / ogbdataset.py
are importable from any notebook inside experiments/notebooks/.
"""
import sys, os

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
