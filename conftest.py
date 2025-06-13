'''
import sys
import os

# Get the absolute path to the directory containing this conftest.py (project root)
project_root = os.path.dirname(os.path.abspath(__file__))

# Construct the absolute path to the 'src' directory
src_path = os.path.join(project_root, 'src')

# Add 'src' directory to the beginning of sys.path if it's not already there
if src_path not in sys.path:
    sys.path.insert(0, src_path)

# Add project_root to sys.path as well, if not already present (pytest usually does this, but good for explicitness)
if project_root not in sys.path:
    sys.path.insert(0, project_root)
'''
