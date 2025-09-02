import sys
from pathlib import Path

HERE = Path(__file__).resolve()
ENT_DIR = HERE.parents[1]          
API_DIR = ENT_DIR.parent          

if str(ENT_DIR) not in sys.path:
    sys.path.insert(0, str(ENT_DIR))

if str(API_DIR) not in sys.path:
    sys.path.insert(0, str(API_DIR))