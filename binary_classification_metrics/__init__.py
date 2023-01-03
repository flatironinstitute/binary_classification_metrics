# required module __init__

# compatibility hack
import numpy as np
if not hasattr(np, "int"):
    np.int = np.int64
    np.float = np.float64