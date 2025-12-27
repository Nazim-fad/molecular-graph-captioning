import sys,traceback,importlib
importlib.invalidate_caches()
try:
    import torch
    print('torch', torch.__version__)
    import generate_description_embeddings
except Exception:
    traceback.print_exc()
    sys.exit(1)
