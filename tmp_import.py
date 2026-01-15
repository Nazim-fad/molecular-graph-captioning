import sys,traceback,importlib
importlib.invalidate_caches()
sys.path.append(".")
try:
    import torch
    print('torch', torch.__version__)
    import scripts.preprocess.generate_description_embeddings as generate_description_embeddings
    print("Successfully imported: scripts.generate_description_embeddings")
    import src.model_gcn as model_gcn
    print("Successfully imported: src.model_gcn")
    import src.data_utils as data_utils
    print("Successfully imported: src.data_utils")
except Exception:
    traceback.print_exc()
    sys.exit(1)


