"""Exact, replayable split manifests and source provenance (no pickle)."""
import functools
import hashlib
import inspect
import json
import os
from pathlib import Path
import numpy as np
import pandas as pd
import torch

USED_MANIFESTS = set()

def _encode(x):
    if torch.is_tensor(x):
        return {"__tensor__": x.detach().cpu().tolist(), "dtype": str(x.dtype).split(".")[-1]}
    if isinstance(x, np.ndarray):
        return {"__array__": x.tolist(), "dtype": str(x.dtype)}
    if isinstance(x, tuple): return {"__tuple__": [_encode(v) for v in x]}
    if isinstance(x, list): return [_encode(v) for v in x]
    if isinstance(x, dict): return {str(k): _encode(v) for k,v in x.items()}
    if isinstance(x, np.generic): return x.item()
    return x

def _decode(x):
    if isinstance(x, list): return [_decode(v) for v in x]
    if not isinstance(x, dict): return x
    if "__tensor__" in x: return torch.tensor(x["__tensor__"], dtype=getattr(torch,x["dtype"]))
    if "__array__" in x: return np.asarray(x["__array__"], dtype=x["dtype"])
    if "__tuple__" in x: return tuple(_decode(v) for v in x["__tuple__"])
    return {k:_decode(v) for k,v in x.items()}

def fingerprint(x):
    if isinstance(x, pd.DataFrame):
        payload = str(list(zip(x.columns, map(str,x.dtypes)))).encode()+pd.util.hash_pandas_object(x,index=True).values.tobytes()
    else: payload=json.dumps(_encode(x),sort_keys=True,default=str).encode()
    return hashlib.sha256(payload).hexdigest()

def replayable_split(fn):
    """Read exact saved positions on subsequent calls with identical data/config."""
    @functools.wraps(fn)
    def wrapped(*args,**kwargs):
        bound=inspect.signature(fn).bind(*args,**kwargs); bound.apply_defaults()
        inputs={k:fingerprint(v) for k,v in bound.arguments.items()}
        key=fingerprint({"function":fn.__module__+"."+fn.__name__,"inputs":inputs,
                         "implementation":inspect.getsource(fn)})
        root=Path(os.environ.get("SPLIT_MANIFEST_DIR","experiments/results/q1/splits")); root.mkdir(parents=True,exist_ok=True)
        path=root/f"{fn.__name__}_{key}.json"
        USED_MANIFESTS.add(str(path))
        if path.exists():
            rec=json.loads(path.read_text())
            if fingerprint(rec["result"]) != rec["result_sha256"]: raise ValueError(f"Corrupt split manifest: {path}")
            return _decode(rec["result"])
        result=fn(*args,**kwargs); encoded=_encode(result)
        path.write_text(json.dumps({"split_id":key,"function":fn.__name__,"input_fingerprints":inputs,
                         "result":encoded,"result_sha256":fingerprint(encoded)},indent=2))
        return result
    return wrapped

def source_hashes(root):
    paths=list((root/"src").rglob("*.py"))+list((root/"experiments").glob("*.py"))
    return {str(f.relative_to(root)):hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(paths)}
