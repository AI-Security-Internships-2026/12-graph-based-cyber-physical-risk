"""Seed-aware aggregation. Never mix incompatible configurations silently."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from src.evaluation.robustness_stats import descriptive_stats

def summarize_frame(df, keys, metrics=None):
    if df.empty:return df
    if "seed" in df:
        if df.duplicated(keys+["seed"]).any():
            raise ValueError(f"Duplicate seed/condition results for {keys}; reconcile artifacts first")
    if metrics is None:
        metrics=[c for c in df.select_dtypes(include="number").columns if c not in keys+["seed"]]
    rows=[]
    for key,g in df.groupby(keys,dropna=False,sort=False):
        key=key if isinstance(key,tuple) else (key,)
        row=dict(zip(keys,key));row["n_seeds"]=g["seed"].nunique() if "seed" in g else len(g)
        row["seeds"]=json.dumps(sorted(g["seed"].astype(int).unique().tolist())) if "seed" in g else "[]"
        if "config_id" in g and g.config_id.nunique()>1: raise ValueError("Incompatible configs in seed summary")
        for col in metrics:
            vals=pd.to_numeric(g[col],errors="coerce").dropna().to_numpy()
            if not len(vals):row[col]=np.nan;continue
            stats=descriptive_stats(vals);row[col]=stats["mean"]
            for stat in ("std","median","ci_95_low","ci_95_high","n"):row[f"{col}_{stat}"]=stats[stat]
        for col in g.columns:
            if col not in row and col not in metrics and col not in keys+["seed"]:
                nonnull=g[col].dropna()
                if len(nonnull) and nonnull.astype(str).nunique()==1: row[col]=nonnull.iloc[0]
        rows.append(row)
    return pd.DataFrame(rows)

def read_seed_frames(path):
    path=Path(path);frames=[]
    for f in sorted(path.parent.glob(path.stem+"_seed*.csv")):
        try:df=pd.read_csv(f)
        except pd.errors.EmptyDataError:continue
        if "seed" not in df:df["seed"]=int(f.stem.rsplit("_seed",1)[1])
        frames.append(df)
    if not frames:raise FileNotFoundError(f"No per-seed artifacts for {path}; run the experiment first")
    return pd.concat(frames,ignore_index=True)

def aggregate_file(path,keys,metrics=None):
    df=read_seed_frames(path)
    out=summarize_frame(df,keys,metrics)
    out.to_csv(path,index=False)
    return out
