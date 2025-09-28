import pandas as pd
import numpy as np
from datetime import time
from scipy.optimize import minimize_scalar

def load_quotes_csv(path):
    df = pd.read_csv(path)
    return df

def clean_quotes(df):
    keep = df["EX"].isin(["T","P","Z"])
    df = df.loc[keep].copy()
    df["ts"] = pd.to_datetime(df["DATE"].astype(str) + " " + df["TIME_M"], utc=True, errors="coerce")
    df["ts"] = df["ts"].dt.floor("s")
    df = df.sort_values(["SYM_ROOT","EX","ts"])
    mask = (df["ts"].dt.time >= time(10,0,0)) & (df["ts"].dt.time <= time(15,59,59))
    df = df.loc[mask].copy()
    df = df[(df["ASK"] > 0) & (df["BID"] > 0)].copy()
    df["spread"] = df["ASK"] - df["BID"]
    df = df[df["spread"] > 0].copy()
    df["spread_cents"] = np.round(df["spread"] * 100).astype("Int64")
    df = df.rename(columns={"SYM_ROOT":"SYMBOL","ASK":"OFR","OFRSIZ":"ASKSIZ"})
    # unify column names
    out = df[["SYMBOL","EX","ts","BID","OFR","BIDSIZ","ASKSIZ","spread","spread_cents"]].copy()
    out["TIME"] = out["ts"].dt.strftime("%H:%M:%S")
    return out

def add_size_deciles(df):
    def decile_col(x):
        return pd.qcut(x, q=10, labels=False, duplicates="drop") + 1
    df = df.copy()
    df["BIDSIZ_dec"] = decile_col(df["BIDSIZ"])
    df["ASKSIZ_dec"] = decile_col(df["ASKSIZ"])
    return df

def empirical_uij(df_symbol):
    sub = df_symbol.copy()
    sub = add_size_deciles(sub)
    sub["mid"] = 0.5 * (sub["BID"] + sub["OFR"])
    sub["mid_next"] = sub.groupby(["SYMBOL","EX"])["mid"].shift(-1)
    sub["chg"] = sub["mid_next"] - sub["mid"]
    next_sign = np.sign(sub["chg"]).astype(float)
    up_next = (next_sign > 0).astype(float)
    m = pd.DataFrame({
        "bid_size_dec": sub["BIDSIZ_dec"].to_numpy(),
        "ask_size_dec": sub["ASKSIZ_dec"].to_numpy(),
        "up_next": up_next
    }).dropna(subset=["bid_size_dec","ask_size_dec","up_next"])
    uij = (m.groupby(["ask_size_dec","bid_size_dec"])["up_next"]
             .mean()
             .unstack("ask_size_dec")
             .sort_index()).fillna(0.0)
    # enforce a square grid ten by ten
    idx = pd.Index(range(1,11), name="bid_size_dec")
    cols = pd.Index(range(1,11), name="ask_size_dec")
    uij = uij.reindex(index=idx, columns=cols, fill_value=0.0)
    return uij

def dij_distribution(df_symbol):
    sub = df_symbol.copy()
    sub = add_size_deciles(sub)
    m = pd.DataFrame({
        "bid_size_dec": sub["BIDSIZ_dec"].to_numpy(),
        "ask_size_dec": sub["ASKSIZ_dec"].to_numpy(),
        "one": 1.0
    }).dropna(subset=["bid_size_dec","ask_size_dec"])
    dij = (m.groupby(["ask_size_dec","bid_size_dec"])["one"]
             .sum()
             .unstack("ask_size_dec")
             .sort_index()).fillna(0.0)
    dij = dij / dij.to_numpy().sum()
    idx = pd.Index(range(1,11), name="bid_size_dec")
    cols = pd.Index(range(1,11), name="ask_size_dec")
    dij = dij.reindex(index=idx, columns=cols, fill_value=0.0)
    return dij

def model_uij(h):
    I = np.arange(1, 11)[:, None]
    J = np.arange(1, 11)[None, :]
    U = (I + h) / (I + J + 2.0*h + 1e-12)
    idx = pd.Index(range(1,11), name="bid_size_dec")
    cols = pd.Index(range(1,11), name="ask_size_dec")
    return pd.DataFrame(U, index=idx, columns=cols)

def fit_h(uij, dij):
    # loss is weighted square error using dij as weights
    def loss(h):
        Uhat = model_uij(h).to_numpy()
        Uemp = uij.to_numpy()
        W = dij.to_numpy()
        L = np.mean(((Uhat - Uemp)**2) * W)
        return L
    # restrict h to positive values to avoid numeric issues while staying expressive
    res = minimize_scalar(loss, bounds=(0.001, 20.0), method="bounded")
    return float(res.x), float(res.fun)

def run_pipeline(quotes_csv, symbol):
    df = load_quotes_csv(quotes_csv)
    df = clean_quotes(df)
    sym = df[df["SYMBOL"] == symbol].copy()
    if len(sym) == 0:
        raise ValueError("no rows for the requested symbol")
    uij = empirical_uij(sym)
    dij = dij_distribution(sym)
    h, loss = fit_h(uij, dij)
    out = {
        "symbol": symbol,
        "implied_h": h,
        "loss": loss,
        "u_empirical": uij,
        "u_model": model_uij(h),
        "d_weights": dij,
    }
    return out