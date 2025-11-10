# =========================================
# app.py — Comparateur Portefeuilles CGP
# - EODHD + VL synthétiques auto (selon horizon réel)
# - Fonds en euros simulé (taux paramétrable)
# - Coller un tableau / Import CSV
# - Versements mensuels et ponctuels
# - Comparaison Client vs Vous + delta gains
# =========================================
import os, re, math, requests, calendar
from datetime import date
from typing import Optional, List, Dict, Any, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

st.set_page_config(page_title="Comparateur Portefeuilles CGP", page_icon="🦉", layout="wide")
TODAY = pd.Timestamp.today().normalize()

# ---------- Utils ----------
def Secret_Token(name: str) -> str:
    v = os.getenv(name) or str(st.secrets.get(name, "")).strip()  # type: ignore[attr-defined]
    if not v:
        raise RuntimeError(f"Secret manquant: {name}")
    return v

def to_eur(x: float) -> str:
    try:
        if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
            return ""
        s = f"{float(x):,.2f}".replace(",", "X").replace(".", ",").replace("X", " ")
        return f"{s} €"
    except Exception:
        return ""

def fmt_date(d: pd.Timestamp | date | None) -> str:
    if d is None: return ""
    if isinstance(d, date) and not isinstance(d, pd.Timestamp):
        d = pd.Timestamp(d)
    return d.strftime("%d/%m/%Y")

# ---------- EODHD ----------
EODHD_BASE = "https://eodhd.com/api"

def eodhd_get(path: str, params: Optional[Dict[str, Any]] = None) -> Any:
    url = f"{EODHD_BASE.rstrip('/')}{path}"
    p = {"api_token": Secret_Token("EODHD_API_KEY"), "fmt": "json"}
    if params: p.update(params)
    r = requests.get(url, params=p, timeout=25)
    r.raise_for_status()
    js = r.json()
    if isinstance(js, str) and js.lower().startswith("error"):
        raise requests.HTTPError(js)
    return js

@st.cache_data(ttl=6*3600, show_spinner=False)
def eod_search(query: str) -> List[Dict[str, Any]]:
    try:
        js = eodhd_get(f"/search/{query.upper()}", params={})
        return js if isinstance(js, list) else []
    except Exception:
        return []

def _looks_like_isin(s: str) -> bool:
    return bool(re.fullmatch(r"[A-Z]{2}[A-Z0-9]{9}[0-9]", (s or "").strip().upper()))

def _eod_ok(sym: str) -> bool:
    try:
        js = eodhd_get(f"/eod/{sym}", params={"period": "d"})
        return isinstance(js, list) and len(js) > 0
    except Exception:
        return False

@st.cache_data(ttl=24*3600, show_spinner=False)
def resolve_symbol(q: str) -> Optional[str]:
    q = (q or "").strip()
    if not q: return None
    if _looks_like_isin(q):
        base = q.upper()
        for suf in [".EUFUND", ".FUND"]:
            cand = f"{base}{suf}"
            if _eod_ok(cand): return cand
        return None
    res = eod_search(q)
    if res:
        for it in res:
            code = str(it.get("Code","")).strip()
            if code and _eod_ok(code): return code
    return None

# ---------- Fonds euros ----------
def _eurofund_series(euro_rate: float,
                     start: pd.Timestamp = pd.Timestamp("1990-01-01"),
                     end: pd.Timestamp = TODAY) -> pd.Series:
    idx = pd.date_range(start=start, end=end, freq="D")
    vals = [1.0]
    for i in range(1, len(idx)):
        d = idx[i]
        v = vals[-1]
        if d.month == 12 and d.day == 31:
            v *= (1.0 + euro_rate/100.0)
        vals.append(v)
    return pd.Series(vals, index=idx, name="Close")

# ---------- VL via EODHD ----------
@st.cache_data(ttl=3*3600, show_spinner=False)
def eod_prices_any(symbol_or_isin: str,
                   start_dt: Optional[pd.Timestamp],
                   euro_rate: float) -> Tuple[pd.DataFrame, str, str]:
    q = (symbol_or_isin or "").strip()
    if not q:
        return pd.DataFrame(), q, "⚠️ identifiant vide."

    if q.upper() in {"EUROFUND", "FONDS EN EUROS"}:
        ser = _eurofund_series(euro_rate=euro_rate, start=pd.Timestamp("1990-01-01"), end=TODAY)
        if start_dt is not None:
            ser = ser.loc[ser.index >= start_dt]
        if not ser.empty: ser = ser / ser.iloc[0]
        return ser.to_frame(), "EUROFUND", f"Fonds en euros simulé ({euro_rate:.2f}%/an)"

    qU = q.upper()
    note = ""

    def _fetch(sym: str, from_dt: Optional[str]) -> pd.DataFrame:
        try:
            params={"period":"d"}
            if from_dt: params["from"]=from_dt
            js = eodhd_get(f"/eod/{sym}", params=params)
            df = pd.DataFrame(js)
            if df.empty: return pd.DataFrame()
            df["date"]=pd.to_datetime(df["date"]); df=df.set_index("date").sort_index()
            df["close"]=pd.to_numeric(df["close"], errors="coerce")
            return df[["close"]].rename(columns={"close":"Close"})
        except Exception:
            return pd.DataFrame()

    sym = resolve_symbol(qU) or qU
    df = _fetch(sym, None)
    if not df.empty:
        if start_dt is not None:
            df = df.loc[df.index >= start_dt]
        return df, sym, note

    return pd.DataFrame(), qU, "⚠️ Aucune VL récupérée."

@st.cache_data(ttl=3600, show_spinner=False)
def load_price_series_any(symbol_or_isin: str, from_dt: Optional[pd.Timestamp], euro_rate: float):
    return eod_prices_any(symbol_or_isin, from_dt, euro_rate)

# ---------- VL synthétique automatique ----------
if "SYNTH_PARAMS" not in st.session_state:
    st.session_state["SYNTH_PARAMS"] = {}

def _simulate_nav_series(start_dt: pd.Timestamp, end_dt: pd.Timestamp,
                         annual_perf_pct: float, vol_pct: float) -> pd.DataFrame:
    dates = pd.bdate_range(start=start_dt, end=end_dt)
    if len(dates) <= 1:
        return pd.DataFrame({"Close": [100.0]}, index=[start_dt])
    mu = (annual_perf_pct / 100.0) / 252.0
    sigma = (vol_pct / 100.0) / (252.0 ** 0.5)
    rnd = np.random.normal(mu, sigma, len(dates))
    prices = 100.0 * np.exp(np.cumsum(rnd))
    return pd.DataFrame({"Close": prices}, index=dates)

def _select_annual_rate(h_years: float,
                        p1: Optional[float], p3: Optional[float],
                        p5: Optional[float], p10: Optional[float]) -> float:
    if h_years <= 1.0 and p1 is not None: return p1
    if 1.0 < h_years <= 3.0 and p3 is not None: return p3
    if 3.0 < h_years <= 5.0 and p5 is not None: return p5
    if h_years > 5.0 and p10 is not None: return p10
    order = [(p, abs(h_years - ref)) for p, ref in [(p1,1),(p3,3),(p5,5),(p10,10)] if p is not None]
    if not order: return 0.0
    return sorted(order, key=lambda x:x[1])[0][0]

def set_synth_params(key: str, base_date: pd.Timestamp, vol_pct: float,
                     p1: float, p3: Optional[float], p5: Optional[float], p10: Optional[float]) -> None:
    st.session_state["SYNTH_PARAMS"][key.upper()] = {
        "base_date": base_date, "vol": vol_pct,
        "p1": p1, "p3": p3, "p5": p5, "p10": p10
    }

def get_synth_series(key: str, from_dt: Optional[pd.Timestamp]) -> pd.DataFrame:
    keyU = key.upper()
    params = st.session_state["SYNTH_PARAMS"].get(keyU)
    if not params:
        return pd.DataFrame()
    start = max(params["base_date"], from_dt or params["base_date"])
    horizon = max(1e-9, (TODAY - start).days / 365.25)
    rate = _select_annual_rate(horizon, params["p1"], params["p3"], params["p5"], params["p10"])
    return _simulate_nav_series(start, TODAY, rate, params["vol"])

def get_price_series(symbol_or_isin: str, from_dt: Optional[pd.Timestamp], euro_rate: float) -> Tuple[pd.DataFrame, str, str]:
    df, sym, note = load_price_series_any(symbol_or_isin, from_dt, euro_rate)
    if not df.empty:
        return df, sym, note
    sdf = get_synth_series(symbol_or_isin, from_dt)
    if not sdf.empty:
        return sdf, f"{symbol_or_isin.upper()}.SYNTH", "VL synthétique utilisée."
    return pd.DataFrame(), sym, note

# ---------- Barre latérale synthétique ----------
with st.sidebar:
    st.header("🧮 VL synthétique (si VL absente)")
    key = st.text_input("ISIN ou Nom du fonds")
    c1,c2 = st.columns(2)
    with c1:
        base_date = st.date_input("Date de départ", value=date(2018,1,1))
        vol = st.number_input("Volatilité annuelle (%)", min_value=0.0, max_value=50.0, value=10.0, step=0.5)
    with c2:
        p1 = st.number_input("Perf 1 an (%)", value=3.0, step=0.1)
        p3 = st.number_input("Perf 3 ans (%) (optionnel)", value=0.0, step=0.1)
        p5 = st.number_input("Perf 5 ans (%) (optionnel)", value=0.0, step=0.1)
        p10 = st.number_input("Perf 10 ans (%) (optionnel)", value=0.0, step=0.1)
    p3 = None if p3 == 0 else p3
    p5 = None if p5 == 0 else p5
    p10 = None if p10 == 0 else p10
    if st.button("✅ Enregistrer la VL synthétique"):
        if not key:
            st.warning("Indique un ISIN ou un nom.")
        else:
            set_synth_params(key, pd.Timestamp(base_date), vol, p1, p3, p5, p10)
            st.success(f"Paramètres enregistrés pour {key.upper()}")

# (Le reste du code — import CSV, ajout lignes, simulation, graphique, delta de gains, etc. — reste inchangé)
