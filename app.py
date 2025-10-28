# ===============================
# 01) IMPORTS & SETUP
# ===============================
import os
from typing import Optional, List, Dict, Any, Tuple

import requests
import numpy as np
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

load_dotenv()
st.set_page_config(page_title="Allocation CGP — Analyse UC (EODHD)", page_icon="🦉", layout="wide")


# ===============================
# 02) SECRET TOKEN HELPER
# ===============================
def Secret_Token(name: str, default: Optional[str] = None) -> str:
    """
    Récupère un secret depuis l'environnement (prioritaire) ou st.secrets.
    Lève une erreur si absent et pas de valeur par défaut fournie.
    """
    v = os.getenv(name)
    if v and v.strip():
        return v.strip()
    try:
        v = st.secrets.get(name)  # type: ignore[attr-defined]
        if v and str(v).strip():
            return str(v).strip()
    except Exception:
        pass
    if default is not None:
        return default
    raise RuntimeError(f"Secret '{name}' is missing. Provide it via environment or st.secrets.")


# ===============================
# 03) EODHD — CLIENT LÉGER (+ mapping renforcé)
# ===============================
def eodhd_base_url() -> str:
    return os.getenv("EODHD_BASE_URL") or st.secrets.get("EODHD_BASE_URL", "https://eodhd.com/api")

def eodhd_headers() -> Dict[str, str]:
    return {"Accept": "application/json"}

def eodhd_params(extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    params = {"fmt": "json", "api_token": Secret_Token("EODHD_API_KEY")}
    if extra:
        params.update(extra)
    return params

@st.cache_data(ttl=6 * 3600, show_spinner=False)
def eodhd_get(path: str, params: Optional[Dict[str, Any]] = None) -> Any:
    url = f"{eodhd_base_url().rstrip('/')}{path}"
    r = requests.get(url, params=eodhd_params(params or {}), headers=eodhd_headers(), timeout=25)
    r.raise_for_status()
    return r.json()

@st.cache_data(ttl=24 * 3600, show_spinner=False)
def eodhd_search(query: str, limit: int = 20) -> List[Dict[str, Any]]:
    """
    Recherche générique EODHD: /search/{query}
    Retourne toujours une liste de dicts (normalisée).
    """
    try:
        data = eodhd_get(f"/search/{query}", params={"limit": limit})
    except Exception:
        return []
    if isinstance(data, dict) and "data" in data:
        return data["data"] or []
    return data if isinstance(data, list) else []

def _field(d: Dict[str, Any], *names: str) -> Optional[str]:
    """Renvoie la 1ère clé présente parmi names (insensible à la casse de la 1ère lettre)."""
    for n in names:
        if n in d and d[n]:
            return str(d[n])
        N = n[0].upper() + n[1:]
        if N in d and d[N]:
            return str(d[N])
    return None

def pick_best_ticker_from_search(items: List[Dict[str, Any]], isin: Optional[str] = None) -> Optional[str]:
    """
    Heuristique de sélection:
      1) match 'isin' strict si dispo
      2) sinon privilégie un exchange EU
      3) sinon premier item
    Ticker = priorité 'code' puis 'symbol' puis 'ticker'/'Ticker'
    """
    if not items:
        return None
    if isin:
        for it in items:
            if str(_field(it, "isin") or "").upper() == isin.upper():
                return _field(it, "code", "symbol", "ticker", "Ticker")
    eu_ex = {"PAR", "XETRA", "MIL", "AMS", "LSE", "VIE", "MAD", "BRU", "LIS", "VTX"}
    eu = [it for it in items if str(_field(it, "exchange") or "").upper() in eu_ex]
    if eu:
        return _field(eu[0], "code", "symbol", "ticker", "Ticker")
    return _field(items[0], "code", "symbol", "ticker", "Ticker")

@st.cache_data(ttl=6 * 3600, show_spinner=False)
def eodhd_fundamentals(ticker_or_isin: str) -> Dict[str, Any]:
    """
    Fundamentals généraux. EODHD accepte parfois un ISIN directement ici.
    Utile en fallback pour récupérer un 'Code' ou 'Ticker'.
    """
    try:
        js = eodhd_get(f"/fundamentals/{ticker_or_isin}")
        return js if isinstance(js, dict) else {}
    except Exception:
        return {}

def find_ticker_best_effort(name: str, isin: Optional[str]) -> Tuple[Optional[str], Dict[str, Any]]:
    """
    Stratégie robuste pour retrouver le ticker:
      A) search(ISIN) → pick_best
      B) fundamentals(ISIN) → 'General'.Code / 'General'.Ticker / Symbol
      C) search(name) → pick_best (en filtrant par isin si présent)
    Renvoie (ticker, debug_dict)
    """
    debug = {"search_isin": None, "fundamentals_isin": None, "search_name": None}

    # A) via /search/ISIN
    if isin:
        items = eodhd_search(isin, limit=25)
        debug["search_isin"] = items
        tick = pick_best_ticker_from_search(items, isin=isin)
        if tick:
            return tick, debug

    # B) via /fundamentals/ISIN (parfois renvoie General.Code utilisable sur /eod)
    if isin:
        f = eodhd_fundamentals(isin)
        debug["fundamentals_isin"] = f
        gen = f.get("General") or {}
        tick = _field(gen, "Code", "Ticker", "Symbol")
        if tick:
            return tick, debug

    # C) via /search/name
    items2 = eodhd_search(name, limit=25)
    debug["search_name"] = items2
    # si des items portent le même ISIN dans leurs champs → on prend celui-là
    if isin:
        for it in items2:
            if str(_field(it, "isin") or "").upper() == isin.upper():
                return _field(it, "code", "symbol", "ticker", "Ticker"), debug
    return pick_best_ticker_from_search(items2, isin=None), debug

@st.cache_data(ttl=6 * 3600, show_spinner=False)
def eodhd_prices_daily(ticker: str, days: int = 450) -> pd.DataFrame:
    """Séries de clôtures quotidiennes /eod/{ticker}?period=d"""
    js = eodhd_get(f"/eod/{ticker}", params={"period": "d"})
    df = pd.DataFrame(js)
    if df.empty or "close" not in df.columns:
        return pd.DataFrame()
    dcol = "date" if "date" in df.columns else "Date"
    df[dcol] = pd.to_datetime(df[dcol])
    df = df.set_index(dcol).sort_index()
    df = df.tail(days)[["close"]].rename(columns={"close": "Close"})
    return df

def perf_series(prices: pd.DataFrame) -> Dict[str, Optional[float]]:
    """Perf % (1M/3M/6M/YTD/1Y) sur la base du dernier close. Renvoie None si pas calculable."""
    out = {"1M": None, "3M": None, "6M": None, "YTD": None, "1Y": None}
    if prices.empty:
        return out
    last = prices["Close"].iloc[-1]
    if last is None or last <= 0:
        return out

    def pct(dt_from: pd.Timestamp) -> Optional[float]:
        s = prices.loc[:dt_from]
        if s.empty:
            return None
        base = s["Close"].iloc[-1]
        if base is None or base <= 0:
            return None
        return (last / base - 1.0) * 100.0

    idx = prices.index
    end = idx[-1]
    try:
        out["1M"]  = pct(end - pd.DateOffset(months=1))
        out["3M"]  = pct(end - pd.DateOffset(months=3))
        out["6M"]  = pct(end - pd.DateOffset(months=6))
        out["YTD"] = pct(pd.Timestamp(year=end.year, month=1, day=1, tz=end.tz))
        out["1Y"]  = pct(end - pd.DateOffset(years=1))
    except Exception:
        pass
    return out


# ===============================
# 04) UNIVERS — ESPACE INVEST 5
# ===============================
UNIVERSE: List[Dict[str, Any]] = [
    {"name": "R-co Valor C EUR", "isin": "FR0011253624", "sri": 4, "sfdr": 8, "type": "UC Actions Monde",
     "vl": False, "vlp": False, "transferts_programmes_eligibles": True, "notes": ""},
    {"name": "Vivalor International", "isin": "FR0014001LS1", "sri": 4, "sfdr": None, "type": "UC Actions Monde",
     "vl": False, "vlp": False, "transferts_programmes_eligibles": False, "notes": "Non éligible aux transferts programmés"},
    {"name": "COMGEST Monde C", "isin": "FR0000284689", "sri": 4, "sfdr": 8, "type": "UC Actions Monde",
     "vl": False, "vlp": False, "transferts_programmes_eligibles": True, "notes": ""},
    {"name": "Echiquier World Equity Growth", "isin": "FR0010859769", "sri": 4, "sfdr": 8, "type": "UC Actions Monde",
     "vl": False, "vlp": False, "transferts_programmes_eligibles": True, "notes": ""},
    {"name": "Franklin Mutual Global Discovery", "isin": "LU0211333298", "sri": 4, "sfdr": 8, "type": "UC Actions Monde",
     "vl": False, "vlp": False, "transferts_programmes_eligibles": True, "notes": ""},
    {"name": "CARMIGNAC INVESTISSEMENT A EUR", "isin": "FR0010148981", "sri": 4, "sfdr": 8, "type": "UC Actions Monde",
     "vl": False, "vlp": False, "transferts_programmes_eligibles": True, "notes": ""},
    {"name": "Natixis - Thematics Meta A EUR", "isin": "LU1951204046", "sri": 5, "sfdr": 8, "type": "UC Thématique Monde",
     "vl": False, "vlp": False, "transferts_programmes_eligibles": True, "notes": ""},
    {"name": "Pictet Global Megatrend Selection P", "isin": "LU0386882277", "sri": 4, "sfdr": 8, "type": "UC Thématique Monde",
     "vl": False, "vlp": False, "transferts_programmes_eligibles": True, "notes": ""},
    {"name": "Morgan Stanley Gl Brands A", "isin": "LU0119620416", "sri": 4, "sfdr": 8, "type": "UC Actions Monde",
     "vl": False, "vlp": False, "transferts_programmes_eligibles": True, "notes": ""},
    {"name": "FIDELITY FUNDS - WORLD FUND", "isin": "LU0069449576", "sri": 4, "sfdr": 8, "type": "UC Actions Monde",
     "vl": False, "vlp": False, "transferts_programmes_eligibles": True, "notes": ""},
    {"name": "CLARTAN VALEURS", "isin": "LU1100076550", "sri": 4, "sfdr": 8, "type": "UC Actions Monde",
     "vl": False, "vlp": False, "transferts_programmes_eligibles": True, "notes": ""},
    {"name": "CARMIGNAC PATRIMOINE", "isin": "FR0010135103", "sri": 3, "sfdr": 8, "type": "UC Diversifié (patrimonial)",
     "vl": False, "vlp": False, "transferts_programmes_eligibles": True, "notes": ""},
    {"name": "SYCOYIELD 2030 RC", "isin": "FR001400MCQ6", "sri": 2, "sfdr": 8, "type": "Obligataire daté 2030",
     "vl": False, "vlp": False, "transferts_programmes_eligibles": True, "notes": ""},
    {"name": "R-Co Target 2029 HY", "isin": None, "sri": None, "sfdr": None, "type": "Obligataire daté 2029 HY",
     "vl": False, "vlp": False, "transferts_programmes_eligibles": True, "notes": "ISIN à compléter"},
    {"name": "Fonds en euros AGGV", "isin": None, "sri": 1, "sfdr": None, "type": "Fonds en euros",
     "vl": False, "vlp": False, "transferts_programmes_eligibles": True, "notes": ""},
]


# ===============================
# 05) TITRES & SIDEBAR
# ===============================
st.title("🦉 Analyse UC — Espace Invest 5 (via EODHD)")
st.caption("Aucune allocation calculée. Analyse variations & infos techniques via EODHD.")

with st.sidebar:
    st.header("Clé API EODHD")
    try:
        _ = Secret_Token("EODHD_API_KEY")
        st.success("Clé EODHD détectée")
    except Exception:
        st.error("Clé EODHD manquante — ajoutez EODHD_API_KEY dans secrets/env.")
    debug_mode = st.toggle("Mode debug EODHD", value=False, help="Affiche les retours bruts des endpoints.")


# ===============================
# 06) SÉLECTION & PARAMS
# ===============================
st.subheader("Sélection des fonds à analyser")
df_univ = pd.DataFrame(UNIVERSE)

choices = st.multiselect(
    "Fonds",
    options=df_univ["name"].tolist(),
    default=[df_univ["name"].iloc[0]] if not df_univ.empty else [],
)

period_days = st.slider("Historique (jours ouvrés)", min_value=120, max_value=750, value=450, step=30)


# ===============================
# 07) ACTION — ANALYSE EODHD
# ===============================
if st.button("🔎 Analyser via EODHD") and choices:
    rows: List[Dict[str, Any]] = []
    charts: Dict[str, pd.DataFrame] = []
    debug_dump: Dict[str, Any] = {}

    for name in choices:
        row = df_univ.loc[df_univ["name"] == name].iloc[0].to_dict()
        isin = row.get("isin")

        ticker, dbg = find_ticker_best_effort(name, isin)
        debug_dump[name] = dbg

        prices = eodhd_prices_daily(ticker, days=period_days) if ticker else pd.DataFrame()
        perfs = perf_series(prices)
        fund  = eodhd_fundamentals(ticker) if ticker else {}

        row.update({
            "ticker": ticker,
            "Close": prices["Close"].iloc[-1] if not prices.empty else None,
            "Perf 1M %": perfs["1M"],
            "Perf 3M %": perfs["3M"],
            "Perf 6M %": perfs["6M"],
            "Perf YTD %": perfs["YTD"],
            "Perf 1Y %": perfs["1Y"],
            "facts": {
                "Currency": (fund.get("General", {}) or {}).get("Currency"),
                "AssetClass": (fund.get("ETF_Data", {}) or {}).get("AssetClass") or (fund.get("General", {}) or {}).get("Type"),
                "Exchange": (fund.get("General", {}) or {}).get("Exchange"),
            }
        })
        rows.append(row)

    # ===============================
    # 08) TABLEAU — FORMATAGE SÛR
    # ===============================
    st.subheader("Tableau récapitulatif")

    view_cols = [
        "name", "isin", "ticker", "type", "sri", "sfdr", "Close",
        "Perf 1M %", "Perf 3M %", "Perf 6M %", "Perf YTD %", "Perf 1Y %", "notes"
    ]
    view = pd.DataFrame(rows)[view_cols].copy()

    # Colonnes numériques -> coerce
    num_cols = ["Close", "Perf 1M %", "Perf 3M %", "Perf 6M %", "Perf YTD %", "Perf 1Y %"]
    for c in num_cols:
        view[c] = pd.to_numeric(view[c], errors="coerce")

    # Formateurs robustes
    def fmt_money(x):
        try:
            if x is None or (isinstance(x, float) and np.isnan(x)):
                return ""
            return f"{float(x):,.2f}"
        except Exception:
            return ""

    def fmt_pct(x):
        try:
            if x is None or (isinstance(x, float) and np.isnan(x)):
                return ""
            return f"{float(x):+.2f}%"
        except Exception:
            return ""

    styled = (
        view.rename(columns={
            "name": "Nom", "isin": "ISIN", "type": "Type", "sri": "SRI", "sfdr": "SFDR",
            "Close": "Dernier cours", "notes": "Notes"
        })
        .style.format({
            "Dernier cours": fmt_money,
            "Perf 1M %": fmt_pct,
            "Perf 3M %": fmt_pct,
            "Perf 6M %": fmt_pct,
            "Perf YTD %": fmt_pct,
            "Perf 1Y %": fmt_pct,
        }, na_rep="")
    )

    st.dataframe(styled, use_container_width=True, hide_index=True)

    # ===============================
    # 09) DEBUG (optionnel)
    # ===============================
    if debug_mode:
        st.subheader("🔍 Debug EODHD")
        st.caption("Bruts des endpoints utilisés pour le mapping. Utile si un ticker ne remonte pas.")
        for nm, dbg in debug_dump.items():
            with st.expander(f"Debug: {nm}", expanded=False):
                st.write(dbg)

else:
    st.info("Sélectionne au moins un fonds puis clique sur « Analyser via EODHD ».")


# ===============================
# 10) FOOTER
# ===============================
st.divider()
st.caption("⚠️ Mapping renforcé ISIN/Name → ticker via /search + /fundamentals. "
           "Si un fond remonte encore à vide, ouvre le debug pour voir les retours EODHD.")
