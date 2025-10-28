# ===============================
# 01) IMPORTS & SETUP
# ===============================
import os
from typing import Optional, List, Dict, Any

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
# 03) EODHD — CLIENT LÉGER
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
def eodhd_search_isin(isin: str) -> List[Dict[str, Any]]:
    """Recherche EODHD pour retrouver le ticker à partir d'un ISIN."""
    try:
        data = eodhd_get(f"/search/{isin}", params={"limit": 20})
    except Exception:
        return []
    if isinstance(data, dict) and "data" in data:
        return data["data"]
    return data if isinstance(data, list) else []

def pick_best_ticker_from_search(items: List[Dict[str, Any]], isin: str) -> Optional[str]:
    """Heuristique de sélection du 'symbol' à partir des résultats de recherche."""
    if not items:
        return None
    for it in items:
        if str(it.get("isin", "")).upper() == isin.upper():
            return it.get("code") or it.get("symbol")
    eu_ex = ("PAR", "XETRA", "MIL", "AMS", "LSE", "VIE", "MAD", "BRU", "LIS", "VTX")
    eu = [it for it in items if str(it.get("exchange", "")).upper() in eu_ex]
    if eu:
        return eu[0].get("code") or eu[0].get("symbol")
    return items[0].get("code") or items[0].get("symbol")

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

@st.cache_data(ttl=6 * 3600, show_spinner=False)
def eodhd_fundamentals(ticker: str) -> Dict[str, Any]:
    """Fundamentals généraux (meilleure-effort — varie selon le type d'actif)."""
    try:
        js = eodhd_get(f"/fundamentals/{ticker}")
        return js if isinstance(js, dict) else {}
    except Exception:
        return {}

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
        out["1M"] = pct(end - pd.DateOffset(months=1))
        out["3M"] = pct(end - pd.DateOffset(months=3))
        out["6M"] = pct(end - pd.DateOffset(months=6))
        ytd_start = pd.Timestamp(year=end.year, month=1, day=1, tz=end.tz)
        out["YTD"] = pct(ytd_start)
        out["1Y"] = pct(end - pd.DateOffset(years=1))
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
# 05) TITRES & SIDEBAR (ÉTAT CLÉ)
# ===============================
st.title("🦉 Analyse UC — Espace Invest 5 (via EODHD)")
st.caption("Aucune allocation calculée. Analyse variations et infos techniques via EODHD.")

with st.sidebar:
    st.header("Clé API EODHD")
    try:
        _ = Secret_Token("EODHD_API_KEY")
        st.success("Clé EODHD détectée")
    except Exception:
        st.error("Clé EODHD manquante — ajoutez EODHD_API_KEY dans secrets/env.")


# ===============================
# 06) SÉLECTION & PARAMÈTRES
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
    charts: Dict[str, pd.DataFrame] = {}

    for name in choices:
        row = df_univ.loc[df_univ["name"] == name].iloc[0].to_dict()
        isin = row.get("isin")
        ticker = None

        if isin:
            items = eodhd_search_isin(isin)
            ticker = pick_best_ticker_from_search(items, isin)

        prices = eodhd_prices_daily(ticker, days=period_days) if ticker else pd.DataFrame()
        perfs = perf_series(prices)
        fund = eodhd_fundamentals(ticker) if ticker else {}

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
        charts[name] = prices


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
    # 09) MINI-CHARTS PRIX
    # ===============================
    st.subheader("Mini-charts (clôtures quotidiennes)")
    for name in choices:
        prices = charts.get(name)
        if prices is None or prices.empty:
            st.info(f"Pas de données prix pour: {name}")
            continue
        st.line_chart(prices, y="Close", height=220, use_container_width=True)

else:
    st.info("Sélectionne au moins un fonds puis clique sur « Analyser via EODHD ».")


# ===============================
# 10) FOOTER
# ===============================
st.divider()
st.caption("⚠️ Best-effort mapping ISIN → ticker via /search. Ajuster si besoin. Aucune recommandation, affichage informatif.")
