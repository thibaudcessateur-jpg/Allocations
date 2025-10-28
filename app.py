# =========================================
# 01) IMPORTS & SETUP
# =========================================
import os
from typing import Optional, List, Dict, Any, Tuple

import requests
import numpy as np
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

load_dotenv()
st.set_page_config(page_title="Allocation CGP — Analyse fonds (EODHD - ISIN only)",
                   page_icon="🦉", layout="wide")


# =========================================
# 02) SECRET TOKEN HELPER
# =========================================
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


# =========================================
# 03) EODHD — CLIENT (ISIN -> EXCHANGE -> /eod)
# =========================================
def eodhd_base_url() -> str:
    return os.getenv("EODHD_BASE_URL") or st.secrets.get("EODHD_BASE_URL", "https://eodhd.com/api")

def eodhd_headers() -> Dict[str, str]:
    return {"Accept": "application/json"}

def eodhd_params(extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    params = {"fmt": "json", "api_token": Secret_Token("EODHD_API_KEY")}
    if extra:
        params.update(extra)
    return params

@st.cache_data(ttl=6*3600, show_spinner=False)
def eodhd_get(path: str, params: Optional[Dict[str, Any]] = None) -> Any:
    url = f"{eodhd_base_url().rstrip('/')}{path}"
    r = requests.get(url, params=eodhd_params(params or {}), headers=eodhd_headers(), timeout=25)
    r.raise_for_status()
    return r.json()

@st.cache_data(ttl=24*3600, show_spinner=False)
def eodhd_search_isin(isin: str) -> List[Dict[str, Any]]:
    """
    Recherche /search/{ISIN}. Retourne toujours une liste.
    On s’attend à recevoir un Exchange (ex: EUFUND) pour les OPCVM.
    """
    try:
        data = eodhd_get(f"/search/{isin}", params={"limit": 5})
    except Exception:
        return []
    if isinstance(data, dict) and "data" in data:
        return data["data"] or []
    return data if isinstance(data, list) else []

@st.cache_data(ttl=24*3600, show_spinner=False)
def eodhd_symbol_candidates_from_isin(isin: str) -> List[str]:
    """
    Construit des candidats pour /eod à partir de l’ISIN :
      1) ISIN.EXCHANGE (si Exchange trouvé par /search)
      2) ISIN (fallback)
    Exemple: FR0011253624 -> ["FR0011253624.EUFUND", "FR0011253624"]
    """
    items = eodhd_search_isin(isin)
    cands: List[str] = []
    if items:
        it0 = items[0]
        exch = None
        for k in ("exchange", "Exchange"):
            if k in it0 and it0[k]:
                exch = str(it0[k]).upper()
                break
        if exch:
            cands.append(f"{isin}.{exch}")
    cands.append(isin)
    # dédup
    seen, out = set(), []
    for s in cands:
        if s not in seen:
            out.append(s); seen.add(s)
    return out

@st.cache_data(ttl=6*3600, show_spinner=False)
def eodhd_prices_daily_safe(candidates: List[str], days: int = 450) -> Tuple[pd.DataFrame, Optional[str], List[str], Optional[str]]:
    """
    Essaie /eod sur la liste de symboles, sans planter.
    Retourne (df, successful_symbol_or_none, tried_symbols, last_http_error_text_or_none)
    """
    tried: List[str] = []
    last_err: Optional[str] = None

    for sym in candidates:
        if not sym:
            continue
        tried.append(sym)
        try:
            js = eodhd_get(f"/eod/{sym}", params={"period": "d"})
            df = pd.DataFrame(js)
            if df.empty or "close" not in df.columns:
                continue
            dcol = "date" if "date" in df.columns else "Date"
            df[dcol] = pd.to_datetime(df[dcol])
            df = df.set_index(dcol).sort_index().tail(days)[["close"]].rename(columns={"close": "Close"})
            if not df.empty:
                return df, sym, tried, None
        except requests.HTTPError:
            last_err = f"HTTPError for {sym}"
            continue
        except Exception as e:
            last_err = f"{type(e).__name__} for {sym}"
            continue

    return pd.DataFrame(), None, tried, last_err

def perf_series(prices: pd.DataFrame) -> Dict[str, Optional[float]]:
    """
    Perf % (1M / YTD / 1Y / 3Y / 5Y / 8Y) à partir du dernier close.
    None si non calculable (pas assez d'historique).
    """
    out = {"1M": None, "YTD": None, "1Y": None, "3Y": None, "5Y": None, "8Y": None}
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
        out["YTD"] = pct(pd.Timestamp(year=end.year, month=1, day=1, tz=end.tz))
        out["1Y"]  = pct(end - pd.DateOffset(years=1))
        out["3Y"]  = pct(end - pd.DateOffset(years=3))
        out["5Y"]  = pct(end - pd.DateOffset(years=5))
        out["8Y"]  = pct(end - pd.DateOffset(years=8))
    except Exception:
        pass
    return out


# =========================================
# 04) UNIVERS — ESPACE INVEST 5
# =========================================
UNIVERSE: List[Dict[str, Any]] = [
    {"name": "R-co Valor C EUR", "isin": "FR0011253624", "sri": 4, "sfdr": 8, "type": "UC Actions Monde",
     "notes": ""},
    {"name": "Vivalor International", "isin": "FR0014001LS1", "sri": 4, "sfdr": None, "type": "UC Actions Monde",
     "notes": "Non éligible aux transferts programmés"},
    {"name": "COMGEST Monde C", "isin": "FR0000284689", "sri": 4, "sfdr": 8, "type": "UC Actions Monde",
     "notes": ""},
    {"name": "Echiquier World Equity Growth", "isin": "FR0010859769", "sri": 4, "sfdr": 8, "type": "UC Actions Monde",
     "notes": ""},
    {"name": "Franklin Mutual Global Discovery", "isin": "LU0211333298", "sri": 4, "sfdr": 8, "type": "UC Actions Monde",
     "notes": ""},
    {"name": "CARMIGNAC INVESTISSEMENT A EUR", "isin": "FR0010148981", "sri": 4, "sfdr": 8, "type": "UC Actions Monde",
     "notes": ""},
    {"name": "Natixis - Thematics Meta A EUR", "isin": "LU1951204046", "sri": 5, "sfdr": 8, "type": "UC Thématique Monde",
     "notes": ""},
    {"name": "Pictet Global Megatrend Selection P", "isin": "LU0386882277", "sri": 4, "sfdr": 8, "type": "UC Thématique Monde",
     "notes": ""},
    {"name": "Morgan Stanley Gl Brands A", "isin": "LU0119620416", "sri": 4, "sfdr": 8, "type": "UC Actions Monde",
     "notes": ""},
    {"name": "FIDELITY FUNDS - WORLD FUND", "isin": "LU0069449576", "sri": 4, "sfdr": 8, "type": "UC Actions Monde",
     "notes": ""},
    {"name": "CLARTAN VALEURS", "isin": "LU1100076550", "sri": 4, "sfdr": 8, "type": "UC Actions Monde",
     "notes": ""},
    {"name": "CARMIGNAC PATRIMOINE", "isin": "FR0010135103", "sri": 3, "sfdr": 8, "type": "UC Diversifié (patrimonial)",
     "notes": ""},
    {"name": "SYCOYIELD 2030 RC", "isin": "FR001400MCQ6", "sri": 2, "sfdr": 8, "type": "Obligataire daté 2030",
     "notes": ""},
    {"name": "R-Co Target 2029 HY", "isin": None, "sri": None, "sfdr": None, "type": "Obligataire daté 2029 HY",
     "notes": "ISIN à compléter"},
    {"name": "Fonds en euros AGGV", "isin": None, "sri": 1, "sfdr": None, "type": "Fonds en euros",
     "notes": ""},
]


# =========================================
# 05) UI — HEADER & DEBUG
# =========================================
st.title("🦉 Analyse fonds — Espace Invest 5 (EODHD, ISIN-only)")
st.caption("ISIN → Exchange (via /search) → /eod: ISIN.EXCHANGE, fallback ISIN. Perfs long terme activées.")

with st.sidebar:
    st.header("Clé API EODHD")
    try:
        _ = Secret_Token("EODHD_API_KEY")
        st.success("Clé EODHD détectée")
    except Exception:
        st.error("Clé EODHD manquante — ajoutez EODHD_API_KEY dans secrets/env.")
    debug_mode = st.toggle("Mode debug", value=False, help="Affiche les candidats et les symboles testés.")


# =========================================
# 06) SÉLECTION & PARAMÈTRES
# =========================================
st.subheader("Sélection des fonds à analyser")
df_univ = pd.DataFrame(UNIVERSE)

choices = st.multiselect(
    "Fonds",
    options=df_univ["name"].tolist(),
    default=[df_univ["name"].iloc[0]] if not df_univ.empty else [],
)

period_days = st.slider("Historique (jours ouvrés)", min_value=120, max_value=2500, value=1500, step=60)


# =========================================
# 07) ACTION — ANALYSE EODHD (ISIN -> /eod)
# =========================================
if st.button("🔎 Analyser via EODHD") and choices:
    rows: List[Dict[str, Any]] = []
    debug_dump: Dict[str, Any] = {}

    for name in choices:
        row = df_univ.loc[df_univ["name"] == name].iloc[0].to_dict()
        isin = row.get("isin")

        candidates = eodhd_symbol_candidates_from_isin(isin) if isin else []
        close_df, ok_symbol, tried, last_err = (pd.DataFrame(), None, [], None)
        if candidates:
            close_df, ok_symbol, tried, last_err = eodhd_prices_daily_safe(candidates, days=period_days)

        debug_dump[name] = {"candidates": candidates, "eod_tries": tried, "last_error": last_err}

        perfs = perf_series(close_df)
        row.update({
            "ticker": ok_symbol if ok_symbol else (candidates[0] if candidates else None),
            "Close": close_df["Close"].iloc[-1] if not close_df.empty else None,
            "Perf 1M %": perfs["1M"],
            "Perf YTD %": perfs["YTD"],
            "Perf 1Y %": perfs["1Y"],
            "Perf 3Y %": perfs["3Y"],
            "Perf 5Y %": perfs["5Y"],
            "Perf 8Y %": perfs["8Y"],
        })
        rows.append(row)

    # =========================================
    # 08) TABLEAU — FORMATAGE SÛR (EUROPE)
    # =========================================
    st.subheader("Tableau récapitulatif")

    view_cols = ["name","isin","ticker","type","sri","sfdr","Close",
                 "Perf 1M %","Perf YTD %","Perf 1Y %","Perf 3Y %","Perf 5Y %","Perf 8Y %","notes"]
    view = pd.DataFrame(rows)[view_cols].copy()

    # Colonnes numériques -> coerce
    num_cols = ["Close","Perf 1M %","Perf YTD %","Perf 1Y %","Perf 3Y %","Perf 5Y %","Perf 8Y %"]
    for c in num_cols:
        view[c] = pd.to_numeric(view[c], errors="coerce")

    # ---------- formatteurs EU ----------
    def to_eur(x: float) -> str:
        """
        4000.64 -> '4 000,64 €' (format européen)
        """
        try:
            if x is None or (isinstance(x, float) and np.isnan(x)):
                return ""
            s = f"{float(x):,.2f}"        # '4,000.64'
            s = s.replace(",", "X")       # '4X000.64'
            s = s.replace(".", ",")       # '4X000,64'
            s = s.replace("X", " ")       # '4 000,64'
            return f"{s} €"
        except Exception:
            return ""

    def pct_eu(x: float) -> str:
        """
        +12.34 -> '+12,34%' (format européen, signe conservé)
        """
        try:
            if x is None or (isinstance(x, float) and np.isnan(x)):
                return ""
            s = f"{float(x):+.2f}"        # '+12.34'
            s = s.replace(",", "X")
            s = s.replace(".", ",")       # '+12,34'
            s = s.replace("X", " ")
            return f"{s}%"
        except Exception:
            return ""

    styled = (
        view.rename(columns={
            "name":"Nom","isin":"ISIN","type":"Type","sri":"SRI","sfdr":"SFDR",
            "Close":"Dernier cours","notes":"Notes"
        })
        .style.format({
            "Dernier cours": to_eur,
            "Perf 1M %": pct_eu,
            "Perf YTD %": pct_eu,
            "Perf 1Y %": pct_eu,
            "Perf 3Y %": pct_eu,
            "Perf 5Y %": pct_eu,
            "Perf 8Y %": pct_eu,
        }, na_rep="")
    )

    st.dataframe(styled, use_container_width=True, hide_index=True)

    # =========================================
    # 09) DEBUG (facultatif)
    # =========================================
    if debug_mode:
        st.subheader("🔍 Debug requêtes")
        st.caption("Candidats dérivés de l’ISIN et symboles testés /eod.")
        for nm, dbg in debug_dump.items():
            with st.expander(f"Debug: {nm}", expanded=False):
                st.write(dbg)

else:
    st.info("Sélectionne au moins un fonds puis clique sur « Analyser via EODHD ».")


# =========================================
# 10) FOOTER
# =========================================
st.divider()
st.caption("Méthode ISIN → Exchange (via /search) → /eod ISIN.EXCHANGE (fallback ISIN). "
           "Perfs: 1M, YTD, 1Y, 3Y, 5Y, 8Y. Format monétaire européen.")
