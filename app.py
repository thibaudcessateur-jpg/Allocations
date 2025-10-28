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
# 03) EODHD — CLIENT LÉGER (+ mapping & fallbacks)
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
def eodhd_search(query: str, limit: int = 25) -> List[Dict[str, Any]]:
    """
    Recherche générique EODHD: /search/{query}
    Normalise en liste.
    """
    try:
        data = eodhd_get(f"/search/{query}", params={"limit": limit})
    except Exception:
        return []
    if isinstance(data, dict) and "data" in data:
        return data["data"] or []
    return data if isinstance(data, list) else []

def _pick(d: Dict[str, Any], *keys: str) -> Optional[str]:
    for k in keys:
        if k in d and d[k]:
            return str(d[k])
        K = k[0].upper() + k[1:]
        if K in d and d[K]:
            return str(d[K])
    return None

def pick_best_item(items: List[Dict[str, Any]], isin: Optional[str]) -> Optional[Dict[str, Any]]:
    """Retourne l'item le plus pertinent (match ISIN > bourse EU > premier)."""
    if not items:
        return None
    if isin:
        for it in items:
            if str(_pick(it, "isin") or "").upper() == isin.upper():
                return it
    eu_ex = {"PAR", "XETRA", "MIL", "AMS", "LSE", "VIE", "MAD", "BRU", "LIS", "VTX"}
    eu = [it for it in items if str(_pick(it, "exchange") or "").upper() in eu_ex]
    return eu[0] if eu else items[0]

def candidate_symbols_from_item(it: Dict[str, Any]) -> List[str]:
    """
    Fabrique des candidats possibles pour /eod:
      - code, symbol, ticker, Code, Symbol, Ticker
      - + suffixe '.EXCHANGE' quand exchange est présent
    """
    cands = []
    code = _pick(it, "code", "symbol", "ticker", "Code", "Symbol", "Ticker")
    exch = _pick(it, "exchange", "Exchange")
    if code:
        cands.append(code)
        if exch and "." not in code:
            cands.append(f"{code}.{exch}")
    # parfois 'name' est directement utilisable
    name = _pick(it, "name", "Name")
    if name:
        cands.append(name)
        if exch and "." not in name:
            cands.append(f"{name}.{exch}")
    # dédupliquer en conservant l'ordre
    seen, out = set(), []
    for s in cands:
        if s and s not in seen:
            out.append(s)
            seen.add(s)
    return out

def find_symbol_best_effort(name: str, isin: Optional[str]) -> Tuple[Optional[str], Dict[str, Any]]:
    """
    Stratégie robuste pour trouver un symbole exploitable par /eod:
      A) /search/{ISIN} → item → candidates
      B) /search/{name} → item → candidates
      C) /fundamentals/{ISIN} → General.Code/Ticker/Symbol (au pire)
    Renvoie (best_candidate_or_none, debug)
    """
    debug: Dict[str, Any] = {"search_isin": None, "search_name": None, "fundamentals_isin": None, "candidates": []}

    # A) par ISIN
    if isin:
        items = eodhd_search(isin, limit=25)
        debug["search_isin"] = items
        it = pick_best_item(items, isin)
        if it:
            debug["candidates"].extend(candidate_symbols_from_item(it))
            if debug["candidates"]:
                return debug["candidates"][0], debug

    # B) par nom
    items2 = eodhd_search(name, limit=25)
    debug["search_name"] = items2
    it2 = pick_best_item(items2, isin)
    if it2:
        debug["candidates"].extend(candidate_symbols_from_item(it2))
        if debug["candidates"]:
            return debug["candidates"][0], debug

    # C) fallback fundamentals sur ISIN (parfois renvoie un code exploitable ailleurs que /eod)
    if isin:
        f = eodhd_get(f"/fundamentals/{isin}") if isin else {}
        debug["fundamentals_isin"] = f
        gen = f.get("General") or {}
        tick = _pick(gen, "Code", "Ticker", "Symbol")
        if tick:
            debug["candidates"].append(tick)
            return tick, debug

    return None, debug

@st.cache_data(ttl=6 * 3600, show_spinner=False)
def eodhd_prices_daily_safe(symbol: str, days: int = 450) -> Tuple[pd.DataFrame, Optional[str]]:
    """
    Essaie plusieurs variantes pour /eod ; ne plante jamais.
    Retourne (df, successful_symbol_or_none).
    """
    # On génère quelques variantes utiles
    tries = [symbol]
    if "." not in symbol:
        # Quelques suffixes courants — on tente en best-effort
        tries += [f"{symbol}.PAR", f"{symbol}.LSE", f"{symbol}.AMS", f"{symbol}.XETRA", f"{symbol}.MIL"]

    for sym in tries:
        try:
            js = eodhd_get(f"/eod/{sym}", params={"period": "d"})
            df = pd.DataFrame(js)
            if df.empty or "close" not in df.columns:
                continue
            dcol = "date" if "date" in df.columns else "Date"
            df[dcol] = pd.to_datetime(df[dcol])
            df = df.set_index(dcol).sort_index()
            df = df.tail(days)[["close"]].rename(columns={"close": "Close"})
            if not df.empty:
                return df, sym
        except requests.HTTPError:
            continue
        except Exception:
            continue
    # échec total
    return pd.DataFrame(), None

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
# 05) UI — HEADER & DEBUG
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
    debug_mode = st.toggle("Mode debug EODHD", value=False, help="Affiche les retours et les symboles testés.")


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
    debug_dump: Dict[str, Any] = {}

    for name in choices:
        row = df_univ.loc[df_univ["name"] == name].iloc[0].to_dict()
        isin = row.get("isin")

        # 1) Trouver un symbole plausible
        symbol, dbg = find_symbol_best_effort(name, isin)
        debug_dump[name] = {"search_debug": dbg, "eod_tries": []}

        # 2) Tenter les /eod sur plusieurs variantes en sécurité
        close_df, ok_symbol = (pd.DataFrame(), None)
        if symbol:
            close_df, ok_symbol = eodhd_prices_daily_safe(symbol, days=period_days)
            debug_dump[name]["eod_tries"] = [symbol] + (
                [f"{symbol}.PAR", f"{symbol}.LSE", f"{symbol}.AMS", f"{symbol}.XETRA", f"{symbol}.MIL"]
                if "." not in symbol else []
            )

        perfs = perf_series(close_df)

        row.update({
            "ticker": ok_symbol or symbol,
            "Close": close_df["Close"].iloc[-1] if not close_df.empty else None,
            "Perf 1M %": perfs["1M"],
            "Perf 3M %": perfs["3M"],
            "Perf 6M %": perfs["6M"],
            "Perf YTD %": perfs["YTD"],
            "Perf 1Y %": perfs["1Y"],
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
        st.caption("Résultats de recherche et symboles tentés pour /eod.")
        for nm, dbg in debug_dump.items():
            with st.expander(f"Debug: {nm}", expanded=False):
                st.write(dbg)

else:
    st.info("Sélectionne au moins un fonds puis clique sur « Analyser via EODHD ».")


# ===============================
# 10) FOOTER
# ===============================
st.divider()
st.caption("⚠️ On teste plusieurs variantes de symbole (CODE, SYMBOL, CODE.EXCHANGE...). "
           "Si un fond n'a pas de série /eod sur EODHD, la ligne reste vide sans casser l'app.")
