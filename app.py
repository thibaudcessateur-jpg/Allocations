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

# ---------- /eod helpers ----------
def _json_to_close_df(js: Any) -> pd.DataFrame:
    df = pd.DataFrame(js)
    if df.empty or "close" not in df.columns:
        return pd.DataFrame()
    dcol = "date" if "date" in df.columns else "Date"
    df[dcol] = pd.to_datetime(df[dcol])
    df = df.set_index(dcol).sort_index()[["close"]].rename(columns={"close": "Close"})
    return df

@st.cache_data(ttl=6*3600, show_spinner=False)
def eodhd_prices_safe(candidates: List[str], period: str = "d", tail_days: Optional[int] = 1500) -> Tuple[pd.DataFrame, Optional[str], List[str], Optional[str]]:
    """
    Essaie /eod sur la liste de symboles pour un 'period' donné ('d', 'w', 'm'), sans planter.
    - Si tail_days est fourni, on ne garde que la queue (utile pour daily).
    Retourne (df, successful_symbol_or_none, tried_symbols, last_http_error_text_or_none)
    """
    tried: List[str] = []
    last_err: Optional[str] = None
    params = {"period": period}

    for sym in candidates:
        if not sym:
            continue
        tried.append(sym)
        try:
            js = eodhd_get(f"/eod/{sym}", params=params)
            df = _json_to_close_df(js)
            if df.empty:
                continue
            if tail_days and period == "d":
                df = df.tail(tail_days)
            if not df.empty:
                return df, sym, tried, None
        except requests.HTTPError:
            last_err = f"HTTPError for {sym}"
            continue
        except Exception as e:
            last_err = f"{type(e).__name__} for {sym}"
            continue

    return pd.DataFrame(), None, tried, last_err

def _perf_from_series(series: pd.DataFrame, target_dt: pd.Timestamp) -> Optional[float]:
    """Perf simple (%) entre le dernier point et le point au plus proche <= target_dt."""
    if series.empty:
        return None
    s = series.loc[:target_dt]
    if s.empty:
        return None
    last = series["Close"].iloc[-1]
    base = s["Close"].iloc[-1]
    if last is None or base is None or base <= 0:
        return None
    return (float(last) / float(base) - 1.0) * 100.0

def perf_series_with_monthly_fallback(prices_d: pd.DataFrame, prices_m: pd.DataFrame) -> Dict[str, Optional[float]]:
    """
    Calcule : 1M / YTD / 1Y / 3Y / 5Y / 8Y / 10Y
    - Utilise d'abord le daily (plus précis). Si la date de référence n'existe pas,
      bascule sur le mensuel.
    """
    out = {"1M": None, "YTD": None, "1Y": None, "3Y": None, "5Y": None, "8Y": None, "10Y": None}

    # Si aucune donnée, on sort
    if prices_d.empty and prices_m.empty:
        return out

    # borne temporelle (on se base sur le plus long des deux)
    idx_ref = (prices_d.index if not prices_d.empty else prices_m.index)
    end = idx_ref[-1]

    def pick_series(dt: pd.Timestamp) -> Tuple[pd.DataFrame, pd.Timestamp]:
        # Essaye daily, sinon mensuel
        if not prices_d.empty and prices_d.index[0] <= dt:
            return prices_d, dt
        return prices_m, dt

    try:
        # 1M
        dt = end - pd.DateOffset(months=1)
        s, dt2 = pick_series(dt)
        out["1M"] = _perf_from_series(s, dt2)
        # YTD
        ytd = pd.Timestamp(year=end.year, month=1, day=1, tz=end.tz)
        s, dt2 = pick_series(ytd)
        out["YTD"] = _perf_from_series(s, dt2)
        # 1Y
        dt = end - pd.DateOffset(years=1)
        s, dt2 = pick_series(dt)
        out["1Y"] = _perf_from_series(s, dt2)
        # 3Y
        dt = end - pd.DateOffset(years=3)
        s, dt2 = pick_series(dt)
        out["3Y"] = _perf_from_series(s, dt2)
        # 5Y
        dt = end - pd.DateOffset(years=5)
        s, dt2 = pick_series(dt)
        out["5Y"] = _perf_from_series(s, dt2)
        # 8Y
        dt = end - pd.DateOffset(years=8)
        s, dt2 = pick_series(dt)
        out["8Y"] = _perf_from_series(s, dt2)
        # 10Y
        dt = end - pd.DateOffset(years=10)
        s, dt2 = pick_series(dt)
        out["10Y"] = _perf_from_series(s, dt2)
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
st.caption("ISIN → Exchange via /search → /eod en daily + fallback mensuel pour perfs longues (8/10 ans).")

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

# Daily: jusqu'à ~16 ans ≈ 4000 séances; mensuel: récupéré en entier.
period_days = st.slider("Historique daily (jours ouvrés)", min_value=120, max_value=4000, value=2000, step=60)


# =========================================
# 07) ACTION — ANALYSE EODHD (daily + monthly fallback)
# =========================================
if st.button("🔎 Analyser via EODHD") and choices:
    rows: List[Dict[str, Any]] = []
    debug_dump: Dict[str, Any] = {}

    for name in choices:
        row = df_univ.loc[df_univ["name"] == name].iloc[0].to_dict()
        isin = row.get("isin")

        candidates = eodhd_symbol_candidates_from_isin(isin) if isin else []
        tried_d, tried_m = [], []
        last_err_d, last_err_m = None, None

        # 1) Daily (pour 1M/YTD/1Y/3Y/5Y si possible)
        daily_df, ok_symbol, tried_d, last_err_d = (pd.DataFrame(), None, [], None)
        if candidates:
            daily_df, ok_symbol, tried_d, last_err_d = eodhd_prices_safe(candidates, period="d", tail_days=period_days)

        # 2) Monthly (pour étendre l'histo et calculer 8Y/10Y si daily trop court)
        monthly_df, ok_symbol_m, tried_m, last_err_m = (pd.DataFrame(), None, [], None)
        if candidates:
            monthly_df, ok_symbol_m, tried_m, last_err_m = eodhd_prices_safe(candidates, period="m", tail_days=None)

        # symbole retenu (daily prioritaire, sinon mensuel)
        final_symbol = ok_symbol or ok_symbol_m

        # Perfs avec fallback mensuel si besoin
        perfs = perf_series_with_monthly_fallback(daily_df, monthly_df)

        row.update({
            "ticker": final_symbol if final_symbol else (candidates[0] if candidates else None),
            "Close": daily_df["Close"].iloc[-1] if not daily_df.empty else (
                monthly_df["Close"].iloc[-1] if not monthly_df.empty else None
            ),
            "Perf 1M %": perfs["1M"],
            "Perf YTD %": perfs["YTD"],
            "Perf 1Y %": perfs["1Y"],
            "Perf 3Y %": perfs["3Y"],
            "Perf 5Y %": perfs["5Y"],
            "Perf 8Y %": perfs["8Y"],
            "Perf 10Y %": perfs["10Y"],
        })
        rows.append(row)

        if debug_mode:
            debug_dump[name] = {
                "candidates": candidates,
                "daily": {"tried": tried_d, "last_error": last_err_d, "rows": int(daily_df.shape[0])},
                "monthly": {"tried": tried_m, "last_error": last_err_m, "rows": int(monthly_df.shape[0])},
            }

    # =========================================
    # 08) TABLEAU — FORMATAGE SÛR (EUROPE)
    # =========================================
    st.subheader("Tableau récapitulatif")

    view_cols = ["name","isin","ticker","type","sri","sfdr","Close",
                 "Perf 1M %","Perf YTD %","Perf 1Y %","Perf 3Y %","Perf 5Y %","Perf 8Y %","Perf 10Y %","notes"]
    view = pd.DataFrame(rows)[view_cols].copy()

    # Colonnes numériques -> coerce
    num_cols = ["Close","Perf 1M %","Perf YTD %","Perf 1Y %","Perf 3Y %","Perf 5Y %","Perf 8Y %","Perf 10Y %"]
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
            "name":"Nom","isin":"ISIN","type":"Type","sri":"SRI","sfdr":"SFFR",
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
            "Perf 10Y %": pct_eu,
        }, na_rep="")
    )

    st.dataframe(styled, use_container_width=True, hide_index=True)

    # =========================================
    # 09) DEBUG (facultatif)
    # =========================================
    if debug_mode:
        st.subheader("🔍 Debug requêtes")
        st.caption("Daily + Monthly. Candidats et symboles testés.")
        for nm, dbg in debug_dump.items():
            with st.expander(f"Debug: {nm}", expanded=False):
                st.write(dbg)

else:
    st.info("Sélectionne au moins un fonds puis clique sur « Analyser via EODHD ».")


# =========================================
# 10) FOOTER
# =========================================
st.divider()
st.caption("Méthode ISIN → Exchange (via /search) → /eod daily + fallback mensuel. "
           "Perfs: 1M, YTD, 1Y, 3Y, 5Y, 8Y, 10Y. Format monétaire européen.")

# =========================================
# 12) INDICATEURS QUANTITATIFS — EODHD (corrigé et robuste)
# =========================================
import numpy as np
import pandas as pd
import plotly.express as px

TRADING_DAYS = 252

# --- Univers des fonds et leurs benchmarks officiels
UNIVERSE = [
    {"name": "R-co Valor C EUR",                    "isin": "FR0011253624", "type": "Fonds Actions Monde",     "benchmark": "MSCI World Index"},
    {"name": "Vivalor International",               "isin": "FR0014001LS1", "type": "Fonds Actions Monde",     "benchmark": "MSCI World Index"},
    {"name": "COMGEST Monde C",                     "isin": "FR0000284689", "type": "Fonds Actions Monde",     "benchmark": "MSCI World Index"},
    {"name": "Echiquier World Equity Growth",       "isin": "FR0010859769", "type": "Fonds Actions Monde",     "benchmark": "MSCI ACWI Index"},
    {"name": "Franklin Mutual Global Discovery",    "isin": "LU0211333298", "type": "Fonds Actions Monde",     "benchmark": "MSCI World Value Index"},
    {"name": "CARMIGNAC INVESTISSEMENT A EUR",      "isin": "FR0010148981", "type": "Fonds Actions Monde",     "benchmark": "MSCI ACWI Index"},
    {"name": "Pictet Global Megatrend Selection P", "isin": "LU0386882277", "type": "Fonds Thématique Monde",  "benchmark": "MSCI World Index"},
    {"name": "CARMIGNAC PATRIMOINE",                "isin": "FR0010135103", "type": "Diversifié Patrimonial",  "benchmark": "50% MSCI World / 50% Euro Aggregate Bond"},
]

df_univ = pd.DataFrame(UNIVERSE)

# --- Protection (si colonne manquante ou df_univ ancien)
if "benchmark" not in df_univ.columns:
    FALLBACK_BENCH = {
        "R-co Valor C EUR": "MSCI World Index",
        "Vivalor International": "MSCI World Index",
        "COMGEST Monde C": "MSCI World Index",
        "Echiquier World Equity Growth": "MSCI ACWI Index",
        "Franklin Mutual Global Discovery": "MSCI World Value Index",
        "CARMIGNAC INVESTISSEMENT A EUR": "MSCI ACWI Index",
        "Pictet Global Megatrend Selection P": "MSCI World Index",
        "CARMIGNAC PATRIMOINE": "50% MSCI World / 50% Euro Aggregate Bond",
    }
    df_univ["benchmark"] = df_univ["name"].map(FALLBACK_BENCH).fillna("MSCI World Index")

# --- Benchmarks -> proxies ETF (ISIN)
BENCHMARKS_ISIN = {
    "MSCI World Index": "IE00B4L5Y983",            # iShares Core MSCI World UCITS ETF (IWDA)
    "MSCI ACWI Index": "IE00B6R52259",             # iShares MSCI ACWI UCITS ETF (SSAC)
    "MSCI World Value Index": "IE00BP3QZB59",      # iShares Edge MSCI World Value Factor ETF
    "Euro Aggregate Bond": "IE00B3DKXQ41",         # iShares Core Euro Aggregate Bond ETF
    "50% MSCI World / 50% Euro Aggregate Bond": [
        ("IE00B4L5Y983", 0.5),
        ("IE00B3DKXQ41", 0.5),
    ],
}

# --- Fonctions utilitaires (prix, retours, métriques)
def _get_close_series_by_isin(isin: str, tail_days: int = 2500, name: str | None = None) -> pd.Series:
    candidates = eodhd_symbol_candidates_from_isin(isin)
    df, ok_sym, _, _ = eodhd_prices_safe(candidates, period="d", tail_days=tail_days)
    if df.empty or "Close" not in df.columns:
        return pd.Series(dtype=float)
    s = df["Close"].copy()
    s.name = name or ok_sym or isin
    return s

def get_benchmark_prices(bench_label: str, tail_days: int = 2500) -> pd.Series:
    """Récupère le prix du benchmark (ETF proxy ou mix 50/50)."""
    item = BENCHMARKS_ISIN.get(bench_label)
    if not item:
        return pd.Series(dtype=float)

    if isinstance(item, str):
        return _get_close_series_by_isin(item, tail_days=tail_days, name=bench_label)

    # panier composite
    parts = []
    for isin, w in item:
        s = _get_close_series_by_isin(isin, tail_days=tail_days, name=None)
        if not s.empty:
            s = s / s.iloc[0]
            parts.append((s, float(w)))
    if not parts:
        return pd.Series(dtype=float)
    idx = parts[0][0].index
    composite = sum(w * s.reindex(idx).fillna(method="ffill") for s, w in parts)
    composite.name = bench_label
    return composite

def _get_benchmark_label_safe(fund_name: str) -> str:
    row = df_univ.loc[df_univ["name"] == fund_name]
    if not row.empty and "benchmark" in row.columns and pd.notna(row.iloc[0]["benchmark"]):
        return str(row.iloc[0]["benchmark"])
    return "MSCI World Index"

def _to_returns(price_df: pd.DataFrame) -> pd.DataFrame:
    return price_df.pct_change().dropna(how="all")

def _annualized_return(ret: pd.Series) -> float:
    if ret.empty: return np.nan
    return (1 + ret.mean()) ** TRADING_DAYS - 1

def _annualized_vol(ret: pd.Series) -> float:
    if ret.empty: return np.nan
    return ret.std(ddof=0) * np.sqrt(TRADING_DAYS)

def _downside_dev(ret: pd.Series, rf_daily: float = 0.0) -> float:
    dd = ret[ret < rf_daily] - rf_daily
    return dd.std(ddof=0) if not dd.empty else 0.0

def _max_drawdown(price: pd.Series) -> float:
    if price.empty: return np.nan
    dd = price / price.cummax() - 1.0
    return float(dd.min()) if not dd.empty else np.nan

def _beta_alpha_r2(ret_fund: pd.Series, ret_bench: pd.Series) -> tuple[float, float, float]:
    aligned = pd.concat([ret_fund, ret_bench], axis=1).dropna()
    if aligned.shape[0] < 60:
        return np.nan, np.nan, np.nan
    rf, rb = aligned.iloc[:, 0], aligned.iloc[:, 1]
    var_b = rb.var(ddof=0)
    if var_b == 0 or np.isnan(var_b):
        return np.nan, np.nan, np.nan
    cov_fb = np.cov(rf, rb, ddof=0)[0, 1]
    beta = cov_fb / var_b
    alpha_daily = (rf - beta * rb).mean()
    r2 = np.corrcoef(rf, rb)[0, 1] ** 2
    return beta, alpha_daily, r2

def _tracking_error(rf: pd.Series, rb: pd.Series) -> float:
    diff = (rf - rb).dropna()
    return diff.std(ddof=0) * np.sqrt(TRADING_DAYS) if not diff.empty else np.nan

def _information_ratio(rf: pd.Series, rb: pd.Series) -> float:
    ann_excess = _annualized_return(rf) - _annualized_return(rb)
    te = _tracking_error(rf, rb)
    return ann_excess / te if te and te > 0 else np.nan

# --- Section Streamlit
st.header("📊 Indicateurs quantitatifs — vs benchmark officiel")

if choices:
    col1, col2 = st.columns(2)
    with col1:
        rf_annual_pct = st.number_input("Taux sans risque (%)", value=0.00, step=0.10)
    with col2:
        tail_days_q = st.slider("Fenêtre historique (jours ouvrés)", 252, 2500, 1500, step=63)
    rf_daily = (1 + rf_annual_pct/100.0) ** (1 / TRADING_DAYS) - 1

    # Récupération des prix des fonds sélectionnés
    price_map = {}
    for nm in choices:
        isin = df_univ.loc[df_univ["name"] == nm, "isin"].iloc[0]
        candidates = eodhd_symbol_candidates_from_isin(isin)
        df, ok_sym, _, _ = eodhd_prices_safe(candidates, period="d", tail_days=tail_days_q)
        if not df.empty and "Close" in df.columns:
            price_map[nm] = df["Close"]
    if not price_map:
        st.warning("Aucune donnée de prix disponible.")
        st.stop()

    prices = pd.concat(price_map, axis=1).dropna(how="all")
    rets = _to_returns(prices)

    # Corrélation simple
    st.subheader("🔗 Corrélation entre fonds")
    corr = rets.corr()
    st.dataframe(corr.style.format("{:.2f}"), use_container_width=True)

    # --- Calcul des indicateurs par fonds
    rows = []
    for nm in rets.columns:
        r = rets[nm].dropna()
        if r.empty: 
            continue

        bench_label = _get_benchmark_label_safe(nm)
        b_price = get_benchmark_prices(bench_label, tail_days=tail_days_q)
        b_ret = _to_returns(pd.DataFrame({bench_label: b_price}))[bench_label] if not b_price.empty else pd.Series(dtype=float)

        ann_ret = _annualized_return(r)
        ann_vol = _annualized_vol(r)
        sharpe = (ann_ret - rf_annual_pct/100.0) / ann_vol if ann_vol else np.nan
        sortino = (ann_ret - rf_annual_pct/100.0) / (_downside_dev(r, rf_daily)*np.sqrt(TRADING_DAYS)) if _downside_dev(r, rf_daily) else np.nan
        mdd = _max_drawdown(prices[nm])
        calmar = ann_ret / abs(mdd) if mdd < 0 else np.nan

        beta = r2 = te = ir = np.nan
        if not b_ret.empty:
            beta, _, r2 = _beta_alpha_r2(r, b_ret)
            te = _tracking_error(r, b_ret)
            ir = _information_ratio(r, b_ret)

        rows.append({
            "Fonds": nm,
            "Benchmark": bench_label,
            "Perf ann. %": ann_ret*100,
            "Vol ann. %": ann_vol*100,
            "Sharpe": sharpe,
            "Sortino": sortino,
            "Max DD %": mdd*100,
            "Calmar": calmar,
            "Beta": beta,
            "R²": r2,
            "Tracking Error %": te*100,
            "Info Ratio": ir,
        })

    dfm = pd.DataFrame(rows)
    st.dataframe(dfm.style.format("{:.2f}"), use_container_width=True)

    with st.expander("ℹ️ Glossaire rapide"):
        st.markdown(
            "- **Sharpe** (>1 bien, >2 excellent) : rendement par unité de risque total.\n"
            "- **Sortino** : comme Sharpe mais ne pénalise que les baisses.\n"
            "- **Max Drawdown** : pire baisse historique.\n"
            "- **Beta** : sensibilité au marché (1 = marché, <0.7 défensif, >1.2 agressif).\n"
            "- **R²** : corrélation avec le benchmark.\n"
            "- **Tracking Error** : écart-type des écarts au benchmark.\n"
            "- **Information Ratio** : surperformance ajustée du TE."
        )
else:
    st.info("Sélectionne au moins un fonds pour afficher les indicateurs.")


