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
# 12) UNIVERSE + BENCHMARKS (EODHD)  ✅
#     - Associe chaque fonds à SON benchmark réel
#     - Mappe chaque benchmark vers un ou plusieurs ISINs (pondérés)
#       (tu peux ajuster les ISINs selon tes préférences produits)
# =========================================

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

# --- Benchmarks -> ISIN(s) ---
# NOTE: vérifie/ajuste les ISIN ci-dessous selon les ETF que TU veux utiliser comme proxy :
# - MSCI World Index  -> iShares Core MSCI World UCITS ETF (IWDA)        ISIN: IE00B4L5Y983
# - MSCI ACWI Index   -> iShares MSCI ACWI UCITS ETF (SSAC)              ISIN: IE00B6R52259
# - MSCI World Value  -> iShares Edge MSCI World Value Factor UCITS ETF  ISIN: IE00BP3QZB59
# - Euro Aggregate    -> iShares Core Euro Aggregate Bond UCITS ETF      ISIN: IE00B3DKXQ41
BENCHMARKS_ISIN = {
    "MSCI World Index": "IE00B4L5Y983",
    "MSCI ACWI Index": "IE00B6R52259",
    "MSCI World Value Index": "IE00BP3QZB59",
    "Euro Aggregate Bond": "IE00B3DKXQ41",
    "50% MSCI World / 50% Euro Aggregate Bond": [("IE00B4L5Y983", 0.5), ("IE00B3DKXQ41", 0.5)],
}

# =========================================
# 12) INDICATEURS QUANTITATIFS — EODHD (avec vrais benchmarks)  ✅
# =========================================
import numpy as np
import pandas as pd
import plotly.express as px

TRADING_DAYS = 252

def _get_close_series_by_name(name: str, tail_days: int = 2500) -> pd.Series:
    """Série prix (Close) d'un fonds sélectionné (via ISIN -> candidats -> eodhd_prices_safe)."""
    row = df_univ.loc[df_univ["name"] == name]
    if row.empty:
        return pd.Series(dtype=float)
    isin = row.iloc[0].get("isin")
    if not isin:
        return pd.Series(dtype=float)

    candidates = eodhd_symbol_candidates_from_isin(isin)
    df, ok_sym, _, _ = eodhd_prices_safe(candidates, period="d", tail_days=tail_days)
    if df.empty or "Close" not in df.columns:
        return pd.Series(dtype=float)
    s = df["Close"].copy()
    s.name = ok_sym or name
    return s

def _get_close_series_by_isin(isin: str, tail_days: int = 2500, name: str | None = None) -> pd.Series:
    """Série prix (Close) via un ISIN (utile pour les benchmarks)."""
    candidates = eodhd_symbol_candidates_from_isin(isin)
    df, ok_sym, _, _ = eodhd_prices_safe(candidates, period="d", tail_days=tail_days)
    if df.empty or "Close" not in df.columns:
        return pd.Series(dtype=float)
    s = df["Close"].copy()
    s.name = name or ok_sym or isin
    return s

def get_benchmark_prices(bench_label: str, tail_days: int = 2500) -> pd.Series:
    """
    Prix du benchmark:
      - Si bench_label est un libellé simple (clé de BENCHMARKS_ISIN -> ISIN str) : prend cet ETF proxy.
      - Si bench_label est un panier (liste [(ISIN, poids), ...]) : construit un indice composite base=1.
    Retour: Series 'Close' (proxy benchmark) nommée bench_label.
    """
    item = BENCHMARKS_ISIN.get(bench_label)
    if not item:
        # Permet de mapper des libellés "composites" personnalisés
        if bench_label == "50% MSCI World / 50% Euro Aggregate Bond":
            item = BENCHMARKS_ISIN["50% MSCI World / 50% Euro Aggregate Bond"]
        else:
            st.warning(f"Benchmark '{bench_label}' introuvable dans BENCHMARKS_ISIN.")
            return pd.Series(dtype=float)

    if isinstance(item, str):
        s = _get_close_series_by_isin(item, tail_days=tail_days, name=bench_label)
        return s

    # Panier pondéré
    parts = []
    for isin, w in item:
        si = _get_close_series_by_isin(isin, tail_days=tail_days, name=None)
        if not si.empty:
            # normalise à 1 au début pour pouvoir sommer pondéré
            si = si / si.iloc[0]
            parts.append((si, float(w)))
    if not parts:
        return pd.Series(dtype=float)
    idx = parts[0][0].index
    composite = sum(w * si.reindex(idx).fillna(method="ffill") for si, w in parts)
    composite.name = bench_label
    return composite

def _to_returns(price_df: pd.DataFrame) -> pd.DataFrame:
    return price_df.pct_change().dropna(how="all")

def _annualized_return(ret: pd.Series) -> float:
    if ret.empty: return np.nan
    mean_daily = ret.mean()
    return (1 + mean_daily) ** TRADING_DAYS - 1

def _annualized_vol(ret: pd.Series) -> float:
    if ret.empty: return np.nan
    return ret.std(ddof=0) * np.sqrt(TRADING_DAYS)

def _downside_dev(ret: pd.Series, rf_daily: float = 0.0) -> float:
    if ret.empty: return 0.0
    dd = ret[ret < rf_daily] - rf_daily
    if dd.empty: return 0.0
    return dd.std(ddof=0)

def _max_drawdown(price: pd.Series) -> float:
    if price.empty: return np.nan
    cummax = price.cummax()
    dd = price / cummax - 1.0
    return float(dd.min()) if not dd.empty else np.nan

def _beta_alpha_r2(ret_fund: pd.Series, ret_bench: pd.Series) -> tuple[float, float, float]:
    aligned = pd.concat([ret_fund, ret_bench], axis=1, join="inner").dropna()
    if aligned.shape[0] < 60:
        return np.nan, np.nan, np.nan
    rf = aligned.iloc[:, 0]
    rb = aligned.iloc[:, 1]
    var_b = rb.var(ddof=0)
    if var_b == 0 or np.isnan(var_b):
        return np.nan, np.nan, np.nan
    cov_fb = np.cov(rf, rb, ddof=0)[0, 1]
    beta = cov_fb / var_b
    alpha_daily = (rf - beta * rb).mean()
    r2 = float(np.corrcoef(rf, rb)[0, 1] ** 2)
    return float(beta), float(alpha_daily), r2

def _tracking_error(ret_fund: pd.Series, ret_bench: pd.Series) -> float:
    diff = (ret_fund - ret_bench).dropna()
    if diff.empty: return np.nan
    return diff.std(ddof=0) * np.sqrt(TRADING_DAYS)

def _information_ratio(ret_fund: pd.Series, ret_bench: pd.Series) -> float:
    ann_excess = _annualized_return(ret_fund) - _annualized_return(ret_bench)
    te = _tracking_error(ret_fund, ret_bench)
    return ann_excess / te if te and te > 0 else np.nan

st.header("📐 Indicateurs quantitatifs (EODHD) — vs benchmark officiel de chaque fonds")

if choices:
    # --- Paramètres ---
    colq1, colq2 = st.columns(2)
    with colq1:
        rf_annual_pct = st.number_input("Taux sans risque annualisé (%)", value=0.00, step=0.10, format="%.2f")
    with colq2:
        tail_days_q = st.slider("Fenêtre historique (jours ouvrés)", min_value=252, max_value=2500, value=1500, step=63)
    rf_daily = (1 + rf_annual_pct/100.0) ** (1 / TRADING_DAYS) - 1

    # --- Prix EOD fonds sélectionnés
    price_map: dict[str, pd.Series] = {}
    for nm in choices:
        s = _get_close_series_by_name(nm, tail_days=tail_days_q)
        if not s.empty:
            price_map[nm] = s
    if len(price_map) < 1:
        st.warning("⚠️ Aucune série de prix disponible.")
        st.stop()

    prices = pd.concat(price_map, axis=1).dropna(how="all")
    prices = prices.reindex(columns=[c for c in choices if c in prices.columns])
    rets = _to_returns(prices)

    # --- Corrélation (entre les fonds) + lecture rapide
    st.subheader("🔗 Corrélation (quotidienne)")
    corr = rets.corr()

    def _corr_summary(corr_df: pd.DataFrame) -> dict:
        pairs = []
        cols = corr_df.columns.tolist()
        for i in range(len(cols)):
            for j in range(i+1, len(cols)):
                pairs.append((cols[i], cols[j], float(corr_df.iloc[i, j])))
        if not pairs:
            return {"avg": np.nan, "min": None, "max": None}
        avg = float(np.mean([p[2] for p in pairs]))
        min_pair = min(pairs, key=lambda x: x[2])
        max_pair = max(pairs, key=lambda x: x[2])
        return {"avg": avg, "min": min_pair, "max": max_pair}

    def _label_corr(v: float) -> str:
        if np.isnan(v): return "n/a"
        if v < 0.0:   return "négative (diversifiante)"
        if v < 0.30:  return "faible (très diversifiante)"
        if v < 0.70:  return "modérée"
        return "forte (peu diversifiante)"

    cs = _corr_summary(corr)
    left, right = st.columns([1,1])
    with left:
        st.dataframe(corr.style.format("{:.2f}"), use_container_width=True, hide_index=True)
    with right:
        st.markdown("### 🧭 Lecture rapide")
        if cs["min"] and cs["max"]:
            fmin1, fmin2, vmin = cs["min"]
            fmax1, fmax2, vmax = cs["max"]
            st.markdown(
                f"- **Paire la moins corrélée** : **{fmin1} / {fmin2}** → {vmin:.2f} → {_label_corr(vmin)} ✅\n"
                f"- **Paire la plus corrélée** : **{fmax1} / {fmax2}** → {vmax:.2f} → {_label_corr(vmax)}\n"
                f"- **Corrélation moyenne du panier** : {cs['avg']:.2f} → {_label_corr(cs['avg'])}"
            )
            if vmin < 0.30:
                st.success("👉 Au moins une paire faiblement corrélée : **bonne diversification**.")
            if vmax > 0.70:
                st.warning("⚠️ Une paire très corrélée : **redondance** possible.")
        else:
            st.info("Sélectionne au moins 2 fonds pour analyser la corrélation.")

    # --- Tableau indicateurs (vs benchmark propre à CHAQUE fonds)
    st.subheader("📊 Indicateurs annualisés — vs benchmark du fonds")
    rows = []
    for nm in rets.columns:
        r = rets[nm].dropna()
        if r.empty:
            continue

        # Benchmark label & prix
        bench_label = df_univ.loc[df_univ["name"] == nm, "benchmark"].iloc[0]
        b_price = get_benchmark_prices(bench_label, tail_days=tail_days_q)
        if b_price.empty:
            # si le libellé composite n'est pas dans le dict, essaye de découper
            if "50%" in bench_label and "World" in bench_label and "Aggregate" in bench_label:
                b_price = get_benchmark_prices("50% MSCI World / 50% Euro Aggregate Bond", tail_days=tail_days_q)
        b_ret = _to_returns(pd.DataFrame({bench_label: b_price}))[bench_label] if not b_price.empty else pd.Series(dtype=float)

        ann_ret = _annualized_return(r)
        ann_vol = _annualized_vol(r)
        sharpe = (ann_ret - rf_annual_pct/100.0) / ann_vol if ann_vol and ann_vol > 0 else np.nan
        dd = _downside_dev(r, rf_daily=rf_daily)
        sortino = (ann_ret - rf_annual_pct/100.0) / (dd * np.sqrt(TRADING_DAYS)) if dd and dd > 0 else np.nan
        mdd = _max_drawdown(prices[nm].dropna())
        calmar = ann_ret / abs(mdd) if mdd and mdd < 0 else np.nan

        beta = alpha_d = r2 = te = ir = np.nan
        if not b_ret.empty:
            beta, alpha_d, r2 = _beta_alpha_r2(r, b_ret)
            te = _tracking_error(r, b_ret)
            ir = _information_ratio(r, b_ret)

        # badges pédagogiques
        def badge_sharpe(x):
            if np.isnan(x): return "—"
            if x >= 2:  return "🟢 excellent (≥2)"
            if x >= 1:  return "🟡 correct (≥1)"
            return "🔴 faible (<1)"
        def badge_beta(x):
            if np.isnan(x): return "—"
            if x < 0.7:    return "🟢 défensif (<0,7)"
            if x <= 1.2:   return "🟡 proche marché (~1)"
            return "🔴 agressif (>1,2)"
        def badge_r2(x):
            if np.isnan(x): return "—"
            if x >= 0.80:  return "🔴 très corrélé au bench (≥0,80)"
            if x >= 0.50:  return "🟡 corrélation modérée"
            return "🟢 faible corrélation (diversifiant)"
        def badge_mdd(x):
            if np.isnan(x): return "—"
            if x > -15/100:   return "🟢 drawdown contenu"
            if x > -30/100:   return "🟡 drawdown moyen"
            return "🔴 drawdown élevé"

        rows.append({
            "Fonds": nm,
            "Benchmark": bench_label,
            "Perf ann. %": ann_ret*100.0,
            "Vol ann. %": ann_vol*100.0,
            "Sharpe": sharpe,
            "Sortino": sortino,
            "Max DD %": mdd*100.0 if pd.notna(mdd) else np.nan,
            "Calmar": calmar,
            "Beta vs bench": beta,
            "R² vs bench": r2,
            "Tracking error %": te*100.0 if pd.notna(te) else np.nan,
            "Information ratio": ir,
            # lectures
            "Lecture Sharpe": badge_sharpe(sharpe),
            "Lecture Beta": badge_beta(beta),
            "Lecture R²": badge_r2(r2),
            "Lecture Drawdown": badge_mdd(mdd),
        })

    df_metrics = pd.DataFrame(rows)

    # format + couleurs
    fmt = {
        "Perf ann. %": "{:.2f}", "Vol ann. %": "{:.2f}", "Sharpe": "{:.2f}", "Sortino": "{:.2f}",
        "Max DD %": "{:.2f}", "Calmar": "{:.2f}", "Beta vs bench": "{:.2f}",
        "R² vs bench": "{:.2f}", "Tracking error %": "{:.2f}", "Information ratio": "{:.2f}",
    }
    def _hl(val, col):
        try: v = float(val)
        except: return ""
        if col == "Sharpe":
            return "background-color: #d1fae5" if v >= 2 else ("background-color: #fef9c3" if v >= 1 else "background-color: #fee2e2")
        if col == "Max DD %":
            return "background-color: #d1fae5" if v > -15 else ("background-color: #fef9c3" if v > -30 else "background-color: #fee2e2")
        if col == "Beta vs bench":
            return "background-color: #d1fae5" if v < 0.7 else ("background-color: #fef9c3" if v <= 1.2 else "background-color: #fee2e2")
        if col == "R² vs bench":
            return "background-color: #d1fae5" if v < 0.5 else ("background-color: #fef9c3" if v < 0.8 else "background-color: #fee2e2")
        return ""

    st.dataframe(
        df_metrics.style.format(fmt, na_rep="")
                  .apply(lambda s: [_hl(v, s.name) for v in s], axis=0,
                         subset=["Sharpe","Max DD %","Beta vs bench","R² vs bench"]),
        use_container_width=True, hide_index=True
    )

    # Rolling corr vs benchmark propre à chaque fonds (option)
    with st.expander("📈 Rolling corr (126 jours) vs benchmark du fonds"):
        win = 126
        plots = []
        for nm in rets.columns:
            bench_label = df_univ.loc[df_univ["name"] == nm, "benchmark"].iloc[0]
            b_price = get_benchmark_prices(bench_label, tail_days=tail_days_q)
            if b_price.empty or nm not in rets.columns:
                continue
            b_ret = _to_returns(pd.DataFrame({bench_label: b_price}))[bench_label]
            r = rets[nm].dropna()
            rc = r.rolling(win).corr(b_ret)
            if rc.dropna().empty:
                continue
            plots.append(pd.DataFrame({f"{nm} vs {bench_label}": rc}))
        if plots:
            fig_rc = px.line(pd.concat(plots, axis=1), title=f"Rolling corr {win} jours vs benchmarks")
            st.plotly_chart(fig_rc, use_container_width=True)
        else:
            st.info("Historique insuffisant pour la rolling corr.")
else:
    st.info("Sélectionne des fonds/ETF puis lance l’analyse pour afficher les indicateurs.")

# --- Glossaire express (inchangé)
with st.expander("ℹ️ Glossaire (ultra-court)"):
    st.markdown(
        "- **Sharpe** (>1 bien, >2 excellent) : rendement par unité de risque total.\n"
        "- **Sortino** : comme Sharpe mais ne pénalise que les baisses.\n"
        "- **Max Drawdown** : pire baisse historique (plus c'est proche de 0 %, mieux c'est).\n"
        "- **Beta** (~1 = comme le bench ; <0,7 défensif ; >1,2 agressif).\n"
        "- **R²** (proximité au bench) : <0,5 = diversifiant ; >0,8 = suit fortement le bench.\n"
        "- **Tracking Error** : écart-type au bench (bas = suit de près).\n"
        "- **Information Ratio** : surperformance ajustée du TE (>0,5 bien, >0,75 très bien)."
    )
