from __future__ import annotations

import json
import sys
import textwrap
import importlib.util
import itertools
from datetime import date
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple

if importlib.util.find_spec("matplotlib") is not None:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
    MATPLOTLIB_ERROR = ""
else:
    plt = None
    MATPLOTLIB_AVAILABLE = False
    MATPLOTLIB_ERROR = "matplotlib non installé"
import numpy as np
import pandas as pd
import requests
import streamlit as st
import altair as alt
if importlib.util.find_spec("pypfopt") is not None:
    from pypfopt import EfficientFrontier, risk_models, expected_returns
    PYPFOPT_AVAILABLE = True
    PYPFOPT_ERROR = ""
else:
    EfficientFrontier = None
    risk_models = None
    expected_returns = None
    PYPFOPT_AVAILABLE = False
    PYPFOPT_ERROR = "pyportfolioopt non installé"
if importlib.util.find_spec("reportlab") is not None:
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib import colors
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.pdfgen import canvas
    REPORTLAB_AVAILABLE = True
    REPORTLAB_ERROR = ""
else:
    SimpleDocTemplate = Paragraph = Spacer = Image = Table = TableStyle = PageBreak = None
    A4 = None
    getSampleStyleSheet = None
    colors = None
    ParagraphStyle = None
    canvas = None
    REPORTLAB_AVAILABLE = False
    REPORTLAB_ERROR = "reportlab non installé"

# ------------------------------------------------------------
# Constantes & univers de fonds recommandés
# ------------------------------------------------------------
TODAY = pd.Timestamp.today().normalize()
APP_TITLE = "Comparateur de portefeuilles"
ANNUAL_FEE_EURO_PCT = 0.9
ANNUAL_FEE_UC_PCT = 1.2

RECO_FUNDS_CORE = [
    ("R-co Valor C EUR", "FR0011253624"),
    ("Vivalor International", "FR0014001LS1"),
    ("CARMIGNAC Investissement A EUR", "FR0010148981"),
    ("FIDELITY FUNDS - WORLD FUND", "LU0069449576"),
    ("CLARTAN VALEURS", "LU1100076550"),
    ("CARMIGNAC PATRIMOINE", "FR0010135103"),
]

RECO_FUNDS_DEF = [
    ("Fonds en euros (EUROFUND)", "EUROFUND"),
    ("SYCOYIELD 2030 RC", "FR001400MCQ6"),
    ("R-Co Target 2029 HY", "FR0014002XJ3"),
    ("Euro Bond 1-3 Years", "LU0321462953"),
]

FUND_NAME_MAP = {isin: name for name, isin in RECO_FUNDS_CORE + RECO_FUNDS_DEF}

# Libellés FR -> codes internes pour l'affectation des versements
ALLOC_LABELS = {
    "Répartition égale": "equal",
    "Personnalisé": "custom",
    "Tout sur une ligne": "single",
}


# ------------------------------------------------------------
# Utils format
# ------------------------------------------------------------

def params_hash(values: Tuple[Any, ...]) -> str:
    try:
        payload = json.dumps(values, default=str, sort_keys=True)
    except Exception:
        payload = repr(values)
    return str(hash(payload))

def to_eur(x: Any) -> str:
    try:
        v = float(x)
    except Exception:
        return "—"
    s = f"{v:,.2f}".replace(",", "X").replace(".", ",").replace("X", " ")
    return s + " €"


def fmt_date(x: Any) -> str:
    try:
        return pd.Timestamp(x).strftime("%d/%m/%Y")
    except Exception:
        return "—"


def fmt_eur_fr(x: Any) -> str:
    try:
        v = float(x)
    except Exception:
        return "—"
    s = f"{v:,.2f}".replace(",", "X").replace(".", ",").replace("X", " ")
    return f"{s} €"


def fmt_pct_fr(x: Any) -> str:
    try:
        v = float(x)
    except Exception:
        return "—"
    s = f"{v:,.2f}".replace(",", "X").replace(".", ",").replace("X", " ")
    return f"{s} %"


# ------------------------------------------------------------
# XIRR
# ------------------------------------------------------------

def _npv(rate: float, cfs: List[Tuple[pd.Timestamp, float]]) -> float:
    t0 = cfs[0][0]
    return sum(cf / ((1 + rate) ** ((t - t0).days / 365.25)) for t, cf in cfs)


def xirr(cash_flows: List[Tuple[pd.Timestamp, float]], guess: float = 0.1) -> Optional[float]:
    if not cash_flows:
        return None
    cfs = sorted(cash_flows, key=lambda x: x[0])
    try:
        r = guess
        for _ in range(100):
            f = _npv(r, cfs)
            h = 1e-6
            f1 = _npv(r + h, cfs)
            d = (f1 - f) / h
            if abs(d) < 1e-12:
                break
            r2 = r - f / d
            if abs(r2 - r) < 1e-9:
                r = r2
                break
            r = r2
        return r
    except Exception:
        return None


# ------------------------------------------------------------
# API EODHD
# ------------------------------------------------------------

def _get_api_key() -> str:
    return st.secrets.get("EODHD_API_KEY", "")


@st.cache_data(show_spinner=False, ttl=3600)
def eodhd_get(path: str, params: Dict[str, Any] | None = None) -> Any:
    base = "https://eodhd.com/api"
    token = _get_api_key()
    p = {"api_token": token, "fmt": "json"}
    if params:
        p.update(params)
    with st.spinner("Chargement EODHD..."):
        r = requests.get(f"{base}{path}", params=p, timeout=20)
    r.raise_for_status()
    try:
        return r.json()
    except Exception:
        return None


@st.cache_data(show_spinner=False, ttl=3600)
def eodhd_search(q: str) -> List[Dict[str, Any]]:
    try:
        js = eodhd_get(f"/search/{q}")
        if isinstance(js, list):
            return js
    except Exception:
        pass
    return []


@st.cache_data(show_spinner=False, ttl=3600)
def eodhd_prices_daily(symbol: str) -> pd.DataFrame:
    try:
        js = eodhd_get(f"/eod/{symbol}", params={"period": "d"})
        if not isinstance(js, list) or len(js) == 0:
            return pd.DataFrame()
        df = pd.DataFrame(js)
        df["date"] = pd.to_datetime(df["date"])
        df.set_index("date", inplace=True)
        if "adjusted_close" in df.columns and pd.notnull(df["adjusted_close"]).any():
            df["Close"] = df["adjusted_close"].astype(float)
        elif "close" in df.columns:
            df["Close"] = df["close"].astype(float)
        else:
            return pd.DataFrame()
        return df[["Close"]].sort_index()
    except Exception:
        return pd.DataFrame()


def _symbol_candidates(isin_or_name: str) -> List[str]:
    val = str(isin_or_name).strip()
    if not val:
        return []
    if val.upper() == "EUROFUND":
        return ["EUROFUND"]
    candidates = [f"{val}.EUFUND", f"{val}.FUND", val]
    try:
        res = eodhd_search(val)
        for it in res:
            code = it.get("Code")
            exch = it.get("Exchange")
            if code and exch:
                candidates.append(f"{code}.{exch}")
    except Exception:
        pass
    seen = set()
    uniq = []
    for c in candidates:
        if c not in seen:
            seen.add(c)
            uniq.append(c)
    return uniq


def _get_close_on(df: pd.DataFrame, d: pd.Timestamp) -> float:
    if df.empty:
        return np.nan
    if d in df.index:
        return float(df.loc[d, "Close"])
    after = df.loc[df.index >= d]
    if not after.empty:
        return float(after.iloc[0]["Close"])
    return float(df.iloc[-1]["Close"])


def apply_annual_fee(df: pd.DataFrame, annual_fee_pct: float) -> pd.DataFrame:
    if df.empty or annual_fee_pct == 0:
        return df
    df = df.copy()
    fee_rate = float(annual_fee_pct) / 100.0
    base_date = df.index[0]
    day_offsets = (df.index - base_date).days.astype(float)
    fee_factors = (1.0 - fee_rate) ** (day_offsets / 365.0)
    df["Close"] = df["Close"].astype(float).to_numpy() * fee_factors
    return df


@st.cache_data(show_spinner=False, ttl=3600)
def get_price_series(
    isin_or_name: str, start: Optional[pd.Timestamp], euro_rate: float
) -> Tuple[pd.DataFrame, str, str]:
    """
    EUROFUND : série synthétique capitalisée à euro_rate %/an
    (cohérente avec Excel : compo sur jours calendaires)
    """
    debug = {"cands": []}
    val = str(isin_or_name).strip()
    if not val:
        return pd.DataFrame(), "", json.dumps(debug)

    # ✅ Fonds en euros — capitalisation annualisée (jours calendaires)
    if val.upper() == "EUROFUND":
        start_dt = (
            pd.Timestamp(start).normalize()
            if start is not None
            else pd.Timestamp("2000-01-03")
        )
        start_dt = max(start_dt, pd.Timestamp("2000-01-03"))

        idx = pd.bdate_range(start=start_dt, end=TODAY, freq="B")
        if len(idx) == 0:
            return pd.DataFrame(), "", "{}"

        df = pd.DataFrame(index=idx, columns=["Close"], dtype=float)
        df.iloc[0, 0] = 1.0

        r = float(euro_rate) / 100.0

        for i in range(1, len(df)):
            prev_val = df.iloc[i - 1, 0]
            delta_days = (df.index[i] - df.index[i - 1]).days  # ✅ jours calendaires
            df.iloc[i, 0] = prev_val * ((1.0 + r) ** (delta_days / 365.0))

        df = apply_annual_fee(df, ANNUAL_FEE_EURO_PCT)
        return df, "EUROFUND", "{}"

    # ✅ Instruments EODHD — recherche candidates puis EOD daily
    cands = _symbol_candidates(val)
    debug["cands"] = cands

    for sym in cands:
        df = eodhd_prices_daily(sym)
        if not df.empty:
            if start is not None:
                df = df.loc[df.index >= start]
            df = apply_annual_fee(df, ANNUAL_FEE_UC_PCT)
            return df, sym, json.dumps(debug)

    return pd.DataFrame(), "", json.dumps(debug)


@st.cache_data(show_spinner=False, ttl=3600)
def structured_series(
    start: pd.Timestamp,
    end: pd.Timestamp,
    annual_rate_pct: float,
    redemption_years: int,
) -> pd.DataFrame:
    """
    Série synthétique autocall (simplifiée) :
    - Prix d'achat = 1.0
    - Plat jusqu'à la date de remboursement estimée
    - Saut à 1 + (rate * years) le jour de remboursement, puis figé
    """
    start_dt = pd.Timestamp(start).normalize()
    end_dt = pd.Timestamp(end).normalize()
    idx = pd.bdate_range(start=start_dt, end=end_dt, freq="B")
    if len(idx) == 0:
        return pd.DataFrame()

    df = pd.DataFrame(index=idx, columns=["Close"], dtype=float)
    df.iloc[0, 0] = 1.0

    r = float(annual_rate_pct) / 100.0
    yrs = int(redemption_years)

    redemption_dt = start_dt + pd.DateOffset(years=yrs)

    # série plate + saut à partir du 1er jour ouvré >= redemption_dt
    redeemed = False
    for i in range(1, len(df)):
        d = df.index[i]
        df.iloc[i, 0] = df.iloc[i - 1, 0]

        if (not redeemed) and (d >= redemption_dt):
            df.iloc[i, 0] = 1.0 + r * yrs
            df.iloc[i:, 0] = df.iloc[i, 0]
            redeemed = True
            break

    # sécurité : propagation si besoin
    for i in range(1, len(df)):
        if pd.isna(df.iloc[i, 0]):
            df.iloc[i, 0] = df.iloc[i - 1, 0]

    return df


def _warn_once(key: str, msg: str) -> None:
    seen = st.session_state.get("WARN_ONCE")
    if not isinstance(seen, set):
        if isinstance(seen, list):
            seen = set(seen)
        else:
            seen = set()
    if key in seen:
        return
    seen.add(key)
    st.session_state["WARN_ONCE"] = seen
    st.warning(msg)


def _safe_struct_params(line: Dict[str, Any]) -> Tuple[float, int]:
    try:
        years = int(line.get("struct_years", 6))
    except Exception:
        years = 6
    years = max(1, years)
    try:
        rate = float(line.get("struct_rate", 8.0))
    except Exception:
        rate = 0.0
    rate = max(0.0, min(rate, 25.0))
    return rate, years


def get_series_for_line(
    line: Dict[str, Any],
    start: Optional[pd.Timestamp],
    euro_rate: float,
) -> Tuple[pd.DataFrame, str]:
    isin_or_name = str(line.get("isin") or line.get("name") or "").strip()
    sym_upper = isin_or_name.upper()

    if sym_upper == "EUROFUND":
        df, _, _ = get_price_series("EUROFUND", start, euro_rate)
        if df.empty:
            _warn_once("series_empty:EUROFUND", "Série indisponible pour le fonds en euros (EUROFUND).")
        return df, "EUROFUND"

    if sym_upper == "STRUCTURED":
        buy_ts = pd.Timestamp(line.get("buy_date") or start or TODAY).normalize()
        rate, years = _safe_struct_params(line)
        df = structured_series(
            start=buy_ts,
            end=TODAY,
            annual_rate_pct=rate,
            redemption_years=years,
        )
        if start is not None:
            df = df.loc[df.index >= pd.Timestamp(start)]
        if df.empty:
            label = line.get("name") or "Produit structuré"
            _warn_once(
                f"series_empty:STRUCTURED:{label}",
                f"Série indisponible pour {label} (STRUCTURED) : ligne ignorée.",
            )
        return df, "STRUCTURED"

    df, sym, _ = get_price_series(isin_or_name, start, euro_rate)
    if df.empty:
        label = line.get("name") or isin_or_name or "Ligne"
        _warn_once(
            f"series_empty:{isin_or_name}",
            f"Série indisponible pour {label} ({isin_or_name}) : ligne ignorée.",
        )
    return df, sym or isin_or_name

# ------------------------------------------------------------
# Alternatives si date < 1ère VL
# ------------------------------------------------------------

def suggest_alternative_funds(buy_date: pd.Timestamp, euro_rate: float) -> List[Tuple[str, str, pd.Timestamp]]:
    """
    Propose des fonds recommandés (core + défensifs) dont la première VL
    est antérieure ou égale à la date d'achat donnée.
    Retourne (nom, isin, date_inception).
    """
    alternatives: List[Tuple[str, str, pd.Timestamp]] = []
    universe = RECO_FUNDS_CORE + RECO_FUNDS_DEF

    for name, isin in universe:
        df, _, _ = get_price_series(isin, None, euro_rate)
        if df.empty:
            continue
        inception = df.index.min()
        if inception <= buy_date:
            alternatives.append((name, isin, inception))

    return alternatives


def correlation_matrix_from_lines(
    lines: List[Dict[str, Any]],
    euro_rate: float,
    years: int = 3,
    min_points: int = 30,
) -> pd.DataFrame:
    """
    Construit une matrice de corrélation des rendements quotidiens
    pour les lignes d'un portefeuille donné.

    - On récupère les VL quotidiennes via get_price_series
    - On restreint à 'years' années de données (fenêtre glissante)
    - On calcule les rendements journaliers (pct_change)
    - On renvoie corrélation de ces rendements.
    """
    series_map: Dict[str, pd.Series] = {}
    cutoff = TODAY - pd.Timedelta(days=365 * years)

    for ln in lines:
        label = ln.get("name") or ln.get("isin") or "Ligne"
        key = f"{label} ({ln.get('isin','')})"

        df, _ = get_series_for_line(ln, None, euro_rate)
        if df.empty:
            continue

        s = df["Close"].astype(float)
        s = s[s.index >= cutoff]
        if s.size < min_points:
            continue

        series_map[key] = s

    if len(series_map) < 2:
        return pd.DataFrame()

    df_prices = pd.DataFrame(series_map).dropna(how="all")
    if df_prices.shape[0] < min_points:
        return pd.DataFrame()

    returns = df_prices.pct_change().dropna(how="any")
    if returns.empty:
        return pd.DataFrame()

    corr = returns.corr()
    return corr


# ------------------------------------------------------------
# Calendrier versements & poids
# ------------------------------------------------------------

def _month_schedule(d0: pd.Timestamp, d1: pd.Timestamp) -> List[pd.Timestamp]:
    if d0 > d1:
        return []
    out = []
    cur = pd.Timestamp(d0.year, d0.month, 1)
    stop = pd.Timestamp(d1.year, d1.month, 1)
    while cur <= stop:
        bdays = pd.bdate_range(start=cur, end=cur + pd.offsets.MonthEnd(0))
        if len(bdays) > 0:
            out.append(bdays[0])
        cur = cur + pd.offsets.MonthBegin(1)
    return out


def _weights_for(
    lines: List[Dict[str, Any]],
    alloc_mode: str,
    custom_weights: Dict[int, float],
    single_target: Optional[int],
) -> Dict[int, float]:
    keys = [id(ln) for ln in lines]
    if len(keys) == 0:
        return {}
    if alloc_mode == "equal":
        w = 1.0 / len(keys)
        return {k: w for k in keys}
    if alloc_mode == "custom":
        tot = sum(max(0.0, float(custom_weights.get(id(ln), 0.0))) for ln in lines)
        if tot <= 0:
            w = 1.0 / len(keys)
            return {k: w for k in keys}
        return {id(ln): max(0.0, float(custom_weights.get(id(ln), 0.0))) / tot for ln in lines}
    if alloc_mode == "single":
        target = single_target
        return {id(ln): (1.0 if id(ln) == target else 0.0) for ln in lines}
    w = 1.0 / len(keys)
    return {k: w for k in keys}


# ------------------------------------------------------------
# Calcul par ligne (avec frais)
# ------------------------------------------------------------

def compute_line_metrics(
    line: Dict[str, Any], fee_pct: float, euro_rate: float
) -> Tuple[float, float, float]:
    amt_brut = float(line.get("amount_gross", 0.0))
    net_amt = amt_brut * (1.0 - fee_pct / 100.0)
    buy_ts = pd.Timestamp(line.get("buy_date"))
    px_manual = line.get("buy_px", None)

    dfp, sym_used = get_series_for_line(line, buy_ts, euro_rate)
    if dfp.empty:
        return float(net_amt), np.nan, 0.0

    sym_upper = str(sym_used or line.get("isin") or "").upper()
    if sym_upper == "EUROFUND":
        px = _get_close_on(dfp, buy_ts)
    else:
        if px_manual not in (None, "", 0, "0"):
            try:
                px = float(px_manual)
            except Exception:
                px = _get_close_on(dfp, buy_ts)
        else:
            px = _get_close_on(dfp, buy_ts)

    qty = (net_amt / px) if px and px > 0 else 0.0
    return float(net_amt), float(px), float(qty)


# ------------------------------------------------------------
# Simulation d'un portefeuille (avec contrôle 1ère VL)
# + distinction poids mensuels / ponctuels
# ------------------------------------------------------------

def simulate_portfolio(
    lines: List[Dict[str, Any]],
    monthly_amt_gross: float,
    one_amt_gross: float,
    one_date: date,
    alloc_mode: str,
    custom_weights_monthly: Optional[Dict[int, float]],
    custom_weights_oneoff: Optional[Dict[int, float]],
    single_target: Optional[int],
    euro_rate: float,
    fee_pct: float,
    portfolio_label: str = "",
) -> Tuple[pd.DataFrame, float, float, float, Optional[float], pd.Timestamp, pd.Timestamp]:
    if not lines:
        return pd.DataFrame(), 0.0, 0.0, 0.0, None, TODAY, TODAY

    price_map: Dict[int, pd.Series] = {}
    eff_buy_date: Dict[int, pd.Timestamp] = {}
    buy_price_used: Dict[int, float] = {}

    invalid_found = False
    date_warnings = st.session_state.setdefault("DATE_WARNINGS", [])

    for ln in lines:
        key_id = id(ln)
        df_full, sym = get_series_for_line(ln, None, euro_rate)

        # Sécurité
        if df_full.empty:
            continue

        inception = df_full.index.min()
        d_buy = pd.Timestamp(ln["buy_date"])

        if d_buy < inception:
            invalid_found = True
            ln["invalid_date"] = True
            ln["inception_date"] = inception

            alts = suggest_alternative_funds(d_buy, euro_rate)
            if alts:
                alt_lines = [
                    f"- {name} ({isin}), historique depuis le {fmt_date(incep)}"
                    for name, isin, incep in alts
                ]
                alt_msg = "\n".join(alt_lines)
            else:
                alt_msg = "Aucun fonds recommandé ne dispose d'un historique suffisant pour cette date."

            date_warnings.append(
                f"[{portfolio_label}] {ln.get('name','(sans nom)')} "
                f"({ln.get('isin','—')}) :\n"
                f"- Date d'achat saisie : {fmt_date(d_buy)}\n"
                f"- 1ère VL disponible : {fmt_date(inception)}\n\n"
                f"Impossible de simuler ce fonds sur toute la période demandée.\n"
                f"Propositions d'alternatives pour l'analyse historique :\n{alt_msg}"
            )
            continue

        ln["sym_used"] = sym
        df = df_full

        if d_buy in df.index:
            px_buy = float(df.loc[d_buy, "Close"])
            eff_dt = d_buy
        else:
            after = df.loc[df.index >= d_buy]
            if after.empty:
                px_buy = float(df.iloc[-1]["Close"])
                eff_dt = df.index[-1]
            else:
                px_buy = float(after.iloc[0]["Close"])
                eff_dt = after.index[0]

        px_manual = ln.get("buy_px", None)
        px_for_qty = float(px_manual) if (px_manual not in (None, "", 0, "0")) else px_buy

        price_map[key_id] = df["Close"].astype(float)
        eff_buy_date[key_id] = eff_dt
        buy_price_used[key_id] = px_for_qty

    if invalid_found and not price_map:
        return pd.DataFrame(), 0.0, 0.0, 0.0, None, TODAY, TODAY
    if not price_map:
        return pd.DataFrame(), 0.0, 0.0, 0.0, None, TODAY, TODAY

    start_min = min(eff_buy_date.values())
    start_full = max(eff_buy_date.values())

    bidx = pd.bdate_range(start=start_min, end=TODAY, freq="B")
    prices = pd.DataFrame(index=bidx)
    for key_id, s in price_map.items():
        prices[key_id] = s.reindex(bidx).ffill()

    qty_events = pd.DataFrame(0.0, index=bidx, columns=prices.columns)
    total_brut = 0.0
    total_net = 0.0
    cash_flows: List[Tuple[pd.Timestamp, float]] = []

    # Achats initiaux
    for ln in lines:
        key_id = id(ln)
        if key_id not in prices.columns:
            continue
        brut = float(ln.get("amount_gross", 0.0))
        net = brut * (1.0 - fee_pct / 100.0)
        px = float(buy_price_used[key_id])
        dt = eff_buy_date[key_id]
        if brut > 0 and px > 0:
            q = net / px
            tgt = dt if dt in qty_events.index else qty_events.index[qty_events.index >= dt][0]
            qty_events.loc[tgt, key_id] += q
            total_brut += brut
            total_net += net
            cash_flows.append((tgt, -brut))

    # Poids pour versements mensuels / ponctuels
    weights_monthly = _weights_for(
        lines,
        alloc_mode,
        custom_weights_monthly or {},
        single_target,
    )
    weights_oneoff = _weights_for(
        lines,
        alloc_mode,
        custom_weights_oneoff or {},
        single_target,
    )

    # Versement ponctuel
    if one_amt_gross > 0:
        dt = pd.Timestamp(one_date)
        if dt not in qty_events.index:
            after = qty_events.index[qty_events.index >= dt]
            if len(after) > 0:
                dt = after[0]
            else:
                dt = None
        if dt is not None:
            net_amt = one_amt_gross * (1.0 - fee_pct / 100.0)
            for ln in lines:
                key_id = id(ln)
                w = weights_oneoff.get(key_id, 0.0)
                if w <= 0 or key_id not in prices.columns:
                    continue
                px = float(prices.loc[dt, key_id])
                if px > 0:
                    qty_events.loc[dt, key_id] += (net_amt * w) / px
            total_brut += float(one_amt_gross)
            total_net += float(net_amt)
            cash_flows.append((dt, -float(one_amt_gross)))

    # Mensuels
    if monthly_amt_gross > 0:
        sched = _month_schedule(start_min, TODAY)
        for dt in sched:
            if dt not in qty_events.index:
                after = qty_events.index[qty_events.index >= dt]
                if len(after) == 0:
                    continue
                dt = after[0]
            net_m = monthly_amt_gross * (1.0 - fee_pct / 100.0)
            for ln in lines:
                key_id = id(ln)
                w = weights_monthly.get(key_id, 0.0)
                if w <= 0 or key_id not in prices.columns:
                    continue
                px = float(prices.loc[dt, key_id])
                if px > 0:
                    qty_events.loc[dt, key_id] += (net_m * w) / px
            total_brut += float(monthly_amt_gross)
            total_net += float(net_m)
            cash_flows.append((dt, -float(monthly_amt_gross)))

    qty_cum = qty_events.cumsum()
    values = (qty_cum * prices).sum(axis=1)
    df_val = pd.DataFrame({"Valeur": values})
    final_val = float(df_val["Valeur"].iloc[-1]) if not df_val.empty else 0.0

    cash_flows.append((TODAY, final_val))
    irr = xirr(cash_flows)

    return df_val, total_brut, total_net, final_val, (irr * 100.0 if irr is not None else None), start_min, start_full

# ------------------------------------------------------------
# Cartes lignes (édition / suppression)
# ------------------------------------------------------------

def _line_card(line: Dict[str, Any], idx: int, port_key: str):
    state_key = f"edit_mode_{port_key}_{idx}"
    if state_key not in st.session_state:
        st.session_state[state_key] = False

    fee_pct = st.session_state.get("FEE_A", 0.0) if port_key == "A_lines" else st.session_state.get("FEE_B", 0.0)
    euro_rate = st.session_state.get("EURO_RATE_PREVIEW", 2.0)
    net_amt, buy_px, qty_disp = compute_line_metrics(line, fee_pct, euro_rate)

    with st.container(border=True):
        cols = st.columns([3, 2, 2, 2, 1])
        with cols[0]:
            st.markdown(f"**{line.get('name','—')}**")
            st.caption(f"ISIN / Code : `{line.get('isin','—')}`")
            st.caption(f"Symbole EODHD : `{line.get('sym_used','—')}`")
            if line.get("invalid_date"):
                st.markdown(
                    f"⚠️ Date d'achat antérieure à la 1ère VL ({fmt_date(line.get('inception_date'))}).",
                )
        with cols[1]:
            st.markdown(f"Investi (brut)\n\n**{to_eur(line.get('amount_gross', 0.0))}**")
            st.caption(f"Net après frais {fee_pct:.1f}% : **{to_eur(net_amt)}**")
            st.caption(f"Date d'achat : {fmt_date(line.get('buy_date'))}")
        with cols[2]:
            st.markdown(f"VL d'achat\n\n**{to_eur(buy_px)}**")
            st.caption(f"Quantité : {qty_disp:.6f}")
            if line.get("note"):
                st.caption(line["note"])
        with cols[3]:
            try:
                dfl, _ = get_series_for_line(line, None, euro_rate)
                last = float(dfl["Close"].iloc[-1]) if not dfl.empty else np.nan
                st.markdown(f"VL actuelle : **{to_eur(last)}**")
            except Exception:
                st.markdown("VL actuelle : —")
        with cols[4]:
            if not st.session_state[state_key]:
                if st.button("✏️", key=f"edit_{port_key}_{idx}", help="Modifier"):
                    st.session_state[state_key] = True
                    st.experimental_rerun()
            if st.button("🗑️", key=f"del_{port_key}_{idx}", help="Supprimer"):
                st.session_state[port_key].pop(idx)
                st.experimental_rerun()

        if st.session_state[state_key]:
            with st.form(key=f"form_edit_{port_key}_{idx}", clear_on_submit=False):
                c1, c2, c3, c4 = st.columns([2, 2, 2, 1])
                with c1:
                    new_amount = st.text_input("Montant investi (brut) €", value=str(line.get("amount_gross", "")))
                with c2:
                    new_date = st.date_input("Date d’achat", value=pd.Timestamp(line.get("buy_date")).date())
                with c3:
                    new_px = st.text_input("Prix d’achat (optionnel)", value=str(line.get("buy_px", "")))
                with c4:
                    st.caption(" ")
                    submitted = st.form_submit_button("💾 Enregistrer")
                if submitted:
                    try:
                        amt_gross = float(str(new_amount).replace(" ", "").replace(",", "."))
                        assert amt_gross > 0
                    except Exception:
                        st.warning("Montant brut invalide.")
                        st.stop()
                    buy_ts = pd.Timestamp(new_date)
                    line["amount_gross"] = float(amt_gross)
                    line["buy_date"] = buy_ts
                    if new_px.strip():
                        try:
                            line["buy_px"] = float(str(new_px).replace(",", "."))
                        except Exception:
                            line["buy_px"] = ""
                    else:
                        line["buy_px"] = ""
                    line.pop("invalid_date", None)
                    line.pop("inception_date", None)
                    st.session_state[state_key] = False
                    st.success("Ligne mise à jour.")
                    st.experimental_rerun()


def portfolio_summary_dataframe(port_key: str) -> pd.DataFrame:
    """
    Construit un DataFrame synthétique par ligne :
    Nom, ISIN, Net investi, Valeur actuelle, Perf € et Perf %.
    """
    fee_pct = st.session_state.get("FEE_A", 0.0) if port_key == "A_lines" else st.session_state.get("FEE_B", 0.0)
    euro_rate = st.session_state.get("EURO_RATE_PREVIEW", 2.0)
    lines = st.session_state.get(port_key, [])

    rows: List[Dict[str, Any]] = []

    for ln in lines:
        net_amt, buy_px, qty = compute_line_metrics(ln, fee_pct, euro_rate)

        dfl, _ = get_series_for_line(ln, None, euro_rate)
        if dfl.empty:
            continue
        if not dfl.empty:
            last_px = float(dfl["Close"].iloc[-1])
        else:
            last_px = np.nan

        val_now = qty * last_px if last_px == last_px else 0.0
        perf_abs = val_now - net_amt
        perf_pct = (val_now / net_amt - 1.0) * 100.0 if net_amt > 0 else np.nan

        rows.append(
            {
                "Nom": ln.get("name", ""),
                "ISIN / Code": ln.get("isin", ""),
                "Net investi €": net_amt,
                "Valeur actuelle €": val_now,
                "Perf €": perf_abs,
                "Perf %": perf_pct,
            }
        )

    df = pd.DataFrame(rows)
    return df



def build_positions_dataframe(port_key: str) -> pd.DataFrame:
    """
    Construit un DataFrame par ligne :
    Nom, ISIN, Date d'achat, Net investi, Valeur actuelle, Perf € et Perf %.
    """
    fee_pct = (
        st.session_state.get("FEE_A", 0.0)
        if port_key == "A_lines"
        else st.session_state.get("FEE_B", 0.0)
    )

    euro_rate = (
        st.session_state.get("EURO_RATE_A", 2.0)
        if port_key == "A_lines"
        else st.session_state.get("EURO_RATE_B", 2.5)
    )

    lines = st.session_state.get(port_key, [])
    rows: List[Dict[str, Any]] = []

    for ln in lines:
        buy_ts = pd.Timestamp(ln.get("buy_date"))
        net_amt, buy_px, qty = compute_line_metrics(ln, fee_pct, euro_rate)
        dfl, _ = get_series_for_line(ln, buy_ts, euro_rate)
        if dfl.empty:
            continue

        if not dfl.empty:
            last_px = float(dfl["Close"].iloc[-1])
        else:
            last_px = np.nan

        val_now = qty * last_px if last_px == last_px else 0.0
        perf_abs = val_now - net_amt
        perf_pct = (val_now / net_amt - 1.0) * 100.0 if net_amt > 0 else np.nan

        rows.append(
            {
                "Nom": ln.get("name", ""),
                "ISIN / Code": ln.get("isin", ""),
                "Date d'achat": fmt_date(ln.get("buy_date")),
                "Net investi €": net_amt,
                "Valeur actuelle €": val_now,
                "Perf €": perf_abs,
                "Perf %": perf_pct,
            }
        )

    return pd.DataFrame(rows)

# ------------------------------------------------------------
# Tableau synthétique par ligne (un seul tableau par portefeuille)
# ------------------------------------------------------------

def positions_table(title: str, port_key: str):
    """
    Affiche un seul tableau synthétique par portefeuille :
    Nom, ISIN, Date d'achat, Net investi, Valeur actuelle, Perf € et Perf %.
    """
    fee_pct = (
        st.session_state.get("FEE_A", 0.0)
        if port_key == "A_lines"
        else st.session_state.get("FEE_B", 0.0)
    )

    # ✅ Taux fonds euros par portefeuille (au lieu de EURO_RATE_PREVIEW)
    euro_rate = (
        st.session_state.get("EURO_RATE_A", 2.0)
        if port_key == "A_lines"
        else st.session_state.get("EURO_RATE_B", 2.5)
    )

    lines = st.session_state.get(port_key, [])
    rows: List[Dict[str, Any]] = []

    for ln in lines:
        buy_ts = pd.Timestamp(ln.get("buy_date"))

        # Montant net investi, VL d'achat et quantité
        net_amt, buy_px, qty = compute_line_metrics(ln, fee_pct, euro_rate)

        # ✅ IMPORTANT : on récupère la série "depuis buy_ts" pour éviter le mismatch EUROFUND
        dfl, _ = get_series_for_line(ln, buy_ts, euro_rate)
        if dfl.empty:
            continue

        if not dfl.empty:
            last_px = float(dfl["Close"].iloc[-1])
        else:
            last_px = np.nan

        # Valeur actuelle et performance
        val_now = qty * last_px if last_px == last_px else 0.0
        perf_abs = val_now - net_amt
        perf_pct = (val_now / net_amt - 1.0) * 100.0 if net_amt > 0 else np.nan

        rows.append(
            {
                "Nom": ln.get("name", ""),
                "ISIN / Code": ln.get("isin", ""),
                "Date d'achat": fmt_date(ln.get("buy_date")),
                "Net investi €": net_amt,
                "Valeur actuelle €": val_now,
                "Perf €": perf_abs,
                "Perf %": perf_pct,
            }
        )

    st.markdown(f"### {title}")
    df = pd.DataFrame(rows)
    if df.empty:
        st.info("Aucune ligne.")
    else:
        st.dataframe(
            df.style.format(
                {
                    "Net investi €": to_eur,
                    "Valeur actuelle €": to_eur,
                    "Perf €": to_eur,
                    "Perf %": "{:,.2f}%".format,
                }
            ),
            hide_index=True,
            use_container_width=True,
        )


def _prepare_pie_df(df_positions: pd.DataFrame, max_items: int = 8, min_pct: float = 0.03) -> pd.DataFrame:
    if df_positions.empty:
        return df_positions
    df = df_positions.copy()
    df = df[df["Valeur actuelle €"] > 0]
    if df.empty:
        return df
    total = df["Valeur actuelle €"].sum()
    df["Part %"] = df["Valeur actuelle €"] / total
    df = df.sort_values("Valeur actuelle €", ascending=False)
    if len(df) > max_items:
        df_main = df.iloc[:max_items].copy()
        df_other = df.iloc[max_items:]
        df_main = pd.concat(
            [
                df_main,
                pd.DataFrame(
                    {
                        "Nom": ["Autres"],
                        "Valeur actuelle €": [df_other["Valeur actuelle €"].sum()],
                        "Part %": [df_other["Valeur actuelle €"].sum() / total],
                    }
                ),
            ],
            ignore_index=True,
        )
        df = df_main
    else:
        small = df[df["Part %"] < min_pct]
        if not small.empty and len(df) > 1:
            df_main = df[df["Part %"] >= min_pct]
            df_other = pd.DataFrame(
                {
                    "Nom": ["Autres"],
                    "Valeur actuelle €": [small["Valeur actuelle €"].sum()],
                    "Part %": [small["Valeur actuelle €"].sum() / total],
                }
            )
            df = pd.concat([df_main, df_other], ignore_index=True)
    df["Part %"] = df["Part %"] * 100.0
    return df


# ------------------------------------------------------------
# Analytics internes : retours, corrélation, volatilité
# ------------------------------------------------------------


def _build_returns_df(
    lines: List[Dict[str, Any]],
    euro_rate: float,
    years: int = 3,
    min_points: int = 60,
) -> pd.DataFrame:
    """
    Construit un DataFrame de rendements journaliers (pct_change)
    pour toutes les lignes du portefeuille avec un historique suffisant.
    Index = dates, colonnes = "Nom (ISIN)".
    """
    cutoff = TODAY - pd.Timedelta(days=365 * years)
    series_map: Dict[str, pd.Series] = {}

    for ln in lines:
        label = (ln.get("name") or ln.get("isin") or "Ligne").strip()
        isin = (ln.get("isin") or "").strip()
        key = f"{label} ({isin})" if isin else label

        df, _ = get_series_for_line(ln, None, euro_rate)
        if df.empty:
            continue

        s = df["Close"].astype(float)
        s = s[s.index >= cutoff]
        if s.size < min_points:
            continue

        series_map[key] = s

    if not series_map:
        return pd.DataFrame()

    df_prices = pd.DataFrame(series_map).dropna(how="any")
    if df_prices.shape[0] < min_points:
        return pd.DataFrame()

    returns = df_prices.pct_change().dropna(how="any")
    return returns



def correlation_matrix_from_lines(
    lines: List[Dict[str, Any]],
    euro_rate: float,
    years: int = 3,
    min_points: int = 60,
) -> pd.DataFrame:
    """
    Matrice de corrélation entre les lignes du portefeuille,
    basée sur les rendements journaliers.
    """
    rets = _build_returns_df(lines, euro_rate, years=years, min_points=min_points)
    if rets.empty:
        return pd.DataFrame()
    return rets.corr()


def volatility_table_from_lines(
    lines: List[Dict[str, Any]],
    euro_rate: float,
    years: int = 3,
    min_points: int = 60,
) -> pd.DataFrame:
    """
    Volatilité annuelle par ligne (et écart-type quotidien).
    """
    rets = _build_returns_df(lines, euro_rate, years=years, min_points=min_points)
    if rets.empty:
        return pd.DataFrame()

    rows = []
    for col in rets.columns:
        r = rets[col]
        daily_std = float(r.std())
        ann_std = daily_std * np.sqrt(252.0)
        rows.append(
            {
                "Ligne": col,
                "Écart-type quotidien %": daily_std * 100.0,
                "Volatilité annuelle %": ann_std * 100.0,
                "Nombre de points": int(r.count()),
            }
        )
    df = pd.DataFrame(rows)
    return df.sort_values("Volatilité annuelle %", ascending=False)


def portfolio_risk_stats(
    lines: List[Dict[str, Any]],
    euro_rate: float,
    years: int = 3,
    min_points: int = 60,
) -> Optional[Dict[str, float]]:
    """
    Calcule quelques stats globales de risque pour le portefeuille :
    - volatilité annuelle
    - max drawdown sur la période.
    Pondération par montant net investi (approximatif).
    """
    rets = _build_returns_df(lines, euro_rate, years=years, min_points=min_points)
    if rets.empty:
        return None

    # Pondération par net investi (approche simple)
    net_by_col: Dict[str, float] = {}
    fee_A = st.session_state.get("FEE_A", 0.0)
    fee_B = st.session_state.get("FEE_B", 0.0)

    # on détecte si c'est A ou B via présence dans session_state
    # (on ne connait pas port_key ici, donc approximation : on regarde les deux)
    for ln in lines:
        label = (ln.get("name") or ln.get("isin") or "Ligne").strip()
        isin = (ln.get("isin") or "").strip()
        key = f"{label} ({isin})" if isin else label

        # On essaie d'utiliser les deux grilles de frais, c'est approximatif
        net_A, _, _ = compute_line_metrics(ln, fee_A, euro_rate)
        net_B, _, _ = compute_line_metrics(ln, fee_B, euro_rate)
        net = max(net_A, net_B)  # on prend le plus élevé comme proxy
        if net > 0:
            net_by_col[key] = net

    # normalisation des poids
    weights = {}
    tot = sum(net_by_col.get(c, 0.0) for c in rets.columns)
    if tot <= 0:
        return None
    for c in rets.columns:
        w = net_by_col.get(c, 0.0) / tot
        weights[c] = w

    # série de rendement du portefeuille
    w_vec = np.array([weights[c] for c in rets.columns])
    rp = rets.to_numpy().dot(w_vec)
    rp = pd.Series(rp, index=rets.index)

    daily_std = float(rp.std())
    vol_ann = daily_std * np.sqrt(252.0)

    # max drawdown
    cum = (1.0 + rp).cumprod()
    running_max = cum.cummax()
    dd = cum / running_max - 1.0
    max_dd = float(dd.min())  # négatif

    return {
        "vol_ann_pct": vol_ann * 100.0,
        "max_dd_pct": max_dd * 100.0,
    }


def _corr_heatmap_chart(corr: pd.DataFrame, title: str) -> Optional[alt.Chart]:
    """
    Construit une heatmap Altair pour visualiser la matrice de corrélation.
    """
    if corr.empty or corr.shape[0] < 2:
        return None

    df_corr = corr.copy()
    df_corr["Ligne1"] = df_corr.index
    df_melt = df_corr.melt(id_vars="Ligne1", var_name="Ligne2", value_name="corr")

    base = (
        alt.Chart(df_melt)
        .encode(
            x=alt.X("Ligne1:O", sort=None, title=""),
            y=alt.Y("Ligne2:O", sort=None, title=""),
        )
    )

    heat = base.mark_rect().encode(
        color=alt.Color("corr:Q", scale=alt.Scale(domain=[-1, 1])),
        tooltip=[
            alt.Tooltip("Ligne1:N", title="Ligne 1"),
            alt.Tooltip("Ligne2:N", title="Ligne 2"),
            alt.Tooltip("corr:Q", title="Corrélation", format=".2f"),
        ],
    )

    text = base.mark_text(baseline="middle").encode(
        text=alt.Text("corr:Q", format=".2f"),
    )

    return (heat + text).properties(title=title, height=300)

# ------------------------------------------------------------
# Blocs de saisie : soit fonds recommandés, soit saisie libre
# ------------------------------------------------------------

# ------------------------------------------------------------
# Blocs de saisie : soit fonds recommandés, soit saisie libre
# ------------------------------------------------------------
# ------------------------------------------------------------
# Blocs de saisie : soit fonds recommandés, soit saisie libre
# ------------------------------------------------------------

def _add_from_reco_block(port_key: str, label: str):
    st.subheader(label)

    cat = st.selectbox(
        "Catégorie",
        ["Core (référence)", "Défensif", "Produits structurés"],
        key=f"reco_cat_{port_key}",
    )

    # ✅ Date d'achat centralisée (versement initial uniquement)
    buy_date = (
        st.session_state.get("INIT_A_DATE", pd.Timestamp("2024-01-02").date())
        if port_key == "A_lines"
        else st.session_state.get("INIT_B_DATE", pd.Timestamp("2024-01-02").date())
    )

    # ============================
    # CAS 1 — PRODUIT STRUCTURÉ
    # ============================
    if cat == "Produits structurés":
        st.markdown("### Produit structuré (Autocall)")

        c1, c2 = st.columns(2)
        with c1:
            amount = st.text_input(
                "Montant investi (brut) €",
                value="",
                key=f"struct_amt_{port_key}",
            )
        with c2:
            struct_years = st.number_input(
                "Durée estimée avant remboursement (années)",
                min_value=1,
                max_value=12,
                value=6,
                step=1,
                key=f"struct_years_{port_key}",
            )

        struct_rate = st.number_input(
            "Rendement annuel estimé (%)",
            min_value=0.0,
            max_value=25.0,
            value=8.0,
            step=0.10,
            key=f"struct_rate_{port_key}",
        )

        st.caption(
            f"Date d’investissement initiale : {pd.Timestamp(buy_date).strftime('%d/%m/%Y')}"
        )

        if st.button("➕ Ajouter le produit structuré", key=f"struct_add_{port_key}"):
            try:
                amt = float(str(amount).replace(" ", "").replace(",", "."))
                assert amt > 0
            except Exception:
                st.warning("Montant invalide.")
                return

            ln = {
                "name": f"Produit structuré ({struct_rate:.2f}% / {int(struct_years)} ans)",
                "isin": "STRUCTURED",
                "amount_gross": float(amt),
                "buy_date": pd.Timestamp(buy_date),
                "buy_px": 1.0,
                "struct_rate": float(struct_rate),
                "struct_years": int(struct_years),
                "note": "",
                "sym_used": "STRUCTURED",
            }
            st.session_state[port_key].append(ln)
            st.success("Produit structuré ajouté.")
        return  # ✅ IMPORTANT : on sort de la fonction pour ne pas afficher la partie fonds

    # ============================
    # CAS 2 — FONDS CLASSIQUES
    # ============================
    if cat == "Core (référence)":
        fonds_list = RECO_FUNDS_CORE
    else:
        fonds_list = RECO_FUNDS_DEF

    options = [f"{nm} ({isin})" for nm, isin in fonds_list]
    choice = st.selectbox("Fonds recommandé", options, key=f"reco_choice_{port_key}")
    idx = options.index(choice) if choice in options else 0
    name, isin = fonds_list[idx]

    c1, c2 = st.columns([2, 2])
    with c1:
        amount = st.text_input("Montant investi (brut) €", value="", key=f"reco_amt_{port_key}")
    with c2:
        st.caption(f"Date d’achat (versement initial) : {pd.Timestamp(buy_date).strftime('%d/%m/%Y')}")

    px = st.text_input("Prix d’achat (optionnel)", value="", key=f"reco_px_{port_key}")

    if st.button("➕ Ajouter ce fonds recommandé", key=f"reco_add_{port_key}"):
        try:
            amt = float(str(amount).replace(" ", "").replace(",", "."))
            assert amt > 0
        except Exception:
            st.warning("Montant invalide.")
            return

        ln = {
            "name": name,
            "isin": isin,
            "amount_gross": float(amt),
            "buy_date": pd.Timestamp(buy_date),
            "buy_px": float(str(px).replace(",", ".")) if px.strip() else "",
            "note": "",
            "sym_used": "",
        }
        st.session_state[port_key].append(ln)
        st.success("Fonds recommandé ajouté.")


def _add_line_form_free(port_key: str, label: str):
    st.subheader(label)

    # ✅ Date d'achat centralisée (versement initial)
    buy_date_central = (
        st.session_state.get("INIT_A_DATE", pd.Timestamp("2024-01-02").date())
        if port_key == "A_lines"
        else st.session_state.get("INIT_B_DATE", pd.Timestamp("2024-01-02").date())
    )

    with st.form(key=f"form_add_free_{port_key}", clear_on_submit=False):
        c1, c2 = st.columns([3, 2])

        with c1:
            name = st.text_input("Nom du fonds (libre)", value="")
            isin = st.text_input("ISIN ou code (peut être 'EUROFUND')", value="")

        with c2:
            amount = st.text_input("Montant investi (brut) €", value="")
            st.caption(
                f"Date d’achat (versement initial) : "
                f"{pd.Timestamp(buy_date_central).strftime('%d/%m/%Y')}"
            )

        px = st.text_input("Prix d’achat (optionnel)", value="")
        note = st.text_input("Note (optionnel)", value="")
        add_btn = st.form_submit_button("➕ Ajouter cette ligne")

    if not add_btn:
        return

    isin_final = isin.strip()
    name_final = name.strip()

    # Si nom vide mais ISIN renseigné : tentative de récupération du nom
    if not name_final and isin_final:
        res = eodhd_search(isin_final)
        match = None
        for it in res:
            if it.get("ISIN") == isin_final:
                match = it
                break
        if match is None and res:
            match = res[0]
        if match:
            name_final = match.get("Name", isin_final)

    if not name_final and isin_final.upper() == "EUROFUND":
        name_final = "Fonds en euros (EUROFUND)"

    if not name_final:
        name_final = isin_final or "—"

    try:
        amt = float(str(amount).replace(" ", "").replace(",", "."))
        assert amt > 0
    except Exception:
        st.warning("Montant invalide.")
        return

    ln = {
        "name": name_final,
        "isin": isin_final or name_final,
        "amount_gross": float(amt),
        "buy_date": pd.Timestamp(buy_date_central),  # ✅ applique la date centrale
        "buy_px": float(str(px).replace(",", ".")) if px.strip() else "",
        "note": note.strip(),
        "sym_used": "",
    }

    st.session_state[port_key].append(ln)
    st.success("Ligne ajoutée.")


@st.cache_data(show_spinner=False, ttl=3600)
def _build_returns_matrix(
    isins: Tuple[str, ...],
    euro_rate: float,
    start: pd.Timestamp,
    end: pd.Timestamp,
    min_points: int = 60,
) -> Tuple[pd.DataFrame, List[str], Dict[str, str]]:
    series_map: Dict[str, pd.Series] = {}
    warnings: List[str] = []
    status: Dict[str, str] = {}

    for isin in isins:
        df, _, _ = get_price_series(isin, None, euro_rate)
        if df.empty or df["Close"].dropna().shape[0] < min_points:
            warnings.append(f"{isin} : historique insuffisant")
            status[isin] = "insufficient"
            continue
        s = df["Close"].astype(float)
        s = s[(s.index >= start) & (s.index <= end)]
        if s.dropna().shape[0] < min_points:
            warnings.append(f"{isin} : historique insuffisant sur la fenêtre")
            status[isin] = "insufficient"
            continue
        series_map[isin] = s
        status[isin] = "ok"

    if not series_map:
        return pd.DataFrame(), warnings, status

    prices = pd.DataFrame(series_map).ffill().dropna(how="any")
    if prices.shape[0] < min_points:
        return pd.DataFrame(), warnings, status
    returns = prices.pct_change().dropna(how="any")
    return returns, warnings, status


def _suggest_weights(
    returns: pd.DataFrame,
    max_weight: float,
    min_funds: int,
) -> Dict[str, float]:
    if returns.empty:
        return {}
    corr = returns.corr()
    vols = returns.std() * np.sqrt(252.0)
    avg_corr = corr.mean()
    ranked = avg_corr.sort_values().index.tolist()
    min_funds = max(1, min(min_funds, len(ranked)))
    selected = ranked[:min_funds]
    inv_vol = 1 / vols[selected]
    weights = (inv_vol / inv_vol.sum()).to_dict()

    # cap weights if needed
    if max_weight > 0 and max_weight < 1:
        for _ in range(10):
            over = {k: v for k, v in weights.items() if v > max_weight}
            if not over:
                break
            excess = sum(v - max_weight for v in over.values())
            for k in over:
                weights[k] = max_weight
            remaining = {k: v for k, v in weights.items() if k not in over}
            if not remaining:
                break
            total_remaining = sum(remaining.values())
            for k in remaining:
                weights[k] += excess * (remaining[k] / total_remaining)

    total = sum(weights.values())
    if total > 0:
        weights = {k: v / total for k, v in weights.items()}
    return weights


def _round_allocations(amounts: Dict[str, float]) -> Dict[str, int]:
    floors = {k: int(np.floor(v)) for k, v in amounts.items()}
    remainder = int(round(sum(amounts.values()) - sum(floors.values())))
    if remainder <= 0:
        return floors
    fractions = sorted(
        ((k, amounts[k] - floors[k]) for k in amounts),
        key=lambda x: x[1],
        reverse=True,
    )
    for i in range(min(remainder, len(fractions))):
        k, _ = fractions[i]
        floors[k] += 1
    return floors


def _apply_weight_caps(weights: Dict[str, float], max_weight: float) -> Dict[str, float]:
    if not weights or max_weight <= 0:
        return weights
    weights = weights.copy()
    for _ in range(10):
        over = {k: v for k, v in weights.items() if v > max_weight}
        if not over:
            break
        excess = sum(v - max_weight for v in over.values())
        for k in over:
            weights[k] = max_weight
        remaining = {k: v for k, v in weights.items() if k not in over}
        if not remaining or excess <= 0:
            break
        total_remaining = sum(remaining.values())
        for k in remaining:
            weights[k] += excess * (remaining[k] / total_remaining)
    total = sum(weights.values())
    if total > 0:
        weights = {k: v / total for k, v in weights.items()}
    return weights


def _returns_for_isins(
    isins: List[str],
    start: pd.Timestamp,
    end: pd.Timestamp,
    euro_rate: float,
    min_points: int = 60,
) -> Tuple[pd.DataFrame, Dict[str, str]]:
    series_map: Dict[str, pd.Series] = {}
    status: Dict[str, str] = {}
    for isin in isins:
        df, _, _ = get_price_series(isin, None, euro_rate)
        if df.empty:
            status[isin] = "insufficient"
            continue
        s = df["Close"].astype(float)
        s = s[(s.index >= start) & (s.index <= end)]
        if s.dropna().shape[0] < min_points:
            status[isin] = "insufficient"
            continue
        series_map[isin] = s
        status[isin] = "ok"
    if not series_map:
        return pd.DataFrame(), status
    prices = pd.DataFrame(series_map).ffill().dropna(how="any")
    if prices.empty or prices.shape[0] < min_points:
        return pd.DataFrame(), status
    returns = prices.pct_change().dropna(how="any")
    return returns, status


def _annualized_stats(returns: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    if returns.empty:
        return pd.Series(dtype=float), pd.Series(dtype=float)
    ann_return = returns.mean() * 252.0
    ann_vol = returns.std() * np.sqrt(252.0)
    return ann_return, ann_vol


def _avg_correlation(corr: pd.DataFrame) -> float:
    if corr.empty or corr.shape[0] < 2:
        return np.nan
    vals = corr.values[np.triu_indices_from(corr, 1)]
    return float(np.nanmean(vals)) if vals.size else np.nan


def _avg_offdiag_corr(corr: pd.DataFrame) -> float:
    if corr.empty or corr.shape[0] < 2:
        return np.nan
    vals = corr.values[np.triu_indices_from(corr, 1)]
    return float(np.nanmean(vals)) if vals.size else np.nan


def _select_min_corr_subset(
    candidates: List[str],
    returns: pd.DataFrame,
    k: int,
    anchor: Optional[str] = None,
) -> List[str]:
    if k <= 0 or not candidates:
        return []
    if anchor and anchor not in candidates:
        return []
    if k >= len(candidates):
        return candidates
    corr = returns.corr()
    if corr.empty:
        return candidates[:k]
    if anchor:
        pool = [c for c in candidates if c != anchor]
        best_combo = [anchor]
        best_score = None
        max_checks = 2000
        for idx, combo in enumerate(itertools.combinations(pool, k - 1)):
            if idx >= max_checks:
                break
            subset = [anchor, *combo]
            score = _avg_offdiag_corr(corr.loc[subset, subset])
            if best_score is None or score < best_score:
                best_score = score
                best_combo = subset
        if best_score is None:
            best_combo = [anchor] + pool[: k - 1]
        return best_combo
    best_combo: List[str] = []
    best_score = None
    max_checks = 2000
    for idx, combo in enumerate(itertools.combinations(candidates, k)):
        if idx >= max_checks:
            break
        subset = list(combo)
        score = _avg_offdiag_corr(corr.loc[subset, subset])
        if best_score is None or score < best_score:
            best_score = score
            best_combo = subset
    if not best_combo:
        best_combo = candidates[:k]
    return best_combo


def _avg_offdiag_corr(corr: pd.DataFrame) -> float:
    if corr.empty or corr.shape[0] < 2:
        return np.nan
    vals = corr.values[np.triu_indices_from(corr, 1)]
    return float(np.nanmean(vals)) if vals.size else np.nan


def _select_min_corr_subset(
    candidates: List[str],
    returns: pd.DataFrame,
    k: int,
    anchor: Optional[str] = None,
) -> List[str]:
    if k <= 0 or not candidates:
        return []
    if anchor and anchor not in candidates:
        return []
    if k >= len(candidates):
        return candidates
    corr = returns.corr()
    if corr.empty:
        return candidates[:k]
    if anchor:
        pool = [c for c in candidates if c != anchor]
        best_combo = [anchor]
        best_score = None
        max_checks = 2000
        combos = itertools.combinations(pool, k - 1)
        for idx, combo in enumerate(combos):
            if idx >= max_checks:
                break
            subset = [anchor, *combo]
            score = _avg_offdiag_corr(corr.loc[subset, subset])
            if best_score is None or score < best_score:
                best_score = score
                best_combo = subset
        if best_score is None:
            best_combo = [anchor] + pool[: k - 1]
        return best_combo
    max_checks = 2000
    best_combo: List[str] = []
    best_score = None
    for idx, combo in enumerate(itertools.combinations(candidates, k)):
        if idx >= max_checks:
            break
        subset = list(combo)
        score = _avg_offdiag_corr(corr.loc[subset, subset])
        if best_score is None or score < best_score:
            best_score = score
            best_combo = subset
    if not best_combo:
        best_combo = candidates[:k]
    return best_combo


def _greedy_select(
    candidates: List[str],
    returns: pd.DataFrame,
    target_count: int,
    forced: Optional[str] = None,
    corr_penalty: float = 0.35,
) -> List[str]:
    selected: List[str] = []
    if forced and forced in candidates:
        selected.append(forced)
    remaining = [c for c in candidates if c not in selected]
    if returns.empty or target_count <= 0:
        return selected
    ann_return, ann_vol = _annualized_stats(returns[remaining + selected] if remaining else returns)
    sharpe = ann_return / ann_vol.replace(0, np.nan)
    corr = returns.corr() if not returns.empty else pd.DataFrame()

    while len(selected) < target_count and remaining:
        best = None
        best_score = None
        for cand in remaining:
            base_sharpe = float(sharpe.get(cand, np.nan))
            if np.isnan(base_sharpe):
                base_sharpe = -1e9
            if selected and not corr.empty:
                avg_corr = float(corr.loc[cand, selected].mean())
            else:
                avg_corr = 0.0
            score = base_sharpe - corr_penalty * avg_corr
            if best_score is None or score > best_score:
                best_score = score
                best = cand
        if best is None:
            break
        selected.append(best)
        remaining = [c for c in remaining if c != best]
    return selected


def _optimize_uc_weights(
    returns: pd.DataFrame,
    objective: str,
    min_weight: float,
    max_weight: float,
    target_vol: Optional[float],
    target_return: Optional[float],
) -> Dict[str, float]:
    if returns.empty:
        return {}
    n_assets = returns.shape[1]
    if n_assets == 0:
        return {}
    bounds = (min_weight, max_weight)
    try:
        if not PYPFOPT_AVAILABLE:
            raise RuntimeError(PYPFOPT_ERROR)
        mu = expected_returns.mean_historical_return(returns, frequency=252)
        cov = risk_models.sample_cov(returns, frequency=252)
        ef = EfficientFrontier(mu, cov, weight_bounds=bounds)
        if objective == "max_sharpe":
            weights = ef.max_sharpe(risk_free_rate=0.0)
        elif objective == "target_vol" and target_vol is not None:
            weights = ef.efficient_risk(target_vol)
        elif objective == "target_return" and target_return is not None:
            weights = ef.efficient_return(target_return)
        else:
            weights = ef.max_sharpe(risk_free_rate=0.0)
        cleaned = ef.clean_weights()
        total = sum(cleaned.values())
        if total > 0:
            return {k: v / total for k, v in cleaned.items()}
    except Exception:
        pass
    equal_weight = 1.0 / n_assets
    return {col: equal_weight for col in returns.columns}


def _select_min_corr_combo(
    returns: pd.DataFrame,
    k: int,
    anchor: Optional[str] = None,
) -> List[str]:
    if returns.empty:
        return []
    cols = list(returns.columns)
    if anchor and anchor not in cols:
        return []
    if k <= 0:
        return []
    if k > len(cols):
        return cols
    corr = returns.corr()
    best_combo: List[str] = []
    best_score = None
    if anchor:
        others = [c for c in cols if c != anchor]
        for combo in itertools.combinations(others, k - 1):
            candidate = [anchor, *combo]
            sub = corr.loc[candidate, candidate].to_numpy()
            triu = sub[np.triu_indices_from(sub, 1)]
            score = float(triu.mean()) if triu.size else 1.0
            if best_score is None or score < best_score:
                best_score = score
                best_combo = candidate
    else:
        for combo in itertools.combinations(cols, k):
            sub = corr.loc[list(combo), list(combo)].to_numpy()
            triu = sub[np.triu_indices_from(sub, 1)]
            score = float(triu.mean()) if triu.size else 1.0
            if best_score is None or score < best_score:
                best_score = score
                best_combo = list(combo)
    return best_combo


def _compute_drawdown(returns: pd.Series) -> Optional[float]:
    if returns.empty:
        return None
    cum = (1.0 + returns).cumprod()
    running_max = cum.cummax()
    dd = cum / running_max - 1.0
    return float(dd.min())


def _fund_name(isin: str) -> str:
    return FUND_NAME_MAP.get(isin, isin)


def _safe_fund_label(name: str, isin: str) -> str:
    cleaned = str(name or "").strip()
    if cleaned:
        return f"{cleaned} ({isin})"
    if isin:
        res = eodhd_search(isin)
        if res:
            maybe = res[0].get("Name") or res[0].get("name")
            if maybe:
                return f"{maybe} ({isin})"
    return f"{isin} ({isin})"


def render_portfolio_builder():
    st.title("Creer le portefeuille parfait")

    st.session_state.setdefault("PP_RUN", False)
    st.session_state.setdefault("PP_PARAMS_HASH", "")
    st.session_state.setdefault("PP_OPT_START_DATE", (TODAY - pd.DateOffset(years=3)).date())
    st.session_state.setdefault("PP_OPT_END_DATE", TODAY.date())

    profile_map = {
        "Prudent": 50,
        "Equilibre": 30,
        "Dynamique": 20,
        "Agressif": 10,
    }

    with st.sidebar:
        st.header("Parametres cles")
        profile = st.selectbox("Profil", list(profile_map.keys()), index=1)
        euro_pct = profile_map[profile]
        st.caption(f"Fonds en euros obligatoire : {euro_pct}% du portefeuille.")

        euro_rate = st.number_input(
            "Rendement annuel du fonds en euros (%)",
            min_value=0.0,
            max_value=10.0,
            value=2.0,
            step=0.10,
            key="PP_EURO_RATE",
        )

        total_budget = st.number_input(
            "Budget total (EUR)",
            min_value=0,
            max_value=10_000_000,
            value=100_000,
            step=10,
            key="PP_TOTAL_BUDGET",
        )

        opt_window_mode = st.radio(
            "Fenetre d'analyse",
            ["1 an", "3 ans", "5 ans", "10 ans", "Dates personnalisees"],
            horizontal=False,
            key="PP_WINDOW_MODE",
        )

        if opt_window_mode == "Dates personnalisees":
            opt_start_date = st.date_input(
                "Date de debut",
                value=st.session_state.get("PP_OPT_START_DATE", (TODAY - pd.DateOffset(years=3)).date()),
                key="PP_OPT_START_DATE",
            )
            opt_end_date = st.date_input(
                "Date de fin",
                value=st.session_state.get("PP_OPT_END_DATE", TODAY.date()),
                key="PP_OPT_END_DATE",
            )
            opt_start = pd.Timestamp(opt_start_date)
            opt_end = pd.Timestamp(opt_end_date)
        else:
            years_map = {"1 an": 1, "3 ans": 3, "5 ans": 5, "10 ans": 10}
            opt_years = years_map[opt_window_mode]
            opt_start = TODAY - pd.DateOffset(years=opt_years)
            opt_end = TODAY

    st.markdown("### Parametres UC")
    c1, c2 = st.columns(2)
    with c1:
        action_count = int(
            st.number_input(
                "Nombre de fonds actions UC",
                min_value=1,
                max_value=6,
                value=3,
                step=1,
                key="PP_ACTION_COUNT",
            )
        )
    with c2:
        bond_count = int(
            st.number_input(
                "Nombre de fonds obligataires UC",
                min_value=0,
                max_value=4,
                value=1,
                step=1,
                key="PP_BOND_COUNT",
            )
        )

    objective_choice = st.selectbox(
        "Objectif",
        [
            "Maximiser Sharpe",
            "Minimiser volatilite",
            "Maximiser rendement annualise",
            "Meilleur compromis (Sharpe + diversification)",
            "Diversification maximale (decorrelation)",
            "Risk parity",
        ],
        key="PP_OBJECTIVE",
    )

    force_fund = st.checkbox("Forcer un fonds action (ancre)", value=False, key="PP_FORCE_ANCHOR")
    forced_isin = None
    if force_fund:
        action_labels = [_safe_fund_label(name, isin) for name, isin in RECO_FUNDS_CORE]
        action_lookup = {label: isin for label, (_, isin) in zip(action_labels, RECO_FUNDS_CORE)}
        forced_choice = st.selectbox("Fonds action impose", action_labels, key="PP_ANCHOR_ISIN")
        forced_isin = action_lookup.get(forced_choice)

    params = (
        profile,
        float(euro_rate),
        int(total_budget),
        opt_window_mode,
        str(opt_start.date()),
        str(opt_end.date()),
        int(action_count),
        int(bond_count),
        objective_choice,
        bool(force_fund),
        forced_isin,
    )
    current_hash = params_hash(params)

    if st.session_state.get("PP_RUN") and st.session_state.get("PP_PARAMS_HASH") != current_hash:
        st.session_state["PP_RUN"] = False
        st.info("Parametres modifies. Cliquez de nouveau sur '✅ Creer le portefeuille parfait'.")

    run_clicked = st.button("✅ Creer le portefeuille parfait", type="primary")
    if run_clicked:
        st.session_state["PP_RUN"] = True
        st.session_state["PP_PARAMS_HASH"] = current_hash

    if not st.session_state.get("PP_RUN"):
        st.info("Renseignez tous les parametres puis cliquez sur '✅ Creer le portefeuille parfait'.")
        return

    if opt_start > opt_end:
        st.warning("La date de debut doit etre anterieure a la date de fin.")
        return

    try:
        actions_universe = [isin for _, isin in RECO_FUNDS_CORE]
        bonds_universe = [isin for _, isin in RECO_FUNDS_DEF if isin != "EUROFUND"]

        all_candidates = sorted(set(actions_universe + bonds_universe))
        returns_all, status_all = _returns_for_isins(all_candidates, opt_start, opt_end, euro_rate=float(euro_rate))

        insufficient = [isin for isin, status in status_all.items() if status != "ok"]
        valid_actions = [isin for isin in actions_universe if status_all.get(isin) == "ok"]
        valid_bonds = [isin for isin in bonds_universe if status_all.get(isin) == "ok"]

        if insufficient:
            st.warning("Certains fonds ont ete exclus (historique insuffisant sur la fenetre).")

        if forced_isin and forced_isin not in valid_actions:
            st.warning("Le fonds action ancre est indisponible sur la periode et a ete ignore.")
            forced_isin = None

        if action_count > len(valid_actions):
            st.warning("Nombre de fonds actions reduit automatiquement (historique insuffisant).")
            action_count = len(valid_actions)
        if bond_count > len(valid_bonds):
            st.warning("Nombre de fonds obligataires reduit automatiquement (historique insuffisant).")
            bond_count = len(valid_bonds)

        if action_count + bond_count == 0:
            st.info("Aucun fonds UC valide sur la fenetre choisie.")
            return

        if returns_all.empty:
            st.info("Historique insuffisant pour calculer les rendements sur la fenetre choisie.")
            return

        def _zscore(s: pd.Series) -> pd.Series:
            if s.empty:
                return s
            std = float(s.std())
            if std == 0 or np.isnan(std):
                return pd.Series(0.0, index=s.index)
            return (s - s.mean()) / std

        def _stats_for(candidates: List[str], ret_df: pd.DataFrame) -> Tuple[pd.Series, pd.Series, pd.Series, pd.DataFrame, pd.Series]:
            if not candidates or ret_df.empty:
                return pd.Series(dtype=float), pd.Series(dtype=float), pd.Series(dtype=float), pd.DataFrame(), pd.Series(dtype=float)
            ann_log_return = ret_df[candidates].mean() * 252.0
            ann_return = np.exp(ann_log_return) - 1.0
            ann_vol = ret_df[candidates].std() * np.sqrt(252.0)
            sharpe = (ann_log_return / ann_vol.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)
            corr = ret_df[candidates].corr()
            avg_corr = corr.apply(lambda row: row.drop(labels=row.name, errors="ignore").mean(), axis=1).fillna(0.0) if not corr.empty else pd.Series(0.0, index=candidates)
            return ann_return, ann_vol, sharpe, corr, avg_corr

        def _greedy_min_corr(candidates: List[str], corr: pd.DataFrame, k: int, seed: Optional[List[str]] = None) -> List[str]:
            if k <= 0 or not candidates:
                return []
            selected = [x for x in (seed or []) if x in candidates]
            remaining = [c for c in candidates if c not in selected]
            while len(selected) < k and remaining:
                best, best_score = None, None
                for cand in remaining:
                    score = float(corr.loc[cand, selected].mean()) if selected and not corr.empty else 0.0
                    if best_score is None or score < best_score:
                        best, best_score = cand, score
                if best is None:
                    break
                selected.append(best)
                remaining = [c for c in remaining if c != best]
            return selected

        def _select_by_objective(candidates: List[str], ret_df: pd.DataFrame, k: int, objective: str, forced: Optional[str] = None) -> List[str]:
            if k <= 0 or not candidates:
                return []
            if ret_df.empty:
                base = candidates[:k]
                if forced and forced in candidates and forced not in base:
                    base = [forced] + [x for x in base if x != forced]
                    base = base[:k]
                return base

            ann_ret, ann_vol, sharpe, corr, avg_corr = _stats_for(candidates, ret_df)
            sharpe_rank = sharpe.fillna(-1e9).sort_values(ascending=False)

            if objective == "Maximiser rendement annualise":
                selected = ann_ret.sort_values(ascending=False).index.tolist()[:k]
            elif objective == "Minimiser volatilite":
                selected = ann_vol.sort_values(ascending=True).index.tolist()[:k]
            elif objective == "Maximiser Sharpe":
                selected = sharpe_rank.index.tolist()[:k]
            elif objective in ("Diversification maximale (decorrelation)", "Risk parity"):
                seed = sharpe_rank.index.tolist()[:1]
                selected = _greedy_min_corr(candidates, corr, k, seed=seed)
            elif objective == "Meilleur compromis (Sharpe + diversification)":
                score = _zscore(sharpe.fillna(0.0)) + _zscore(1.0 - avg_corr)
                selected = score.sort_values(ascending=False).index.tolist()[:k]
            else:
                selected = sharpe_rank.index.tolist()[:k]

            if forced and forced in candidates:
                if forced not in selected:
                    if objective in ("Diversification maximale (decorrelation)", "Risk parity"):
                        selected = _greedy_min_corr(candidates, corr, k, seed=[forced])
                    else:
                        rest = [x for x in selected if x != forced]
                        selected = [forced] + rest
                        selected = selected[:k]
                else:
                    selected = [forced] + [x for x in selected if x != forced]
            return selected[:k]

        action_returns = returns_all[valid_actions] if valid_actions else pd.DataFrame()
        bond_returns = returns_all[valid_bonds] if valid_bonds else pd.DataFrame()

        selected_actions = _select_by_objective(valid_actions, action_returns, int(action_count), objective_choice, forced=forced_isin)
        selected_bonds = _select_by_objective(valid_bonds, bond_returns, int(bond_count), objective_choice, forced=None)

        selected_isins = [isin for isin in (selected_actions + selected_bonds) if isin]
        if not selected_isins:
            st.info("Aucun fonds selectionne apres filtrage.")
            return

        returns_selected = returns_all[selected_isins].dropna(how="any")
        if returns_selected.empty:
            st.info("Historique insuffisant apres selection des UC.")
            return

        uc_total = max(0.0, 1.0 - (float(euro_pct) / 100.0))
        if uc_total <= 0.0:
            st.info("Part UC nulle avec le profil choisi.")
            return

        cap_uc_final = 0.25
        uc_max_bound = min(cap_uc_final / uc_total, 1.0)

        def _risk_parity_weights(ret_df: pd.DataFrame) -> Dict[str, float]:
            cov = ret_df.cov()
            if cov.empty:
                return {}
            vols = np.sqrt(np.diag(cov.values))
            inv = np.where(vols > 0, 1.0 / vols, 0.0)
            if float(inv.sum()) <= 0:
                return {c: 1.0 / len(ret_df.columns) for c in ret_df.columns}
            w = inv / inv.sum()
            return {c: float(w[i]) for i, c in enumerate(ret_df.columns)}

        weights_uc_raw: Dict[str, float] = {}
        if PYPFOPT_AVAILABLE:
            try:
                mu = expected_returns.mean_historical_return(returns_selected, returns_data=True, frequency=252)
                cov = risk_models.sample_cov(returns_selected, returns_data=True, frequency=252)
                ef = EfficientFrontier(mu, cov, weight_bounds=(0.0, uc_max_bound))

                if objective_choice == "Maximiser Sharpe":
                    ef.max_sharpe(risk_free_rate=0.0)
                    weights_uc_raw = ef.clean_weights()
                elif objective_choice == "Minimiser volatilite":
                    ef.min_volatility()
                    weights_uc_raw = ef.clean_weights()
                elif objective_choice == "Maximiser rendement annualise":
                    if not mu.empty:
                        target = float(mu.max()) - 1e-6
                        try:
                            ef.efficient_return(target_return=target)
                        except Exception:
                            ef.max_sharpe(risk_free_rate=0.0)
                    else:
                        ef.max_sharpe(risk_free_rate=0.0)
                    weights_uc_raw = ef.clean_weights()
                elif objective_choice == "Risk parity":
                    weights_uc_raw = _risk_parity_weights(returns_selected)
                else:
                    ef.min_volatility() if objective_choice == "Diversification maximale (decorrelation)" else ef.max_sharpe(risk_free_rate=0.0)
                    weights_uc_raw = ef.clean_weights()
            except Exception:
                weights_uc_raw = {}

        if not weights_uc_raw:
            if objective_choice == "Risk parity":
                weights_uc_raw = _risk_parity_weights(returns_selected)
            elif objective_choice == "Minimiser volatilite":
                vol = returns_selected.std() * np.sqrt(252.0)
                score = (1.0 / vol.replace(0, np.nan)).fillna(0.0)
                weights_uc_raw = (score / score.sum()).to_dict() if float(score.sum()) > 0 else {}
            elif objective_choice == "Maximiser rendement annualise":
                ann_ret = np.exp(returns_selected.mean() * 252.0) - 1.0
                score = ann_ret.clip(lower=0.0)
                weights_uc_raw = (score / score.sum()).to_dict() if float(score.sum()) > 0 else {}
            elif objective_choice == "Meilleur compromis (Sharpe + diversification)":
                ann_log = returns_selected.mean() * 252.0
                vol = returns_selected.std() * np.sqrt(252.0)
                sharpe = (ann_log / vol.replace(0, np.nan)).fillna(0.0)
                corr = returns_selected.corr()
                avg_corr = corr.apply(lambda row: row.drop(labels=row.name, errors="ignore").mean(), axis=1).fillna(0.0) if not corr.empty else pd.Series(0.0, index=returns_selected.columns)
                score = _zscore(sharpe) + _zscore(1.0 - avg_corr)
                score = score.clip(lower=0.0)
                weights_uc_raw = (score / score.sum()).to_dict() if float(score.sum()) > 0 else {}
            elif objective_choice == "Diversification maximale (decorrelation)":
                corr = returns_selected.corr()
                avg_corr = corr.apply(lambda row: row.drop(labels=row.name, errors="ignore").mean(), axis=1).fillna(0.0) if not corr.empty else pd.Series(0.0, index=returns_selected.columns)
                score = (1.0 - avg_corr).clip(lower=0.0)
                weights_uc_raw = (score / score.sum()).to_dict() if float(score.sum()) > 0 else {}
            else:
                ann_log = returns_selected.mean() * 252.0
                vol = returns_selected.std() * np.sqrt(252.0)
                score = (ann_log / vol.replace(0, np.nan)).fillna(0.0).clip(lower=0.0)
                weights_uc_raw = (score / score.sum()).to_dict() if float(score.sum()) > 0 else {}

        if not weights_uc_raw:
            weights_uc_raw = {isin: 1.0 / len(selected_isins) for isin in selected_isins}

        weights_uc_raw = {k: float(v) for k, v in weights_uc_raw.items() if k in selected_isins}
        if not weights_uc_raw:
            weights_uc_raw = {isin: 1.0 / len(selected_isins) for isin in selected_isins}

        weights_uc_raw = _apply_weight_caps(weights_uc_raw, uc_max_bound)
        total_uc_raw = float(sum(weights_uc_raw.values()))
        if total_uc_raw <= 0:
            weights_uc_raw = {isin: 1.0 / len(selected_isins) for isin in selected_isins}
            weights_uc_raw = _apply_weight_caps(weights_uc_raw, uc_max_bound)
            total_uc_raw = float(sum(weights_uc_raw.values()))
        if total_uc_raw > 0:
            weights_uc_raw = {k: v / total_uc_raw for k, v in weights_uc_raw.items()}

        weights_final = {"EUROFUND": float(euro_pct) / 100.0}
        for isin in selected_isins:
            weights_final[isin] = float(weights_uc_raw.get(isin, 0.0)) * uc_total

        # Security clamp + exact renormalization
        weights_final = {k: max(0.0, float(v)) for k, v in weights_final.items()}
        total_weight = float(sum(weights_final.values()))
        if total_weight > 0:
            weights_final = {k: v / total_weight for k, v in weights_final.items()}

        def _round_allocations_to_step(amounts: Dict[str, float], step: int, total: int) -> Dict[str, int]:
            if step <= 0:
                step = 1
            rounded = {k: int(np.floor(max(0.0, v) / step)) * step for k, v in amounts.items()}
            remaining = int(total - sum(rounded.values()))
            if remaining > 0 and rounded:
                frac_rank = sorted(
                    ((k, (max(0.0, amounts[k]) / step) - np.floor(max(0.0, amounts[k]) / step)) for k in rounded.keys()),
                    key=lambda x: x[1],
                    reverse=True,
                )
                idx = 0
                while remaining >= step and frac_rank:
                    k = frac_rank[idx % len(frac_rank)][0]
                    rounded[k] += step
                    remaining -= step
                    idx += 1
                if remaining != 0:
                    last_key = list(rounded.keys())[-1]
                    rounded[last_key] = max(0, rounded[last_key] + remaining)
            elif remaining < 0 and rounded:
                last_key = list(rounded.keys())[-1]
                rounded[last_key] = max(0, rounded[last_key] + remaining)

            delta = int(total - sum(rounded.values()))
            if rounded and delta != 0:
                last_key = list(rounded.keys())[-1]
                rounded[last_key] = max(0, rounded[last_key] + delta)
            return rounded

        amounts_raw = {k: float(total_budget) * float(w) for k, w in weights_final.items()}
        amounts = _round_allocations_to_step(amounts_raw, step=10, total=int(total_budget))

        rows = [
            {
                "Support": "Fonds en euros (EUROFUND)",
                "Categorie": "EUROFUND",
                "%": weights_final.get("EUROFUND", 0.0) * 100.0,
                "Montant EUR": amounts.get("EUROFUND", 0),
            }
        ]
        for isin in selected_isins:
            rows.append(
                {
                    "Support": _fund_name(isin),
                    "Categorie": "Actions UC" if isin in selected_actions else "Obligataires UC",
                    "%": weights_final.get(isin, 0.0) * 100.0,
                    "Montant EUR": amounts.get(isin, 0),
                }
            )
        df_alloc = pd.DataFrame(rows)

        st.markdown("**Allocation finale**")
        st.dataframe(
            df_alloc.style.format({"%": "{:,.2f}%".format, "Montant EUR": to_eur}),
            use_container_width=True,
            hide_index=True,
        )

        if MATPLOTLIB_AVAILABLE:
            fig, ax = plt.subplots(figsize=(5.0, 3.2))
            ax.pie(df_alloc["Montant EUR"], labels=df_alloc["Support"], autopct="%1.1f%%")
            ax.set_title("Repartition finale")
            st.pyplot(fig)
            plt.close(fig)
        else:
            st.warning(f"Graphique indisponible ({MATPLOTLIB_ERROR}).")

        if len(selected_isins) >= 2:
            st.markdown("**Heatmap correlation UC**")
            corr_uc = returns_selected.corr().copy()
            corr_uc["Ligne1"] = corr_uc.index
            heat_df = corr_uc.melt(id_vars="Ligne1", var_name="Ligne2", value_name="corr")
            heat = (
                alt.Chart(heat_df)
                .mark_rect()
                .encode(
                    x=alt.X("Ligne1:O", sort=None, title=""),
                    y=alt.Y("Ligne2:O", sort=None, title=""),
                    color=alt.Color("corr:Q", scale=alt.Scale(domain=[-1, 1])),
                    tooltip=[
                        alt.Tooltip("Ligne1:N", title="Ligne 1"),
                        alt.Tooltip("Ligne2:N", title="Ligne 2"),
                        alt.Tooltip("corr:Q", title="Correlation", format=".2f"),
                    ],
                )
                .properties(height=260)
            )
            st.altair_chart(heat, use_container_width=True)

        uc_weights_norm = pd.Series({k: weights_uc_raw.get(k, 0.0) for k in selected_isins}, index=selected_isins)
        uc_weights_norm = uc_weights_norm / uc_weights_norm.sum() if float(uc_weights_norm.sum()) > 0 else pd.Series(1.0 / len(selected_isins), index=selected_isins)

        port_log_ret = returns_selected[selected_isins].dot(uc_weights_norm.values)
        ann_log = float(port_log_ret.mean() * 252.0)
        ann_ret_uc = float(np.exp(ann_log) - 1.0)
        ann_vol_uc = float(port_log_ret.std() * np.sqrt(252.0))
        sharpe_uc = ann_log / ann_vol_uc if ann_vol_uc > 0 else np.nan

        euro_df, _, _ = get_price_series("EUROFUND", None, float(euro_rate))
        euro_df = euro_df.loc[(euro_df.index >= opt_start) & (euro_df.index <= opt_end)]
        if euro_df.empty:
            euro_total_ret = 0.0
        else:
            euro_total_ret = float(euro_df["Close"].iloc[-1] / euro_df["Close"].iloc[0] - 1.0)

        uc_path = np.exp(port_log_ret.cumsum())
        uc_total_ret = float(uc_path.iloc[-1] - 1.0) if not uc_path.empty else 0.0

        total_ret = (float(euro_pct) / 100.0) * euro_total_ret + uc_total * uc_total_ret

        k1, k2, k3, k4 = st.columns(4)
        with k1:
            st.metric("Rendement annualise UC", fmt_pct_fr(ann_ret_uc * 100.0))
        with k2:
            st.metric("Volatilite annualisee UC", fmt_pct_fr(ann_vol_uc * 100.0))
        with k3:
            st.metric("Sharpe UC", f"{sharpe_uc:.2f}" if sharpe_uc == sharpe_uc else "-")
        with k4:
            st.metric("Rendement total (EUROFUND+UC)", fmt_pct_fr(total_ret * 100.0))

        st.markdown(
            """
- Allocation calibree sur la fenetre d'analyse et l'objectif choisi.
- EUROFUND est traite uniquement via son taux annuel parametre.
- Les UC respectent un cap strict de 25% du portefeuille final par fonds.
- En mode ancre, le fonds impose est conserve dans la poche actions.
- En cas de donnees insuffisantes, la selection est reduite automatiquement.
"""
        )

        if insufficient:
            st.caption("Fonds exclus : " + ", ".join(insufficient))
        st.caption(f"Fenetre utilisee : {fmt_date(opt_start)} a {fmt_date(opt_end)}")

    except Exception as e:
        st.error("Une erreur est survenue dans le builder.")
        st.exception(e)

def render_app(run_page_config: bool = True):
    # ------------------------------------------------------------
    # Layout principal
    # ------------------------------------------------------------
    if run_page_config:
        st.set_page_config(page_title=APP_TITLE, layout="wide")
    st.title(APP_TITLE)
    st.info(f"App chargée, statut {st.session_state.get('APP_STATUS', 'OK')}")
    # Init state
    st.session_state.setdefault("A_lines", [])
    st.session_state.setdefault("B_lines", [])
    st.session_state.setdefault("FEE_A", 3.0)
    st.session_state.setdefault("FEE_B", 2.0)
    st.session_state.setdefault("M_A", 0.0)
    st.session_state.setdefault("M_B", 0.0)
    st.session_state.setdefault("ONE_A", 0.0)
    st.session_state.setdefault("ONE_B", 0.0)
    st.session_state.setdefault("ONE_A_DATE", pd.Timestamp("2024-07-01").date())
    st.session_state.setdefault("ONE_B_DATE", pd.Timestamp("2024-07-01").date())
    st.session_state.setdefault("ALLOC_MODE", "equal")
    st.session_state.setdefault("DATE_WARNINGS", [])
    st.session_state.setdefault("INIT_A_DATE", pd.Timestamp("2024-01-02").date())
    st.session_state.setdefault("INIT_B_DATE", pd.Timestamp("2024-01-02").date())
    st.session_state.setdefault("EURO_RATE_A", 2.0)
    st.session_state.setdefault("EURO_RATE_B", 2.5)

    # -------------------------------------------------------------------
    # Sidebar : paramètres globaux
    # -------------------------------------------------------------------
    with st.sidebar:
        # Fonds en euros — Taux annuel (par portefeuille)
        st.header("Fonds en euros — Taux annuel")

        EURO_RATE_A = st.number_input(
            "Portefeuille 1 (Client) — taux annuel (%)",
            0.0,
            10.0,
            st.session_state.get("EURO_RATE_A", 2.0),
            0.10,
            key="EURO_RATE_A",
        )

        EURO_RATE_B = st.number_input(
            "Portefeuille 2 (Valority) — taux annuel (%)",
            0.0,
            10.0,
            st.session_state.get("EURO_RATE_B", 2.5),
            0.10,
            key="EURO_RATE_B",
        )

        st.caption(
            "Le taux est appliqué annuellement sur la part investie en fonds euros (EUROFUND), "
            f"net des frais UC : {ANNUAL_FEE_EURO_PCT:.1f}%/an pour le fonds euros "
            f"et {ANNUAL_FEE_UC_PCT:.1f}%/an pour les unités de compte."
        )

        # Frais d’entrée
        st.header("Frais d’entrée (%)")

        FEE_A = st.number_input(
            "Frais d’entrée — Portefeuille 1 (Client)",
            0.0,
            10.0,
            st.session_state.get("FEE_A", 3.0),
            0.10,
            key="FEE_A",
        )

        FEE_B = st.number_input(
            "Frais d’entrée — Portefeuille 2 (Valority)",
            0.0,
            10.0,
            st.session_state.get("FEE_B", 2.0),
            0.10,
            key="FEE_B",
        )

        st.caption("Les frais s’appliquent sur chaque investissement (initial, mensuel, ponctuel).")

        # Date du versement initial (centralisée)
        st.header("Date du versement initial")

        st.date_input(
            "Portefeuille 1 (Client) — date d’investissement initiale",
            value=st.session_state.get("INIT_A_DATE", pd.Timestamp("2024-01-02").date()),
            key="INIT_A_DATE",
        )

        st.date_input(
            "Portefeuille 2 (Valority) — date d’investissement initiale",
            value=st.session_state.get("INIT_B_DATE", pd.Timestamp("2024-01-02").date()),
            key="INIT_B_DATE",
        )

        # Paramètres de versement
        st.header("Paramètres de versement")

        with st.expander("Portefeuille 1 — Client"):
            M_A = st.number_input(
                "Mensuel brut (€)",
                0.0,
                1_000_000.0,
                st.session_state.get("M_A", 0.0),
                100.0,
                key="M_A",
            )
            ONE_A = st.number_input(
                "Ponctuel brut (€)",
                0.0,
                1_000_000.0,
                st.session_state.get("ONE_A", 0.0),
                100.0,
                key="ONE_A",
            )
            ONE_A_DATE = st.date_input(
                "Date du ponctuel",
                value=st.session_state.get("ONE_A_DATE", pd.Timestamp("2024-07-01").date()),
                key="ONE_A_DATE",
            )

        with st.expander("Portefeuille 2 — Valority"):
            M_B = st.number_input(
                "Mensuel brut (€)",
                0.0,
                1_000_000.0,
                st.session_state.get("M_B", 0.0),
                100.0,
                key="M_B",
            )
            ONE_B = st.number_input(
                "Ponctuel brut (€)",
                0.0,
                1_000_000.0,
                st.session_state.get("ONE_B", 0.0),
                100.0,
                key="ONE_B",
            )
            ONE_B_DATE = st.date_input(
                "Date du ponctuel",
                value=st.session_state.get("ONE_B_DATE", pd.Timestamp("2024-07-01").date()),
                key="ONE_B_DATE",
            )

        # Règle d’affectation
        st.header("Règle d’affectation des versements")

        current_code = st.session_state.get("ALLOC_MODE", "equal")
        inv_labels = {v: k for k, v in ALLOC_LABELS.items()}
        current_label = inv_labels.get(current_code, "Répartition égale")

        mode_label = st.selectbox(
            "Mode",
            list(ALLOC_LABELS.keys()),
            index=list(ALLOC_LABELS.keys()).index(current_label),
            help="Répartition des versements entre les lignes.",
        )

        st.session_state["ALLOC_MODE"] = ALLOC_LABELS[mode_label]

        st.divider()
        st.header("Mode d’analyse")

        mode_ui = st.radio(
            "Choix",
            ["Comparer Client vs Valority", "Analyser uniquement Valority", "Analyser uniquement Client"],
            index=0,
            key="MODE_ANALYSE_UI",
        )

        if "Comparer" in mode_ui:
            st.session_state["MODE_ANALYSE"] = "compare"
        elif "Valority" in mode_ui:
            st.session_state["MODE_ANALYSE"] = "valority"
        else:
            st.session_state["MODE_ANALYSE"] = "client"

        st.divider()
        debug_mode = st.checkbox("Mode debug", value=False)
        if debug_mode:
            st.subheader("Debug")
            st.caption("Versions & état")
            st.code(
                f"Python: {sys.version.split()[0]}\n"
                f"Streamlit: {st.__version__}\n"
                f"Pandas: {pd.__version__}"
            )
            st.caption("Modules")
            st.code(
                f"Matplotlib: {MATPLOTLIB_AVAILABLE} ({MATPLOTLIB_ERROR})\n"
                f"Reportlab: {REPORTLAB_AVAILABLE} ({REPORTLAB_ERROR})"
            )
            st.caption("Session state (clés)")
            st.write(sorted(list(st.session_state.keys())))
            st.caption("Dernière exception")
            st.write(st.session_state.get("LAST_EXCEPTION", "—"))
            st.caption("Test rapide EODHD")
            if _get_api_key():
                try:
                    res = eodhd_get("/status")
                    st.write(res if res is not None else "Réponse vide")
                except Exception as e:
                    st.write(f"Erreur EODHD: {e}")
            else:
                st.write("Token EODHD absent")


    # Onglets principaux : Client / Valority
    tabs = st.tabs(["Portefeuille Client", "Portefeuille Valority"])

    with tabs[0]:
        subtabs = st.tabs(["Fonds recommandés", "Saisie libre"])
        with subtabs[0]:
            _add_from_reco_block("A_lines", "Ajouter un fonds recommandé (Client)")
        with subtabs[1]:
            _add_line_form_free("A_lines", "Portefeuille 1 — Client : saisie libre")
        st.markdown("#### Lignes actuelles — Portefeuille Client")
        for i, ln in enumerate(st.session_state.get("A_lines", [])):
            _line_card(ln, i, "A_lines")

    with tabs[1]:
        subtabs = st.tabs(["Fonds recommandés", "Saisie libre"])
        with subtabs[0]:
            _add_from_reco_block("B_lines", "Ajouter un fonds recommandé (Valority)")
        with subtabs[1]:
            _add_line_form_free("B_lines", "Portefeuille 2 — Valority : saisie libre")
        st.markdown("#### Lignes actuelles — Portefeuille Valority")
        for i, ln in enumerate(st.session_state.get("B_lines", [])):
            _line_card(ln, i, "B_lines")

    # ------------------------------------------------------------
    # Simulation (selon mode)
    # ------------------------------------------------------------
    mode = st.session_state.get("MODE_ANALYSE", "compare")
    show_client = mode in ("compare", "client")
    show_valority = mode in ("compare", "valority")

    single_target_A = id(st.session_state["A_lines"][0]) if (show_client and st.session_state["A_lines"]) else None
    single_target_B = id(st.session_state["B_lines"][0]) if (show_valority and st.session_state["B_lines"]) else None

    alloc_mode_code = st.session_state.get("ALLOC_MODE", "equal")

    custom_month_weights_A: Optional[Dict[int, float]] = None
    custom_oneoff_weights_A: Optional[Dict[int, float]] = None
    custom_month_weights_B: Optional[Dict[int, float]] = None
    custom_oneoff_weights_B: Optional[Dict[int, float]] = None

    if alloc_mode_code == "custom":
        if show_client:
            cmA = st.session_state.get("CUSTOM_M_A", {}) or {}
            coA = st.session_state.get("CUSTOM_O_A", {}) or {}
            tot_mA = sum(v for v in cmA.values() if v > 0)
            tot_oA = sum(v for v in coA.values() if v > 0)
            if tot_mA > 0:
                custom_month_weights_A = {k: v / tot_mA for k, v in cmA.items() if v > 0}
            if tot_oA > 0:
                custom_oneoff_weights_A = {k: v / tot_oA for k, v in coA.items() if v > 0}

        if show_valority:
            cmB = st.session_state.get("CUSTOM_M_B", {}) or {}
            coB = st.session_state.get("CUSTOM_O_B", {}) or {}
            tot_mB = sum(v for v in cmB.values() if v > 0)
            tot_oB = sum(v for v in coB.values() if v > 0)
            if tot_mB > 0:
                custom_month_weights_B = {k: v / tot_mB for k, v in cmB.items() if v > 0}
            if tot_oB > 0:
                custom_oneoff_weights_B = {k: v / tot_oB for k, v in coB.items() if v > 0}

    # Reset warnings avant chaque run
    st.session_state["DATE_WARNINGS"] = []

    # Valeurs par défaut (si on ne simule pas un des portefeuilles)
    dfA, brutA, netA, valA, xirrA, startA_min, fullA = pd.DataFrame(), 0.0, 0.0, 0.0, None, TODAY, TODAY
    dfB, brutB, netB, valB, xirrB, startB_min, fullB = pd.DataFrame(), 0.0, 0.0, 0.0, None, TODAY, TODAY

    if show_client:
        dfA, brutA, netA, valA, xirrA, startA_min, fullA = simulate_portfolio(
            st.session_state.get("A_lines", []),
            st.session_state.get("M_A", 0.0),
            st.session_state.get("ONE_A", 0.0),
            st.session_state.get("ONE_A_DATE", pd.Timestamp("2024-07-01").date()),
            alloc_mode_code,
            custom_month_weights_A,
            custom_oneoff_weights_A,
            single_target_A,
            st.session_state.get("EURO_RATE_A", 2.0),
            st.session_state.get("FEE_A", 0.0),
            portfolio_label="Client",
        )

    if show_valority:
        dfB, brutB, netB, valB, xirrB, startB_min, fullB = simulate_portfolio(
            st.session_state.get("B_lines", []),
            st.session_state.get("M_B", 0.0),
            st.session_state.get("ONE_B", 0.0),
            st.session_state.get("ONE_B_DATE", pd.Timestamp("2024-07-01").date()),
            alloc_mode_code,
            custom_month_weights_B,
            custom_oneoff_weights_B,
            single_target_B,
            st.session_state.get("EURO_RATE_B", 2.5),
            st.session_state.get("FEE_B", 0.0),
            portfolio_label="Valority",
        )

    # ------------------------------------------------------------
    # Avertissements sur les dates / 1ère VL
    # ------------------------------------------------------------
    if st.session_state.get("DATE_WARNINGS"):
        with st.expander("⚠️ Problèmes d'historique / dates de VL"):
            for msg in st.session_state["DATE_WARNINGS"]:
                st.warning(msg)

    # ------------------------------------------------------------
    # Graphique (évolution des portefeuilles)
    # ------------------------------------------------------------
    st.subheader("Évolution de la valeur des portefeuilles")

    # Déterminer le start_plot uniquement sur les portefeuilles affichés
    full_dates: List[pd.Timestamp] = []
    if show_client and isinstance(fullA, pd.Timestamp):
        full_dates.append(fullA)
    if show_valority and isinstance(fullB, pd.Timestamp):
        full_dates.append(fullB)

    start_plot = max(full_dates) if full_dates else TODAY

    idx = pd.bdate_range(start=start_plot, end=TODAY, freq="B")
    chart_df = pd.DataFrame(index=idx)

    if show_client and not dfA.empty:
        chart_df["Client"] = dfA.reindex(idx)["Valeur"].ffill()

    if show_valority and not dfB.empty:
        chart_df["Valority"] = dfB.reindex(idx)["Valeur"].ffill()

    # Passage en format long pour Altair
    chart_long = chart_df.reset_index().rename(columns={"index": "Date"})
    chart_long = chart_long.melt("Date", var_name="variable", value_name="Valeur (€)")

    if chart_long.dropna().empty:
        st.info("Ajoutez des lignes et/ou vérifiez vos paramètres pour afficher le graphique.")
    else:
        base = (
            alt.Chart(chart_long)
            .mark_line()
            .encode(
                x=alt.X("Date:T", title="Date"),
                y=alt.Y("Valeur (€):Q", title="Valeur (€)"),
                color=alt.Color("variable:N", title="Portefeuille"),
                tooltip=[
                    alt.Tooltip("Date:T", title="Date"),
                    alt.Tooltip("variable:N", title="Portefeuille"),
                    alt.Tooltip("Valeur (€):Q", title="Valeur", format=",.2f"),
                ],
            )
            .properties(height=360, width="container")
        )
        st.altair_chart(base, use_container_width=True)

    # ------------------------------------------------------------
    # Synthèse chiffrée : cartes Client / Valority
    # ------------------------------------------------------------
    st.subheader("Synthèse chiffrée")

    mode = st.session_state.get("MODE_ANALYSE", "compare")

    # Période analysée (uniquement sur ce qui est affiché)
    period_dates: List[pd.Timestamp] = []
    if mode in ("compare", "client") and isinstance(startA_min, pd.Timestamp):
        period_dates.append(startA_min)
    if mode in ("compare", "valority") and isinstance(startB_min, pd.Timestamp):
        period_dates.append(startB_min)

    if period_dates:
        start_global = min(period_dates)
        st.caption(f"Période analysée : du **{fmt_date(start_global)}** au **{fmt_date(TODAY)}**")

    perf_tot_client = (valA / netA - 1.0) * 100.0 if (show_client and netA > 0) else None
    perf_tot_valority = (valB / netB - 1.0) * 100.0 if (show_valority and netB > 0) else None

    # ✅ 2 colonnes si compare, sinon 1 colonne (container)
    if mode == "compare":
        col_client, col_valority = st.columns(2)
    else:
        col_client = st.container()
        col_valority = st.container()

    # ----- Carte Client -----
    if mode in ("compare", "client"):
        with col_client:
            with st.container(border=True):
                st.markdown("#### 🧍 Situation actuelle — Client")
                st.metric("Valeur actuelle", to_eur(valA))
                st.markdown(
                    f"""
- Montants réellement investis (après frais) : **{to_eur(netA)}**
- Montants versés (brut) : {to_eur(brutA)}
- Rendement total depuis le début : **{perf_tot_client:.2f}%**
"""
                    if perf_tot_client is not None
                    else f"""
- Montants réellement investis (après frais) : **{to_eur(netA)}**
- Montants versés (brut) : {to_eur(brutA)}
- Rendement total depuis le début : **—**
"""
                )
                st.markdown(
                    f"- Rendement annualisé (XIRR) : **{xirrA:.2f}%**"
                    if xirrA is not None
                    else "- Rendement annualisé (XIRR) : **—**"
                )


    # ----- Carte Valority -----
    if mode in ("compare", "valority"):
        with col_valority:
            with st.container(border=True):
                st.markdown("#### 🏢 Simulation — Allocation Valority")
                st.metric("Valeur actuelle simulée", to_eur(valB))
                st.markdown(
                    f"""
- Montants réellement investis (après frais) : **{to_eur(netB)}**
- Montants versés (brut) : {to_eur(brutB)}
- Rendement total depuis le début : **{perf_tot_valority:.2f}%**
"""
                    if perf_tot_valority is not None
                    else f"""
- Montants réellement investis (après frais) : **{to_eur(netB)}**
- Montants versés (brut) : {to_eur(brutB)}
- Rendement total depuis le début : **—**
"""
                )
                st.markdown(
                    f"- Rendement annualisé (XIRR) : **{xirrB:.2f}%**"
                    if xirrB is not None
                    else "- Rendement annualisé (XIRR) : **—**"
                )


    def build_html_report(report: Dict[str, Any]) -> str:
        """
        Construit un rapport HTML exportable pour le client.
        Le contenu repose sur 'report', préparé plus bas dans le code.
        """
        as_of = report.get("as_of", "")
        mode = report.get("mode", "compare")
        synthA = report.get("client_summary", {})
        synthB = report.get("valority_summary", {})
        comp = report.get("comparison", {})

        dfA_lines = report.get("df_client_lines")
        dfB_lines = report.get("df_valority_lines")
        dfA_val = report.get("dfA_val")
        dfB_val = report.get("dfB_val")

        def _fmt_eur(x):
            try:
                return f"{x:,.2f} €".replace(",", " ").replace(".", ",")
            except Exception:
                return str(x)

        # Tables en HTML
        html_client_lines = dfA_lines.to_html(index=False, border=0, justify="left") if dfA_lines is not None else ""
        html_valority_lines = dfB_lines.to_html(index=False, border=0, justify="left") if dfB_lines is not None else ""

        if dfA_val is not None:
            html_A_val = dfA_val.to_html(index=False, border=0, justify="left")
        else:
            html_A_val = ""

        if dfB_val is not None:
            html_B_val = dfB_val.to_html(index=False, border=0, justify="left")
        else:
            html_B_val = ""

        if mode == "compare":
            synth_html = f"""
<div class="block">
  <h3>Situation actuelle — Client</h3>
  <ul>
    <li>Valeur actuelle : <b>{_fmt_eur(synthA.get("val", 0))}</b></li>
    <li>Montants réellement investis (net) : {_fmt_eur(synthA.get("net", 0))}</li>
    <li>Montants versés (brut) : {_fmt_eur(synthA.get("brut", 0))}</li>
    <li>Rendement total depuis le début : <b>{synthA.get("perf_tot_pct", 0):.2f} %</b></li>
    <li>Rendement annualisé (XIRR) : <b>{synthA.get("irr_pct", 0):.2f} %</b></li>
  </ul>
</div>

<div class="block">
  <h3>Simulation — Allocation Valority</h3>
  <ul>
    <li>Valeur actuelle simulée : <b>{_fmt_eur(synthB.get("val", 0))}</b></li>
    <li>Montants réellement investis (net) : {_fmt_eur(synthB.get("net", 0))}</li>
    <li>Montants versés (brut) : {_fmt_eur(synthB.get("brut", 0))}</li>
    <li>Rendement total depuis le début : <b>{synthB.get("perf_tot_pct", 0):.2f} %</b></li>
    <li>Rendement annualisé (XIRR) : <b>{synthB.get("irr_pct", 0):.2f} %</b></li>
  </ul>
</div>

<div class="block">
  <h3>Comparaison Client vs Valority</h3>
  <ul>
    <li>Différence de valeur finale : <b>{_fmt_eur(comp.get("delta_val", 0))}</b></li>
    <li>Écart de performance totale (Valority – Client) :
        <b>{comp.get("delta_perf_pct", 0):.2f} %</b></li>
  </ul>
</div>
"""
            detail_html = f"""
<h2>2. Détail des lignes</h2>

<h3>Portefeuille Client</h3>
{html_client_lines}

<h3>Portefeuille Valority</h3>
{html_valority_lines}
"""
            hist_html = f"""
<h2>3. Historique de la valeur des portefeuilles</h2>

<h3>Client – Valeur du portefeuille par date</h3>
{html_A_val}

<h3>Valority – Valeur du portefeuille par date</h3>
{html_B_val}
"""
        elif mode == "valority":
            synth_html = f"""
<div class="block">
  <h3>Simulation — Allocation Valority</h3>
  <ul>
    <li>Valeur actuelle simulée : <b>{_fmt_eur(synthB.get("val", 0))}</b></li>
    <li>Montants réellement investis (net) : {_fmt_eur(synthB.get("net", 0))}</li>
    <li>Montants versés (brut) : {_fmt_eur(synthB.get("brut", 0))}</li>
    <li>Rendement total depuis le début : <b>{synthB.get("perf_tot_pct", 0):.2f} %</b></li>
    <li>Rendement annualisé (XIRR) : <b>{synthB.get("irr_pct", 0):.2f} %</b></li>
  </ul>
</div>
"""
            detail_html = f"""
<h2>2. Détail des lignes</h2>

<h3>Portefeuille Valority</h3>
{html_valority_lines}
"""
            hist_html = f"""
<h2>3. Historique de la valeur du portefeuille</h2>

<h3>Valority – Valeur du portefeuille par date</h3>
{html_B_val}
"""
        else:
            synth_html = f"""
<div class="block">
  <h3>Situation actuelle — Client</h3>
  <ul>
    <li>Valeur actuelle : <b>{_fmt_eur(synthA.get("val", 0))}</b></li>
    <li>Montants réellement investis (net) : {_fmt_eur(synthA.get("net", 0))}</li>
    <li>Montants versés (brut) : {_fmt_eur(synthA.get("brut", 0))}</li>
    <li>Rendement total depuis le début : <b>{synthA.get("perf_tot_pct", 0):.2f} %</b></li>
    <li>Rendement annualisé (XIRR) : <b>{synthA.get("irr_pct", 0):.2f} %</b></li>
  </ul>
</div>
"""
            detail_html = f"""
<h2>2. Détail des lignes</h2>

<h3>Portefeuille Client</h3>
{html_client_lines}
"""
            hist_html = f"""
<h2>3. Historique de la valeur du portefeuille</h2>

<h3>Client – Valeur du portefeuille par date</h3>
{html_A_val}
"""

        html = f"""
<!DOCTYPE html>
<html lang="fr">
<head>
<meta charset="utf-8" />
<title>Rapport de portefeuille</title>
<style>
body {{
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  margin: 24px;
  color: #222;
}}
h1, h2, h3 {{
  margin-top: 24px;
}}
table {{
  border-collapse: collapse;
  width: 100%;
  margin: 8px 0 16px 0;
  font-size: 14px;
}}
th, td {{
  border: 1px solid #ddd;
  padding: 6px 8px;
}}
th {{
  background-color: #f5f5f5;
  text-align: left;
}}
.small {{
  font-size: 12px;
  color: #666;
}}
.block {{
  border: 1px solid #eee;
  border-radius: 8px;
  padding: 12px 16px;
  margin-bottom: 16px;
  background-color: #fafafa;
}}
</style>
</head>
<body>

<h1>Rapport de portefeuille</h1>
<p class="small">Date de génération : {as_of}</p>

<h2>1. Synthèse chiffrée</h2>

{synth_html}
{detail_html}
{hist_html}

<p class="small">
Ce document est fourni à titre informatif uniquement et ne constitue pas un conseil en investissement
personnalisé.
</p>

</body>
    </html>
"""
        return html


    def _add_table_to_story(
        story: List[Any],
        df: pd.DataFrame,
        col_widths: Optional[List[float]] = None,
        font_size: int = 9,
    ):
        if df.empty:
            story.append(Paragraph("Données indisponibles.", getSampleStyleSheet()["Normal"]))
            return
        headers = list(df.columns)
        fmt_rows = []
        for _, row in df.iterrows():
            formatted = []
            for col, val in row.items():
                if "€" in col:
                    formatted.append(fmt_eur_fr(val))
                elif "%" in col:
                    formatted.append(fmt_pct_fr(val))
                else:
                    formatted.append(str(val))
            fmt_rows.append(formatted)
        data = [headers] + fmt_rows
        table = Table(data, repeatRows=1, colWidths=col_widths)
        style = [
            ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
            ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), font_size),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("LEFTPADDING", (0, 0), (-1, -1), 4),
            ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ]
        for i in range(1, len(data)):
            if i % 2 == 0:
                style.append(("BACKGROUND", (0, i), (-1, i), colors.whitesmoke))
        table.setStyle(TableStyle(style))
        story.append(table)


    def _fig_to_rl_image(fig: plt.Figure, width: float = 480, height: float = 270) -> Image:
        if not MATPLOTLIB_AVAILABLE:
            return None
        buf = BytesIO()
        fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
        buf.seek(0)
        plt.close(fig)
        return Image(buf, width=width, height=height)


    def _build_value_chart(df_map: Dict[str, pd.DataFrame]) -> Optional[Image]:
        if not MATPLOTLIB_AVAILABLE:
            return None
        if not df_map:
            return None
        fig, ax = plt.subplots(figsize=(6, 3))
        has_data = False
        for label, df in df_map.items():
            if df is None or df.empty or "Valeur" not in df.columns:
                continue
            ax.plot(df.index, df["Valeur"], label=label)
            has_data = True
        if not has_data:
            plt.close(fig)
            return None
        ax.set_title("Évolution de la valeur du portefeuille")
        ax.set_xlabel("Date")
        ax.set_ylabel("Valeur (€)")
        ax.legend(loc="best")
        fig.autofmt_xdate()
        return _fig_to_rl_image(fig)


    def _wrap_label(label: str, width: int = 28) -> str:
        if not label:
            return "—"
        return "\n".join(textwrap.wrap(str(label), width=width)) or str(label)


    def _allocation_from_positions(df_positions: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
        df = df_positions.copy()
        df = df[df["Valeur actuelle €"] >= 0]
        df["Nom"] = df["Nom"].fillna("—")
        df["ISIN / Code"] = df["ISIN / Code"].fillna("—")

        total_value = df["Valeur actuelle €"].sum()
        if total_value > 0:
            df["Poids"] = df["Valeur actuelle €"] / total_value
            basis_label = "Valeur actuelle"
        else:
            total_net = df["Net investi €"].sum()
            if total_net > 0:
                df["Poids"] = df["Net investi €"] / total_net
            else:
                df["Poids"] = 0.0
                if len(df) > 0:
                    df.loc[df.index[0], "Poids"] = 1.0
            basis_label = "Net investi"

        df = df.sort_values("Poids", ascending=False)
        if len(df) > 8:
            df_main = df.iloc[:8].copy()
            df_other = df.iloc[8:]
            other_row = pd.DataFrame(
                {
                    "Nom": ["Autres"],
                    "ISIN / Code": ["—"],
                    "Net investi €": [df_other["Net investi €"].sum()],
                    "Valeur actuelle €": [df_other["Valeur actuelle €"].sum()],
                    "Poids": [df_other["Poids"].sum()],
                }
            )
            df = pd.concat([df_main, other_row], ignore_index=True)

        df["Part %"] = df["Poids"] * 100.0
        return df, basis_label


    def _build_allocation_donut(
        df_alloc: pd.DataFrame,
        title: str,
        figsize: Tuple[float, float] = (6.0, 3.4),
    ) -> Optional[Image]:
        if not MATPLOTLIB_AVAILABLE or df_alloc.empty:
            return None
        fig, ax = plt.subplots(figsize=figsize)
        wedges, _ = ax.pie(
            df_alloc["Poids"],
            startangle=90,
            labels=None,
            wedgeprops=dict(width=0.35, edgecolor="white"),
        )
        labels = [
            f"{_wrap_label(nm)} ({pct:.1f}%)"
            for nm, pct in zip(df_alloc["Nom"], df_alloc["Part %"])
        ]
        ax.legend(
            wedges,
            labels,
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            frameon=False,
            fontsize=8,
        )
        ax.set_title(title)
        ax.set_aspect("equal")
        fig.tight_layout(rect=[0, 0, 0.78, 1])
        return _fig_to_rl_image(fig, width=460, height=280)


    def _build_envelope_breakdown(
        lines: List[Dict[str, Any]],
        title: str,
    ) -> Tuple[Optional[Image], Optional[str]]:
        if not lines:
            return None, "Répartition par enveloppe : —"
        categories = {"Fonds euros": 0.0, "UC": 0.0, "Structurés": 0.0}
        for ln in lines:
            isin = str(ln.get("isin", "")).upper()
            val = float(ln.get("value", 0.0))
            if isin == "EUROFUND":
                categories["Fonds euros"] += val
            elif isin == "STRUCTURED":
                categories["Structurés"] += val
            else:
                categories["UC"] += val
        total = sum(categories.values())
        if total <= 0:
            return None, "Répartition par enveloppe : —"

        shares = {k: v / total for k, v in categories.items() if v > 0}
        major = max(shares.items(), key=lambda x: x[1])
        if sum(1 for v in shares.values() if v >= 0.01) < 2:
            return None, f"Répartition par enveloppe : {major[1] * 100:.1f}% {major[0]}"

        if not MATPLOTLIB_AVAILABLE:
            return None, None
        labels = list(shares.keys())
        values = [shares[k] * 100 for k in labels]
        fig, ax = plt.subplots(figsize=(6.0, 1.8))
        ax.barh(labels, values, color="#4C78A8")
        ax.set_xlim(0, 100)
        ax.set_xlabel("%")
        ax.set_title(title)
        for i, v in enumerate(values):
            ax.text(min(v + 1, 98), i, f"{v:.1f}%", va="center", fontsize=8)
        fig.tight_layout()
        return _fig_to_rl_image(fig, width=460, height=140), None


    def _build_contribution_bar(df_positions: pd.DataFrame) -> Optional[Image]:
        if not MATPLOTLIB_AVAILABLE or df_positions.empty:
            return None
        df = df_positions.copy()
        if not {"Nom", "Valeur actuelle €", "Net investi €"}.issubset(df.columns):
            return None
        df["Contribution €"] = df["Valeur actuelle €"] - df["Net investi €"]
        df = df.sort_values("Contribution €", ascending=False)
        fig_height = max(2.0, min(4.2, 0.35 * len(df) + 1.2))
        fig, ax = plt.subplots(figsize=(6.2, fig_height))
        ax.barh(df["Nom"], df["Contribution €"], color="#2F6F9F")
        ax.invert_yaxis()
        ax.set_title("Contribution à la performance (€)")
        ax.axvline(0, color="black", linewidth=0.5)
        ax.tick_params(axis="y", labelsize=8)
        for i, v in enumerate(df["Contribution €"]):
            offset = 0.01 * abs(v) if v != 0 else 0.5
            x_pos = v + offset if v >= 0 else v - offset
            ax.text(x_pos, i, fmt_eur_fr(v), va="center", fontsize=8)
        fig.tight_layout()
        return _fig_to_rl_image(fig, width=460, height=200)


    def _build_single_line_bar(label: str, value: float, title: str) -> Optional[Image]:
        if not MATPLOTLIB_AVAILABLE:
            return None
        fig, ax = plt.subplots(figsize=(6.0, 1.3))
        ax.barh([label], [100], color="#4C78A8")
        ax.set_xlim(0, 100)
        ax.set_title(title)
        ax.set_xticks([])
        ax.tick_params(axis="y", labelsize=8)
        fig.tight_layout()
        return _fig_to_rl_image(fig, width=460, height=90)


    def fmt_eur_pdf(x: Any) -> str:
        return fmt_eur_fr(x)


    def generate_pdf_report(report: Dict[str, Any]) -> bytes:
        if not REPORTLAB_AVAILABLE:
            raise RuntimeError(f"PDF indisponible: {REPORTLAB_ERROR}")
        if not MATPLOTLIB_AVAILABLE:
            raise RuntimeError(f"PDF indisponible: {MATPLOTLIB_ERROR}")

        class NumberedCanvas(canvas.Canvas):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self._saved_page_states = []

            def showPage(self):
                self._saved_page_states.append(dict(self.__dict__))
                self._startPage()

            def save(self):
                page_count = len(self._saved_page_states)
                for state in self._saved_page_states:
                    self.__dict__.update(state)
                    self._draw_header_footer(page_count)
                    super().showPage()
                super().save()

            def _draw_header_footer(self, page_count: int):
                width, height = A4
                self.setFillColor(colors.HexColor("#1F3B6D"))
                self.setFont("Helvetica-Bold", 10)
                self.drawString(36, height - 30, "Rapport de portefeuille – Valority")
                self.setFillColor(colors.grey)
                self.setFont("Helvetica", 8)
                self.drawRightString(width - 36, height - 30, report.get("as_of", ""))
                self.setFillColor(colors.grey)
                self.setFont("Helvetica", 7)
                self.drawString(
                    36,
                    24,
                    "Document informatif, ne constitue pas un conseil en investissement.",
                )
                self.drawRightString(width - 36, 24, f"Page {self.getPageNumber()} / {page_count}")

        buffer = BytesIO()
        doc = SimpleDocTemplate(
            buffer,
            pagesize=A4,
            leftMargin=36,
            rightMargin=36,
            topMargin=54,
            bottomMargin=48,
        )
        base_styles = getSampleStyleSheet()
        styles = {
            "title": ParagraphStyle(
                "Title",
                parent=base_styles["Title"],
                textColor=colors.HexColor("#1F3B6D"),
                fontSize=20,
                spaceAfter=12,
            ),
            "h1": ParagraphStyle(
                "H1",
                parent=base_styles["Heading1"],
                textColor=colors.HexColor("#1F3B6D"),
                fontSize=14,
                spaceAfter=8,
            ),
            "h2": ParagraphStyle(
                "H2",
                parent=base_styles["Heading2"],
                textColor=colors.HexColor("#4B5563"),
                fontSize=12,
                spaceAfter=6,
            ),
            "small": ParagraphStyle(
                "Small",
                parent=base_styles["Normal"],
                fontSize=8,
                textColor=colors.grey,
            ),
            "kpi": ParagraphStyle(
                "KPI",
                parent=base_styles["Normal"],
                fontSize=10,
                textColor=colors.HexColor("#111827"),
            ),
        }
        story: List[Any] = []

        def _kpi_table(title: str, rows: List[Tuple[str, str]]) -> Table:
            data = [[Paragraph(f"<b>{title}</b>", styles["h2"]), ""]]
            for label, value in rows:
                data.append([Paragraph(label, styles["small"]), Paragraph(value, styles["kpi"])])
            table = Table(data, colWidths=[160, 120])
            table.setStyle(
                TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#EEF2F7")),
                        ("BACKGROUND", (0, 1), (-1, -1), colors.whitesmoke),
                        ("BOX", (0, 0), (-1, -1), 0.5, colors.lightgrey),
                        ("INNERGRID", (0, 1), (-1, -1), 0.25, colors.lightgrey),
                        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                        ("LEFTPADDING", (0, 0), (-1, -1), 6),
                        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                    ]
                )
            )
            return table

        story.append(Paragraph("Rapport client", styles["title"]))
        story.append(Paragraph(f"Date de génération : {report.get('as_of', '')}", styles["small"]))
        story.append(Spacer(1, 12))

        mode = report.get("mode", "compare")
        synthA = report.get("client_summary", {})
        synthB = report.get("valority_summary", {})
        comp = report.get("comparison", {})

        story.append(Paragraph("Synthèse", styles["h1"]))
        if mode == "compare":
            client_rows = [
                ("Valeur actuelle", fmt_eur_fr(synthA.get("val", 0))),
                ("Net investi", fmt_eur_fr(synthA.get("net", 0))),
                ("Brut versé", fmt_eur_fr(synthA.get("brut", 0))),
                ("Perf totale", fmt_pct_fr(synthA.get("perf_tot_pct", 0))),
                ("XIRR", fmt_pct_fr(synthA.get("irr_pct", 0))),
            ]
            valority_rows = [
                ("Valeur actuelle", fmt_eur_fr(synthB.get("val", 0))),
                ("Net investi", fmt_eur_fr(synthB.get("net", 0))),
                ("Brut versé", fmt_eur_fr(synthB.get("brut", 0))),
                ("Perf totale", fmt_pct_fr(synthB.get("perf_tot_pct", 0))),
                ("XIRR", fmt_pct_fr(synthB.get("irr_pct", 0))),
            ]
            table = Table(
                [[_kpi_table("Client", client_rows), _kpi_table("Valority", valority_rows)]],
                colWidths=[240, 240],
            )
            story.append(table)
            story.append(Spacer(1, 8))
            comp_rows = [
                ("Différence de valeur", fmt_eur_fr(comp.get("delta_val", 0))),
                ("Écart de performance", fmt_pct_fr(comp.get("delta_perf_pct", 0))),
            ]
            story.append(_kpi_table("Comparaison", comp_rows))
        else:
            title = "Valority" if mode == "valority" else "Client"
            synth = synthB if mode == "valority" else synthA
            rows = [
                ("Valeur actuelle", fmt_eur_fr(synth.get("val", 0))),
                ("Net investi", fmt_eur_fr(synth.get("net", 0))),
                ("Brut versé", fmt_eur_fr(synth.get("brut", 0))),
                ("Perf totale", fmt_pct_fr(synth.get("perf_tot_pct", 0))),
                ("XIRR", fmt_pct_fr(synth.get("irr_pct", 0))),
            ]
            story.append(_kpi_table(title, rows))
            fees = report.get("fees_analysis", {})
            if fees:
                story.append(Spacer(1, 8))
                fees_rows = [
                    ("Frais d’entrée payés", fmt_eur_fr(fees.get("fees_paid", 0))),
                    ("Valeur créée", fmt_eur_fr(fees.get("value_created", 0))),
                    ("Valeur/an", fmt_eur_fr(fees.get("value_per_year", 0))),
                ]
                story.append(_kpi_table("Frais & valeur créée", fees_rows))

        story.append(Spacer(1, 12))
        story.append(Paragraph("Graphiques", styles["h1"]))

        value_chart = _build_value_chart(report.get("df_map", {}))
        if value_chart is not None:
            story.append(value_chart)
            story.append(Spacer(1, 8))
        else:
            story.append(Paragraph("Données indisponibles pour la courbe de valeur.", styles["small"]))

        def add_portfolio_details_section(
            label: str,
            positions_df: pd.DataFrame,
            lines_values: List[Dict[str, Any]],
        ):
            # PAGE 2/4 — Détail du portefeuille
            story.append(PageBreak())
            story.append(Paragraph(f"Détail du portefeuille — {label}", styles["h1"]))

            if isinstance(positions_df, pd.DataFrame) and not positions_df.empty:
                df_alloc, basis_label = _allocation_from_positions(positions_df)

                if len(df_alloc) >= 2:
                    story.append(Paragraph(f"Allocation par ligne ({basis_label})", styles["h2"]))
                    donut = _build_allocation_donut(df_alloc, "Allocation par ligne")
                    if donut is not None:
                        story.append(donut)
                        story.append(Spacer(1, 6))
                else:
                    line = df_alloc.iloc[0] if not df_alloc.empty else None
                    name = line["Nom"] if line is not None else "—"
                    story.append(
                        Paragraph(
                            f"Portefeuille concentré : 100% sur <b>{name}</b>.",
                            styles["kpi"],
                        )
                    )
                    bar = _build_single_line_bar(_wrap_label(name), 100.0, "Répartition 100%")
                    if bar is not None:
                        story.append(bar)
                        story.append(Spacer(1, 6))

                alloc_table = df_alloc[
                    ["Nom", "ISIN / Code", "Part %", "Net investi €", "Valeur actuelle €"]
                ].copy()
                _add_table_to_story(
                    story,
                    alloc_table,
                    col_widths=[170, 80, 55, 95, 95],
                    font_size=9,
                )
                story.append(Spacer(1, 6))

                envelope_chart, envelope_text = _build_envelope_breakdown(
                    lines_values,
                    "Répartition par enveloppe",
                )
                if envelope_chart is not None:
                    story.append(envelope_chart)
                elif envelope_text:
                    story.append(Paragraph(envelope_text, styles["small"]))
            else:
                story.append(Paragraph("Données indisponibles pour la composition.", styles["small"]))

            # PAGE 3/5 — Contribution & Positions
            story.append(PageBreak())
            story.append(Paragraph(f"Contribution & positions — {label}", styles["h1"]))

            if isinstance(positions_df, pd.DataFrame) and not positions_df.empty:
                if len(positions_df) == 1:
                    ln = positions_df.iloc[0]
                    story.append(
                        Paragraph(
                            f"Contribution : <b>{ln['Nom']}</b> = {fmt_eur_pdf(ln['Valeur actuelle €'] - ln['Net investi €'])}",
                            styles["kpi"],
                        )
                    )
                    bar = _build_single_line_bar(_wrap_label(ln["Nom"]), 100.0, "Contribution (ligne unique)")
                    if bar is not None:
                        story.append(bar)
                        story.append(Spacer(1, 6))
                else:
                    contrib_chart = _build_contribution_bar(positions_df)
                    if contrib_chart is not None:
                        story.append(contrib_chart)
                        story.append(Spacer(1, 6))
            else:
                story.append(Paragraph("Données indisponibles pour la contribution.", styles["small"]))

            story.append(Paragraph("Positions", styles["h2"]))
            if isinstance(positions_df, pd.DataFrame) and not positions_df.empty:
                positions_table = positions_df[
                    [
                        "Nom",
                        "ISIN / Code",
                        "Date d'achat",
                        "Net investi €",
                        "Valeur actuelle €",
                        "Perf €",
                        "Perf %",
                    ]
                ].copy()
                _add_table_to_story(
                    story,
                    positions_table,
                    col_widths=[150, 70, 65, 80, 80, 50, 45],
                    font_size=9,
                )
            else:
                story.append(Paragraph("Données indisponibles.", styles["small"]))

        if mode == "compare":
            add_portfolio_details_section(
                "Client",
                report.get("positions_df_client", pd.DataFrame()),
                report.get("lines_client", []),
            )
            add_portfolio_details_section(
                "Valority",
                report.get("positions_df_valority", pd.DataFrame()),
                report.get("lines_valority", []),
            )
        else:
            add_portfolio_details_section(
                "Valority" if mode == "valority" else "Client",
                report.get("positions_df", pd.DataFrame()),
                report.get("lines", []),
            )

        doc.build(story, canvasmaker=NumberedCanvas)
        buffer.seek(0)
        return buffer.read()


    def _years_between(d0: pd.Timestamp, d1: pd.Timestamp) -> float:
        return max(0.0, (d1 - d0).days / 365.25)


    report_data = {
        "as_of": fmt_date(TODAY),
        "mode": st.session_state.get("MODE_ANALYSE", "compare"),
    }

    mode_report = report_data["mode"]
    df_client_lines = build_positions_dataframe("A_lines") if show_client else pd.DataFrame()
    df_valority_lines = build_positions_dataframe("B_lines") if show_valority else pd.DataFrame()

    report_data["df_client_lines"] = df_client_lines
    report_data["df_valority_lines"] = df_valority_lines
    report_data["positions_df_client"] = df_client_lines
    report_data["positions_df_valority"] = df_valority_lines
    report_data["dfA_val"] = (
        dfA.reset_index().rename(columns={"index": "Date"}) if (show_client and not dfA.empty) else pd.DataFrame()
    )
    report_data["dfB_val"] = (
        dfB.reset_index().rename(columns={"index": "Date"}) if (show_valority and not dfB.empty) else pd.DataFrame()
    )
    report_data["client_summary"] = {
        "val": valA,
        "net": netA,
        "brut": brutA,
        "perf_tot_pct": perf_tot_client or 0.0,
        "irr_pct": xirrA or 0.0,
    }
    report_data["valority_summary"] = {
        "val": valB,
        "net": netB,
        "brut": brutB,
        "perf_tot_pct": perf_tot_valority or 0.0,
        "irr_pct": xirrB or 0.0,
    }
    report_data["comparison"] = {
        "delta_val": (valB - valA) if (valA is not None and valB is not None) else 0.0,
        "delta_perf_pct": (perf_tot_valority - perf_tot_client)
        if (perf_tot_client is not None and perf_tot_valority is not None)
        else 0.0,
    }

    df_map: Dict[str, pd.DataFrame] = {}
    if mode_report in ("compare", "client") and not dfA.empty:
        df_map["Client"] = dfA
    if mode_report in ("compare", "valority") and not dfB.empty:
        df_map["Valority"] = dfB
    report_data["df_map"] = df_map

    if mode_report == "compare":
        positions_df = df_valority_lines if not df_valority_lines.empty else df_client_lines
        lines = st.session_state.get("B_lines", []) or st.session_state.get("A_lines", [])
    else:
        if mode_report == "valority":
            positions_df = df_valority_lines
            lines = st.session_state.get("B_lines", [])
            start_min = startB_min
            brut = brutB
            net = netB
            val = valB
        else:
            positions_df = df_client_lines
            lines = st.session_state.get("A_lines", [])
            start_min = startA_min
            brut = brutA
            net = netA
            val = valA

        years = _years_between(start_min, TODAY) if isinstance(start_min, pd.Timestamp) else 0.0
        fees_paid = max(0.0, brut - net) if brut is not None and net is not None else 0.0
        value_created = (val - net) if val is not None and net is not None else 0.0
        value_per_year = (value_created / years) if years > 0 else 0.0
        report_data["fees_analysis"] = {
            "fees_paid": fees_paid,
            "value_created": value_created,
            "value_per_year": value_per_year,
        }

    def _build_lines_with_values(
        lines_src: List[Dict[str, Any]],
        positions_src: pd.DataFrame,
    ) -> List[Dict[str, Any]]:
        items: List[Dict[str, Any]] = []
        if isinstance(positions_src, pd.DataFrame) and not positions_src.empty:
            for ln in lines_src:
                isin = ln.get("isin", "")
                name = ln.get("name", "")
                match = positions_src[
                    (positions_src["Nom"] == name) & (positions_src["ISIN / Code"] == isin)
                ]
                val = float(match["Valeur actuelle €"].iloc[0]) if not match.empty else 0.0
                items.append({"isin": isin, "value": val})
        return items

    report_data["positions_df"] = positions_df
    report_data["lines"] = _build_lines_with_values(lines, positions_df)
    report_data["lines_client"] = _build_lines_with_values(
        st.session_state.get("A_lines", []),
        df_client_lines,
    )
    report_data["lines_valority"] = _build_lines_with_values(
        st.session_state.get("B_lines", []),
        df_valority_lines,
    )

    st.session_state["REPORT_DATA"] = report_data

    if report_data is not None:
        html_report = build_html_report(report_data)
        st.download_button(
            "📄 Télécharger le rapport complet (HTML)",
            data=html_report.encode("utf-8"),
            file_name="rapport_portefeuille_valority.html",
            mime="text/html",
        )
        pdf_bytes = None
        try:
            pdf_bytes = generate_pdf_report(report_data)
        except Exception as e:
            st.warning(f"PDF indisponible: {e}")
            st.session_state["LAST_EXCEPTION"] = str(e)
        if pdf_bytes:
            st.download_button(
                "📄 Télécharger le rapport (PDF)",
                data=pdf_bytes,
                file_name="rapport_portefeuille.pdf",
                mime="application/pdf",
            )

    # ------------------------------------------------------------
    # Bloc final : Comparaison OU "Frais & valeur créée"
    # ------------------------------------------------------------
    mode = st.session_state.get("MODE_ANALYSE", "compare")

    # ============================
    # CAS 1 — MODE COMPARAISON
    # ============================
    if mode == "compare":
        st.subheader("📌 Comparaison : Client vs Valority")

        gain_vs_client = (valB - valA) if (valA is not None and valB is not None) else 0.0
        delta_xirr = (xirrB - xirrA) if (xirrA is not None and xirrB is not None) else None
        perf_diff_tot = (
            (perf_tot_valority - perf_tot_client)
            if (perf_tot_client is not None and perf_tot_valority is not None)
            else None
        )

        with st.container(border=True):
            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("Gain en valeur", to_eur(gain_vs_client))
            with c2:
                st.metric(
                    "Surperformance totale",
                    f"{perf_diff_tot:+.2f}%" if perf_diff_tot is not None else "—",
                )
            with c3:
                st.metric(
                    "Surperformance annualisée (Δ XIRR)",
                    f"{delta_xirr:+.2f}%" if delta_xirr is not None else "—",
                )

            st.markdown(
                f"""
Aujourd’hui, avec votre allocation actuelle, votre portefeuille vaut **{to_eur(valA)}**.  
Avec l’allocation Valority, il serait autour de **{to_eur(valB)}**, soit environ **{to_eur(gain_vs_client)}** de plus.
"""
            )

    # ============================
    # CAS 2 — MODE ANALYSE SIMPLE
    # ============================
    else:
        # Sélection des variables selon le mode
        if mode == "valority":
            brut = brutB
            net = netB
            val = valB
            start_min = startB_min
            irr = xirrB
            fee_pct = st.session_state.get("FEE_B", 0.0)
            title = "🏢 Allocation Valority — Frais & valeur créée"
        else:  # mode == "client"
            brut = brutA
            net = netA
            val = valA
            start_min = startA_min
            irr = xirrA
            fee_pct = st.session_state.get("FEE_A", 0.0)
            title = "🧍 Portefeuille — Frais & valeur créée"

        st.subheader("📌 Analyse : frais & valeur créée")

        if brut > 0 and net >= 0 and val >= 0 and isinstance(start_min, pd.Timestamp):
            fees_paid = max(0.0, brut - net)     # frais d'entrée réellement payés
            value_created = val - net            # valeur créée vs capital réellement investi
            years = _years_between(start_min, TODAY)
            value_per_year = (value_created / years) if years > 0 else None

            with st.container(border=True):
                st.markdown(f"#### {title}")
                st.caption(
                    f"Période : **{fmt_date(start_min)} → {fmt_date(TODAY)}** "
                    f"• Frais d’entrée : **{fee_pct:.2f}%**"
                )

                c1, c2, c3 = st.columns(3)
                with c1:
                    st.metric("Frais d’entrée payés", to_eur(fees_paid))
                with c2:
                    st.metric("Valeur créée (net)", to_eur(value_created))
                with c3:
                    st.metric(
                        "Valeur créée / an (moyenne)",
                        to_eur(value_per_year) if value_per_year is not None else "—",
                    )

                st.markdown(
                    f"""
- Montants versés (brut) : **{to_eur(brut)}**
- Montants réellement investis (après frais) : **{to_eur(net)}**
- Valeur actuelle : **{to_eur(val)}**
"""
                )

                if irr is not None:
                    st.markdown(f"- Rendement annualisé (XIRR) : **{irr:.2f}%**")
                else:
                    st.markdown("- Rendement annualisé (XIRR) : **—**")

                # Message "vendeur" mais strictement factuel
                if fees_paid > 0:
                    ratio = (value_created / fees_paid) if fees_paid > 0 else None
                    if ratio is not None:
                        st.markdown(
                            f"**Lecture :** {to_eur(fees_paid)} de frais d’entrée ont été compensés par "
                            f"**{to_eur(value_created)}** de valeur nette créée à date "
                            f"(**×{ratio:.1f}**)."
                        )
        else:
            st.info("Ajoutez des lignes (et/ou des versements) pour afficher l’analyse frais & valeur créée.")


    # ------------------------------------------------------------
    # Tables positions
    # ------------------------------------------------------------
    if show_client:
        positions_table("Portefeuille 1 — Client", "A_lines")
    if show_valority:
        positions_table("Portefeuille 2 — Valority", "B_lines")

    st.subheader("Composition du portefeuille")

    def _render_portfolio_pie(port_key: str, title: str):
        if not MATPLOTLIB_AVAILABLE:
            st.warning(f"{title} : Camembert indisponible ({MATPLOTLIB_ERROR}).")
            return
        df_positions = build_positions_dataframe(port_key)
        if df_positions.empty:
            st.info(f"{title} : Données indisponibles.")
            return
        df_pie = _prepare_pie_df(df_positions)
        if df_pie.empty:
            st.info(f"{title} : Données indisponibles.")
            return
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.pie(
            df_pie["Valeur actuelle €"],
            labels=df_pie["Nom"],
            autopct="%1.1f%%",
        )
        ax.set_title(title)
        st.pyplot(fig)
        plt.close(fig)
        st.dataframe(
            df_pie[["Nom", "Valeur actuelle €", "Part %"]].style.format(
                {
                    "Valeur actuelle €": to_eur,
                    "Part %": "{:,.2f}%".format,
                }
            ),
            hide_index=True,
            use_container_width=True,
        )

    if show_client and show_valority:
        col_a, col_b = st.columns(2)
        with col_a:
            _render_portfolio_pie("A_lines", "Portefeuille Client")
        with col_b:
            _render_portfolio_pie("B_lines", "Portefeuille Valority")
    elif show_client:
        _render_portfolio_pie("A_lines", "Portefeuille Client")
    elif show_valority:
        _render_portfolio_pie("B_lines", "Portefeuille Valority")

    # APP – Composition
    def _wrap_label_app(label: str, width: int = 28) -> str:
        if not label:
            return "—"
        return "\n".join(textwrap.wrap(str(label), width=width)) or str(label)

    def _render_valority_composition_section():
        if not MATPLOTLIB_AVAILABLE:
            st.warning(f"Valority : graphique indisponible ({MATPLOTLIB_ERROR}).")
            return
        df_positions = build_positions_dataframe("B_lines")
        if df_positions.empty:
            st.info("Aucune donnée pour le portefeuille Valority.")
            return

        df = df_positions.copy()
        total_val = df["Valeur actuelle €"].sum()
        if total_val > 0:
            df["Poids %"] = df["Valeur actuelle €"] / total_val * 100.0
        else:
            total_net = df["Net investi €"].sum()
            if total_net > 0:
                df["Poids %"] = df["Net investi €"] / total_net * 100.0
            else:
                df["Poids %"] = 0.0
                if len(df) > 0:
                    df.loc[df.index[0], "Poids %"] = 100.0
        df = df.sort_values("Poids %", ascending=False)

        if len(df) > 8:
            df_main = df.iloc[:8].copy()
            df_other = df.iloc[8:]
            other_row = pd.DataFrame(
                {
                    "Nom": ["Autres"],
                    "ISIN / Code": ["—"],
                    "Date d'achat": ["—"],
                    "Net investi €": [df_other["Net investi €"].sum()],
                    "Valeur actuelle €": [df_other["Valeur actuelle €"].sum()],
                    "Perf €": [df_other["Perf €"].sum()],
                    "Perf %": [np.nan],
                    "Poids %": [df_other["Poids %"].sum()],
                }
            )
            df = pd.concat([df_main, other_row], ignore_index=True)

        if len(df) >= 2:
            fig, ax = plt.subplots(figsize=(5.2, 3.2))
            wedges, _ = ax.pie(
                df["Poids %"],
                startangle=90,
                labels=None,
                wedgeprops=dict(width=0.35, edgecolor="white"),
            )
            labels = [
                f"{_wrap_label_app(nm)} ({pct:.1f}%)"
                for nm, pct in zip(df["Nom"], df["Poids %"])
            ]
            ax.legend(
                wedges,
                labels,
                loc="center left",
                bbox_to_anchor=(1.02, 0.5),
                frameon=False,
                fontsize=8,
            )
            ax.set_aspect("equal")
            fig.tight_layout(rect=[0, 0, 0.78, 1])
            st.pyplot(fig)
            plt.close(fig)
        else:
            st.info("Portefeuille concentré : 100% sur une seule ligne.")

        df_table = df[["Nom", "ISIN / Code", "Poids %", "Net investi €", "Valeur actuelle €"]]
        st.dataframe(
            df_table.style.format(
                {
                    "Poids %": "{:,.2f}%".format,
                    "Net investi €": to_eur,
                    "Valeur actuelle €": to_eur,
                }
            ),
            hide_index=True,
            use_container_width=True,
        )

    if show_valority:
        st.subheader("Composition du portefeuille (Valority)")
        _render_valority_composition_section()

    with st.expander("Aide rapide"):
        st.markdown(
            """
- Dans chaque portefeuille, vous pouvez **soit** ajouter des *fonds recommandés* (onglet dédié),
  **soit** utiliser la *saisie libre* avec ISIN / code.
- Pour le **fonds en euros**, utilisez le symbole **EUROFUND** (taux paramétrable dans la barre de gauche).
- Les frais d’entrée s’appliquent à chaque investissement.
- Le **rendement total** est la performance globale depuis l’origine (valeur actuelle / net investi).
- Le **rendement annualisé** utilise le XIRR (prise en compte des dates et montants).
- En mode **Personnalisé**, vous pouvez affecter précisément les versements mensuels et ponctuels à chaque ligne,
  avec un contrôle automatique de cohérence par rapport aux montants bruts saisis.
            """
        )

    # ------------------------------------------------------------
    # Analyse interne — Corrélation & volatilité (réservé conseiller)
    # ------------------------------------------------------------
    st.markdown("---")
    with st.expander("🔒 Analyse interne — Corrélation, volatilité et profil de risque", expanded=False):
        st.caption(
            "Section réservée au conseiller : analyse technique basée sur les valeurs liquidatives "
            "(corrélations, volatilités, drawdown)."
        )

        euro_rate = st.session_state.get("EURO_RATE_PREVIEW", 2.0)
        linesA = st.session_state.get("A_lines", [])
        linesB = st.session_state.get("B_lines", [])

        # Portefeuille Client
        if show_client:
            st.markdown("### Portefeuille 1 — Client")
            corrA = correlation_matrix_from_lines(linesA, euro_rate)
            volA = volatility_table_from_lines(linesA, euro_rate)
            riskA = portfolio_risk_stats(linesA, euro_rate)

            if corrA.empty and volA.empty:
                st.info("Pas assez d'historique ou de lignes pour analyser ce portefeuille.")
            else:
                if riskA is not None:
                    c1, c2 = st.columns(2)
                    with c1:
                        st.metric(
                            "Volatilité annuelle estimée",
                            f"{riskA['vol_ann_pct']:.2f} %",
                        )
                    with c2:
                        st.metric(
                            "Max drawdown (historique sur la période)",
                            f"{riskA['max_dd_pct']:.2f} %",
                        )

                if not volA.empty:
                    st.markdown("**Volatilité par ligne**")
                    st.dataframe(
                        volA.style.format(
                            {
                                "Écart-type quotidien %": "{:,.2f}%".format,
                                "Volatilité annuelle %": "{:,.2f}%".format,
                            }
                        ),
                        use_container_width=True,
                    )

                if not corrA.empty:
                    chartA = _corr_heatmap_chart(corrA, "Corrélation des lignes — Portefeuille Client")
                    if chartA is not None:
                        st.altair_chart(chartA, use_container_width=True)

        if show_client and show_valority:
            st.markdown("---")

        if show_valority:
            st.markdown("### Portefeuille 2 — Valority")
            corrB = correlation_matrix_from_lines(linesB, euro_rate)
            volB = volatility_table_from_lines(linesB, euro_rate)
            riskB = portfolio_risk_stats(linesB, euro_rate)

            if corrB.empty and volB.empty:
                st.info("Pas assez d'historique ou de lignes pour analyser ce portefeuille.")
            else:
                if riskB is not None:
                    c1, c2 = st.columns(2)
                    with c1:
                        st.metric(
                            "Volatilité annuelle estimée",
                            f"{riskB['vol_ann_pct']:.2f} %",
                        )
                    with c2:
                        st.metric(
                            "Max drawdown (historique sur la période)",
                            f"{riskB['max_dd_pct']:.2f} %",
                        )

                if not volB.empty:
                    st.markdown("**Volatilité par ligne**")
                    st.dataframe(
                        volB.style.format(
                            {
                                "Écart-type quotidien %": "{:,.2f}%".format,
                                "Volatilité annuelle %": "{:,.2f}%".format,
                            }
                        ),
                        use_container_width=True,
                    )

                if not corrB.empty:
                    chartB = _corr_heatmap_chart(corrB, "Corrélation des lignes — Portefeuille Valority")
                    if chartB is not None:
                        st.altair_chart(chartB, use_container_width=True)



def run_comparator():
    render_app(run_page_config=False)


def run_perfect_portfolio():
    render_portfolio_builder()


def render_mode_router():
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    mode = st.radio(
        "Mode",
        ["Comparer des portefeuilles", "Créer le portefeuille parfait"],
        horizontal=True,
    )
    if mode == "Comparer des portefeuilles":
        run_comparator()
    else:
        run_perfect_portfolio()


def _render_with_crash_shield():
    try:
        render_mode_router()
        st.session_state["APP_STATUS"] = "OK"
    except Exception as e:
        st.session_state["APP_STATUS"] = "KO"
        st.session_state["LAST_EXCEPTION"] = str(e)
        st.title(APP_TITLE)
        st.info("App chargée, statut KO")
        st.error("Une erreur est survenue pendant le rendu.")
        st.exception(e)
        st.markdown("""
Conseils :
- Vérifiez vos dépendances (reportlab/matplotlib).
- Vérifiez la clé EODHD dans les secrets.
- Réessayez après avoir vidé le cache Streamlit.
""")


_render_with_crash_shield()
