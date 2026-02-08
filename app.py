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
    if after.empty:
        return float(df.iloc[-1]["Close"])
    return float(after.iloc[0]["Close"])


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

        df, _, _ = get_price_series(ln.get("isin") or ln.get("name"), None, euro_rate)
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
# Metrics ligne
# ------------------------------------------------------------

def compute_line_metrics(
    ln: Dict[str, Any],
    fee_pct: float,
    euro_rate: float,
) -> Tuple[float, float, float]:
    gross = float(ln.get("amount_gross", 0.0))
    net = gross * (1.0 - fee_pct / 100.0)
    d_buy = pd.Timestamp(ln["buy_date"])

    if str(ln.get("isin", "")).strip().upper() == "STRUCTURED":
        df = structured_series(
            d_buy,
            TODAY,
            float(ln.get("struct_rate", 8.0)),
            int(ln.get("struct_years", 6)),
        )
    else:
        df, _, _ = get_price_series(ln.get("isin") or ln.get("name"), None, euro_rate)

    if df.empty:
        buy_px = 1.0
    else:
        buy_px = _get_close_on(df, d_buy)

    manual = ln.get("buy_px", None)
    if manual not in (None, "", 0, "0"):
        buy_px = float(manual)

    qty = net / buy_px if buy_px > 0 else 0.0
    return net, buy_px, qty


# ------------------------------------------------------------
# Calc portefeuille
# ------------------------------------------------------------

def compute_portfolio(
    lines: List[Dict[str, Any]],
    fee_pct: float,
    euro_rate: float,
    monthly_amt_gross: float,
    one_amt_gross: float,
    one_date: pd.Timestamp,
    alloc_mode: str,
    custom_weights_monthly: Optional[Dict[int, float]],
    custom_weights_oneoff: Optional[Dict[int, float]],
    single_target: Optional[int],
    portfolio_label: str,
) -> Tuple[pd.DataFrame, float, float, float, Optional[float], pd.Timestamp, pd.Timestamp]:

    price_map: Dict[int, pd.Series] = {}
    eff_buy_date: Dict[int, pd.Timestamp] = {}
    buy_price_used: Dict[int, float] = {}

    invalid_found = False
    date_warnings = st.session_state.setdefault("DATE_WARNINGS", [])

    for ln in lines:
        key_id = id(ln)
        isin_or_name = ln.get("isin") or ln.get("name")

        # 🔹 CAS PRODUIT STRUCTURÉ (série synthétique)
        if str(isin_or_name).strip().upper() == "STRUCTURED":
            d_buy = pd.Timestamp(ln["buy_date"])
            df_full = structured_series(
                start=d_buy,
                end=TODAY,
                annual_rate_pct=float(ln.get("struct_rate", 8.0)),
                redemption_years=int(ln.get("struct_years", 6)),
            )
            sym = "STRUCTURED"
        else:
            df_full, sym, _ = get_price_series(isin_or_name, None, euro_rate)

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
                dfl, _, _ = get_price_series(line.get("isin") or line.get("name"), None, euro_rate)
                last = float(dfl["Close"].iloc[-1]) if not dfl.empty else np.nan
                st.markdown(f"VL actuelle : **{to_eur(last)}**")
            except Exception:
                st.markdown("VL actuelle : —")
        with cols[4]:
            if not st.session_state[state_key]:
                if st.button("✏️", key=f"edit_{port_key}_{idx}", help="Modifier"):
                    st.session_state[state_key] = True
                    st.experimental_rerun()
                if st.button("❌", key=f"del_{port_key}_{idx}", help="Supprimer"):
                    del st.session_state[port_key][idx]
                    st.experimental_rerun()
            else:
                if st.button("💾", key=f"save_{port_key}_{idx}", help="Enregistrer"):
                    st.session_state[state_key] = False
                    st.experimental_rerun()
                if st.button("✖️", key=f"cancel_{port_key}_{idx}", help="Annuler"):
                    st.session_state[state_key] = False
                    st.experimental_rerun()

    if st.session_state[state_key]:
        with st.form(key=f"edit_form_{port_key}_{idx}"):
            c1, c2 = st.columns(2)
            with c1:
                name = st.text_input("Nom", value=line.get("name", ""))
                isin = st.text_input("ISIN / Code", value=line.get("isin", ""))
            with c2:
                amount = st.text_input("Montant (brut) €", value=str(line.get("amount_gross", 0.0)))
                buy_date = st.date_input("Date d'achat", value=pd.Timestamp(line["buy_date"]).date())
            px = st.text_input("VL d'achat (optionnel)", value=str(line.get("buy_px", "")))
            note = st.text_input("Note", value=line.get("note", ""))
            submitted = st.form_submit_button("Sauvegarder")
            if submitted:
                try:
                    line["name"] = name.strip()
                    line["isin"] = isin.strip()
                    line["amount_gross"] = float(str(amount).replace(" ", "").replace(",", "."))
                    line["buy_date"] = pd.Timestamp(buy_date)
                    line["buy_px"] = float(str(px).replace(",", ".")) if px.strip() else ""
                    line["note"] = note.strip()
                    st.session_state[state_key] = False
                    st.experimental_rerun()
                except Exception as e:
                    st.error(f"Erreur lors de la sauvegarde : {e}")


def simple_positions_summary(port_key: str) -> pd.DataFrame:
    fee_pct = st.session_state.get("FEE_A", 0.0) if port_key == "A_lines" else st.session_state.get("FEE_B", 0.0)
    euro_rate = st.session_state.get("EURO_RATE_PREVIEW", 2.0)
    lines = st.session_state.get(port_key, [])
    rows: List[Dict[str, Any]] = []

    for ln in lines:
        net_amt, buy_px, qty = compute_line_metrics(ln, fee_pct, euro_rate)

        dfl, _, _ = get_price_series(ln.get("isin") or ln.get("name"), None, euro_rate)
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
        dfl, _, _ = get_price_series(ln.get("isin") or ln.get("name"), buy_ts, euro_rate)

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
        dfl, _, _ = get_price_series(ln.get("isin") or ln.get("name"), buy_ts, euro_rate)

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

        df, _, _ = get_price_series(ln.get("isin") or ln.get("name"), None, euro_rate)
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
    min_points: int = 30,
) -> pd.DataFrame:
    """
    Matrice de corrélation (retours quotidiens).
    """
    returns = _build_returns_df(lines, euro_rate, years, min_points)
    if returns.empty or returns.shape[1] < 2:
        return pd.DataFrame()
    return returns.corr()


def volatility_table_from_lines(
    lines: List[Dict[str, Any]],
    euro_rate: float,
    years: int = 3,
    min_points: int = 60,
) -> pd.DataFrame:
    """
    Tableau avec nom, écart-type quotidien %, volatilité annualisée %.
    """
    returns = _build_returns_df(lines, euro_rate, years, min_points)
    if returns.empty:
        return pd.DataFrame()

    rows: List[Dict[str, Any]] = []
    for col in returns.columns:
        std_daily = returns[col].std()
        vol_annual = std_daily * np.sqrt(252.0)
        rows.append(
            {
                "Nom / ISIN": col,
                "Écart-type quotidien %": std_daily * 100.0,
                "Volatilité annuelle %": vol_annual * 100.0,
            }
        )
    return pd.DataFrame(rows)


def portfolio_risk_stats(
    lines: List[Dict[str, Any]],
    euro_rate: float,
    years: int = 3,
) -> Optional[Dict[str, float]]:
    """
    Calcule la volatilité annuelle et le max drawdown global d'un portefeuille.
    On suppose poids équipondérés.
    """
    returns = _build_returns_df(lines, euro_rate, years)
    if returns.empty:
        return None

    n_cols = returns.shape[1]
    if n_cols == 0:
        return None

    equal_w = np.ones(n_cols) / n_cols
    portfolio_returns = returns.dot(equal_w)
    vol_ann = portfolio_returns.std() * np.sqrt(252.0)

    cum = (1.0 + portfolio_returns).cumprod()
    running_max = cum.cummax()
    dd = cum / running_max - 1.0
    max_dd = float(dd.min())

    return {
        "vol_ann_pct": vol_ann * 100.0,
        "max_dd_pct": max_dd * 100.0,
    }


def _corr_heatmap_chart(corr: pd.DataFrame, title: str) -> Optional[alt.Chart]:
    """
    Heatmap corrélation avec Altair.
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
            f"Date d'investissement initiale : {pd.Timestamp(buy_date).strftime('%d/%m/%Y')}"
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
        st.caption(f"Date d'achat (versement initial) : {pd.Timestamp(buy_date).strftime('%d/%m/%Y')}")

    px = st.text_input("Prix d'achat (optionnel)", value="", key=f"reco_px_{port_key}")

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
                f"Date d'achat (versement initial) : "
                f"{pd.Timestamp(buy_date_central).strftime('%d/%m/%Y')}"
            )

        px = st.text_input("Prix d'achat (optionnel)", value="")
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
    if best_score is None:
        best_combo = candidates[:k]
    return best_combo


def _greedy_select(
    candidates: List[str],
    returns: pd.DataFrame,
    k: int,
    forced: Optional[str] = None,
    corr_penalty: float = 0.6,
) -> List[str]:
    if not candidates or k <= 0:
        return []
    if k >= len(candidates):
        return candidates
    corr = returns.corr()
    ann_return, ann_vol = _annualized_stats(returns)
    selected: List[str] = []
    pool = list(candidates)
    if forced and forced in pool:
        selected.append(forced)
        pool.remove(forced)
        k -= 1
    while len(selected) < k and pool:
        best_score = None
        best_candidate = None
        for cand in pool:
            if cand in selected:
                continue
            subset = selected + [cand]
            score_sharpe = (
                (ann_return[cand] / ann_vol[cand])
                if ann_vol[cand] > 0
                else 0.0
            )
            if len(selected) > 0:
                sub_corr = corr.loc[subset, subset]
                avg_c = _avg_offdiag_corr(sub_corr)
                score = score_sharpe - corr_penalty * avg_c
            else:
                score = score_sharpe
            if best_score is None or score > best_score:
                best_score = score
                best_candidate = cand
        if best_candidate:
            selected.append(best_candidate)
            pool.remove(best_candidate)
        else:
            break
    if len(selected) < k:
        remaining = [c for c in pool if c not in selected]
        selected.extend(remaining[: k - len(selected)])
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
    st.title("Créer le portefeuille parfait")
    try:
        st.markdown("### Configuration du portefeuille")
        
        # ✅ MODIFICATION 1 : Profil de risque
        profile_map = {
            "Prudent": 50,
            "Équilibré": 30,
            "Dynamique": 20,
            "Agressif": 10,
        }
        profile = st.selectbox("Profil de risque", list(profile_map.keys()))
        euro_pct = profile_map[profile]
        st.caption(f"Fonds en euros obligatoire : {euro_pct}% du portefeuille.")

        total_budget = st.number_input(
            "Budget total (€)",
            min_value=0,
            max_value=10_000_000,
            value=100_000,
            step=1_000,
        )

        opt_window_mode = st.radio(
            "Fenêtre d'analyse",
            ["1 an", "3 ans", "5 ans", "10 ans", "Dates personnalisées"],
            horizontal=True,
        )
        if opt_window_mode == "Dates personnalisées":
            date_cols = st.columns(2)
            with date_cols[0]:
                opt_start_date = st.date_input(
                    "Date de début",
                    value=(TODAY - pd.DateOffset(years=3)).date(),
                    key="OPT_START_DATE",
                )
            with date_cols[1]:
                opt_end_date = st.date_input(
                    "Date de fin",
                    value=TODAY.date(),
                    key="OPT_END_DATE",
                )
            opt_start = pd.Timestamp(opt_start_date)
            opt_end = pd.Timestamp(opt_end_date)
        else:
            years_map = {"1 an": 1, "3 ans": 3, "5 ans": 5, "10 ans": 10}
            opt_years = years_map[opt_window_mode]
            opt_start = TODAY - pd.DateOffset(years=opt_years)
            opt_end = TODAY

        if opt_start > opt_end:
            st.warning("La date de début doit être antérieure à la date de fin.")
            return

        # ✅ MODIFICATION 2 : Taux fonds euros (pas de volatilité)
        euro_rate = st.number_input("Taux fonds euros (%)", min_value=0.0, max_value=10.0, value=2.0, step=0.10)
        # La volatilité du fonds euros est maintenant fixée à 0% (pas d'input utilisateur)
        euro_vol = 0.0  # ✅ Volatilité du fonds en euros fixée à 0%

        action_count = st.selectbox("Nombre de fonds actions", [2, 3, 4, 5, 6], index=1)
        # ✅ MODIFICATION 3 : Permettre 0 fonds obligataires
        bond_count = st.selectbox("Nombre de fonds obligataires UC", [0, 1, 2, 3], index=1)
        max_weight_pct = 25
        objective_choice = st.selectbox(
            "Objectif d'optimisation",
            [
                "Maximiser le ratio de Sharpe (UC)",
                "Minimiser la volatilité (UC)",
                "Maximiser le rendement annualisé (UC)",
                "Meilleur compromis rendement/risque (UC)",
                "Diversification maximale",
                "Risk Parity (UC)",
            ],
        )
        objective_desc = {
            "Maximiser le ratio de Sharpe (UC)": "Optimise rendement/risque avec Sharpe sur les UC.",
            "Minimiser la volatilité (UC)": "Recherche la volatilité la plus faible sur les UC.",
            "Maximiser le rendement annualisé (UC)": "Priorise le rendement annualisé des UC.",
            "Meilleur compromis rendement/risque (UC)": "Sharpe avec pénalités de corrélation/concentration.",
            "Diversification maximale": "Minimise corrélation moyenne et concentration.",
            "Risk Parity (UC)": "Équilibre les contributions au risque des UC.",
        }
        st.caption(objective_desc.get(objective_choice, ""))

        force_fund = st.checkbox("Forcer un fonds croissance (anchor)", value=False)
        forced_isin = None
        if force_fund:
            force_options = [_safe_fund_label(name, isin) for name, isin in RECO_FUNDS_CORE]
            force_lookup = {label: isin for label, (_, isin) in zip(force_options, RECO_FUNDS_CORE)}
            forced_choice = st.selectbox("Fonds croissance imposé", force_options)
            forced_isin = force_lookup.get(forced_choice)

        extra_actions_input = st.text_input("Ajouter un ISIN actions externe (optionnel)", value="")
        extra_bonds_input = st.text_input("Ajouter un ISIN obligataire externe (optionnel)", value="")
        extra_actions = [x.strip() for x in extra_actions_input.split(",") if x.strip()]
        extra_bonds = [x.strip() for x in extra_bonds_input.split(",") if x.strip()]

        # ✅ MODIFICATION 4 : Bouton pour lancer les calculs
        st.markdown("---")
        run_optimization = st.button("🚀 Créer le portefeuille parfait", type="primary", use_container_width=True)
        
        # Ne continuer que si le bouton a été cliqué
        if not run_optimization:
            st.info("👆 Configurez les paramètres ci-dessus puis cliquez sur le bouton pour générer votre portefeuille optimal.")
            return

        # Le reste du code ne s'exécute que si le bouton est cliqué
        actions_universe = [isin for _, isin in RECO_FUNDS_CORE] + extra_actions
        bonds_universe = [isin for _, isin in RECO_FUNDS_DEF if isin != "EUROFUND"] + extra_bonds
        all_candidates = sorted(set(actions_universe + bonds_universe))

        if not all_candidates:
            st.info("Aucun fonds UC disponible dans l'univers.")
            return

        returns_all, status_all = _returns_for_isins(all_candidates, opt_start, opt_end, euro_rate=euro_rate)
        insufficient = [isin for isin, status in status_all.items() if status != "ok"]
        if insufficient:
            st.warning("Certains fonds n'ont pas assez d'historique et ont été exclus.")

        valid_actions = [isin for isin in actions_universe if status_all.get(isin) == "ok"]
        valid_bonds = [isin for isin in bonds_universe if status_all.get(isin) == "ok"]

        if forced_isin and forced_isin not in valid_actions:
            st.warning("Le fonds imposé n'a pas assez d'historique et a été exclu.")
            forced_isin = None

        if action_count > len(valid_actions):
            st.warning("Nombre de fonds actions réduit faute d'historique suffisant.")
            action_count = len(valid_actions)
        if bond_count > len(valid_bonds):
            st.warning("Nombre de fonds obligataires réduit faute d'historique suffisant.")
            bond_count = len(valid_bonds)
        if action_count + bond_count == 0:
            st.info("Pas assez de fonds valides. Élargissez la période ou l'univers.")
            return

        action_returns = returns_all[valid_actions] if valid_actions and not returns_all.empty else pd.DataFrame()
        bond_returns = returns_all[valid_bonds] if valid_bonds and not returns_all.empty else pd.DataFrame()
        action_ann_return, action_ann_vol = _annualized_stats(action_returns)
        bond_ann_return, bond_ann_vol = _annualized_stats(bond_returns)

        if objective_choice == "Maximiser le rendement annualisé (UC)":
            selected_actions = action_ann_return.sort_values(ascending=False).index.tolist()[:action_count]
            selected_bonds = bond_ann_return.sort_values(ascending=False).index.tolist()[:bond_count] if bond_count > 0 else []
        elif objective_choice == "Minimiser la volatilité (UC)":
            selected_actions = action_ann_vol.sort_values().index.tolist()[:action_count]
            selected_bonds = bond_ann_vol.sort_values().index.tolist()[:bond_count] if bond_count > 0 else []
        elif objective_choice == "Diversification maximale":
            selected_actions = _select_min_corr_subset(valid_actions, action_returns, action_count, anchor=forced_isin)
            selected_bonds = _select_min_corr_subset(valid_bonds, bond_returns, bond_count) if bond_count > 0 else []
        elif objective_choice == "Risk Parity (UC)":
            selected_actions = _select_min_corr_subset(valid_actions, action_returns, action_count, anchor=forced_isin)
            selected_bonds = _select_min_corr_subset(valid_bonds, bond_returns, bond_count) if bond_count > 0 else []
        elif objective_choice == "Meilleur compromis rendement/risque (UC)":
            selected_actions = _greedy_select(
                valid_actions,
                action_returns,
                action_count,
                forced=forced_isin,
                corr_penalty=0.8,
            )
            selected_bonds = _greedy_select(
                valid_bonds,
                bond_returns,
                bond_count,
                corr_penalty=0.8,
            ) if bond_count > 0 else []
        else:
            selected_actions = _greedy_select(
                valid_actions,
                action_returns,
                action_count,
                forced=forced_isin,
                corr_penalty=0.6,
            )
            selected_bonds = _greedy_select(
                valid_bonds,
                bond_returns,
                bond_count,
                corr_penalty=0.6,
            ) if bond_count > 0 else []

        if forced_isin and forced_isin in valid_actions and forced_isin not in selected_actions:
            selected_actions = [forced_isin] + [isin for isin in selected_actions if isin != forced_isin]
            selected_actions = selected_actions[:action_count]
        selected_isins = [isin for isin in selected_actions + selected_bonds if isin]
        if not selected_isins:
            st.info("Aucun fonds disponible pour l'optimisation.")
            return

        returns_selected = returns_all[selected_isins] if not returns_all.empty else pd.DataFrame()
        if returns_selected.empty:
            st.info("Historique insuffisant pour l'optimisation. Élargissez la période.")
            return

        uc_total = max(0.0, 1.0 - euro_pct / 100.0)
        if uc_total <= 0:
            st.info("La part UC est nulle, ajustez le profil de risque.")
            return

        max_w = max_weight_pct / 100.0
        uc_max_bound = min(max_w / uc_total, 1.0)

        def _cap_normalize(weights: Dict[str, float]) -> Dict[str, float]:
            capped = _apply_weight_caps(weights, uc_max_bound) if weights else {}
            total = sum(capped.values())
            return {k: v / total for k, v in capped.items()} if total > 0 else {}

        def _risk_parity_weights(cov: pd.DataFrame) -> Dict[str, float]:
            if cov.empty:
                return {}
            vols = np.sqrt(np.diag(cov))
            inv_vol = np.where(vols > 0, 1.0 / vols, 0.0)
            weights = inv_vol / inv_vol.sum() if inv_vol.sum() > 0 else np.ones_like(inv_vol) / len(inv_vol)
            w = pd.Series(weights, index=cov.columns)
            for _ in range(20):
                port_var = float(w.T @ cov @ w)
                if port_var <= 0:
                    break
                mrc = cov @ w
                rc = w * mrc / np.sqrt(port_var)
                target = rc.mean()
                adj = target / rc.replace(0, np.nan)
                w = w * adj.fillna(1.0)
                w = w / w.sum()
            return w.to_dict()

        ann_return, ann_vol = _annualized_stats(returns_selected)
        corr = returns_selected.corr()
        cov = returns_selected.cov()
        weights_uc_raw: Dict[str, float] = {}

        if objective_choice == "Maximiser le ratio de Sharpe (UC)":
            weights_uc_raw = _optimize_uc_weights(
                returns_selected,
                "max_sharpe",
                0.0,
                uc_max_bound,
                None,
                None,
            )
        elif objective_choice == "Minimiser la volatilité (UC)":
            if PYPFOPT_AVAILABLE:
                weights_uc_raw = _optimize_uc_weights(
                    returns_selected,
                    "target_vol",
                    0.0,
                    uc_max_bound,
                    ann_vol.min() if not ann_vol.empty else None,
                    None,
                )
            else:
                score = (1.0 / ann_vol.replace(0, np.nan)).fillna(0.0)
                if score.sum() > 0:
                    weights_uc_raw = (score / score.sum()).to_dict()
        elif objective_choice == "Maximiser le rendement annualisé (UC)":
            score = ann_return.clip(lower=0.0)
            if score.sum() > 0:
                weights_uc_raw = (score / score.sum()).to_dict()
        elif objective_choice == "Meilleur compromis rendement/risque (UC)":
            avg_corr = corr.mean().fillna(0.0)
            base_score = (ann_return / ann_vol.replace(0, np.nan)).fillna(0.0)
            score = (base_score - 0.4 * avg_corr).clip(lower=0.0)
            if score.sum() > 0:
                weights_uc_raw = (score / score.sum()).to_dict()
                concentration_penalty = sum(v ** 2 for v in weights_uc_raw.values())
                if concentration_penalty > 0:
                    weights_uc_raw = {k: v / concentration_penalty for k, v in weights_uc_raw.items()}
                    total = sum(weights_uc_raw.values())
                    if total > 0:
                        weights_uc_raw = {k: v / total for k, v in weights_uc_raw.items()}
        elif objective_choice == "Diversification maximale":
            avg_corr = corr.mean().fillna(0.0)
            score = (1.0 - avg_corr).clip(lower=0.0)
            if score.sum() > 0:
                weights_uc_raw = (score / score.sum()).to_dict()
        elif objective_choice == "Risk Parity (UC)":
            weights_uc_raw = _risk_parity_weights(cov)

        if not weights_uc_raw:
            weights_uc_raw = {isin: 1.0 / len(selected_isins) for isin in selected_isins}

        weights_uc_raw = _cap_normalize(weights_uc_raw)
        weights_uc = {k: v * uc_total for k, v in weights_uc_raw.items()}

        corr = returns_selected.corr()
        redundant_pairs = []
        for i, isin_a in enumerate(selected_isins):
            for isin_b in selected_isins[i + 1 :]:
                c = corr.loc[isin_a, isin_b]
                if c > 0.85:
                    redundant_pairs.append((isin_a, isin_b, c))
        if redundant_pairs:
            st.warning(
                "Certains fonds présentent une corrélation > 0.85 (redondance possible) : "
                + ", ".join(f"{_fund_name(a)} / {_fund_name(b)} ({c:.2f})" for a, b, c in redundant_pairs)
            )

        euro_amount = int(round(total_budget * (1.0 - uc_total)))
        uc_budget = int(total_budget - euro_amount)
        uc_amounts = {isin: weights_uc.get(isin, 0.0) * uc_budget for isin in selected_isins}
        uc_amounts = _round_allocations(uc_amounts)
        remainder = uc_budget - sum(uc_amounts.values())

        rows: List[Dict[str, Any]] = []
        rows.append(
            {
                "Nom": "Fonds en euros (EUROFUND)",
                "ISIN": "EUROFUND",
                "Catégorie": "Fonds en euros",
                "Poids %": (1.0 - uc_total) * 100.0,
                "Montant €": euro_amount,
            }
        )
        for isin in selected_isins:
            cat = "Actions UC" if isin in selected_actions else "Obligataires UC"
            rows.append(
                {
                    "Nom": _fund_name(isin),
                    "ISIN": isin,
                    "Catégorie": cat,
                    "Poids %": weights_uc.get(isin, 0.0) * 100.0,
                    "Montant €": uc_amounts.get(isin, 0),
                }
            )
        df_alloc = pd.DataFrame(rows)

        st.markdown("**Allocation finale**")
        st.dataframe(
            df_alloc.style.format(
                {
                    "Poids %": "{:,.2f}%".format,
                    "Montant €": to_eur,
                }
            ),
            use_container_width=True,
            hide_index=True,
        )
        st.caption(f"Reste non alloué (UC) : {to_eur(remainder)}")

        if MATPLOTLIB_AVAILABLE:
            fig, ax = plt.subplots(figsize=(4.4, 3.4))
            comp_labels = ["Fonds en euros", "Obligataires UC", "Actions UC"]
            comp_values = [
                euro_amount,
                sum(uc_amounts.get(i, 0) for i in selected_bonds),
                sum(uc_amounts.get(i, 0) for i in selected_actions),
            ]
            ax.pie(comp_values, labels=comp_labels, autopct="%1.1f%%")
            ax.set_title("Composition globale")
            st.pyplot(fig)
            plt.close(fig)

            fig, ax = plt.subplots(figsize=(4.4, 3.4))
            ax.pie(
                [row["Montant €"] for row in rows],
                labels=[row["Nom"] for row in rows],
                autopct="%1.1f%%",
            )
            ax.set_title("Répartition par ligne")
            st.pyplot(fig)
            plt.close(fig)
        else:
            st.warning(f"Graphique indisponible ({MATPLOTLIB_ERROR}).")

        uc_weights_norm = {k: (weights_uc.get(k, 0.0) / uc_total) for k in selected_isins if uc_total > 0}
        w_vec = np.array([uc_weights_norm.get(k, 0.0) for k in returns_selected.columns])
        port_ret = returns_selected.dot(w_vec)
        ann_ret = float(port_ret.mean() * 252.0)
        ann_vol = float(port_ret.std() * np.sqrt(252.0))
        sharpe = ann_ret / ann_vol if ann_vol > 0 else np.nan

        kpi_cols = st.columns(3)
        with kpi_cols[0]:
            st.metric("Rendement annualisé (UC)", fmt_pct_fr(ann_ret * 100))
        with kpi_cols[1]:
            st.metric("Volatilité annualisée (UC)", fmt_pct_fr(ann_vol * 100))
        with kpi_cols[2]:
            st.metric("Sharpe (UC)", f"{sharpe:.2f}" if sharpe == sharpe else "—")

        euro_return = euro_rate / 100.0
        euro_vol_ann = euro_vol / 100.0  # Toujours 0% maintenant
        g_ann_ret = uc_total * ann_ret + (1.0 - uc_total) * euro_return
        g_ann_vol = np.sqrt((uc_total * ann_vol) ** 2 + ((1.0 - uc_total) * euro_vol_ann) ** 2)
        g_sharpe = g_ann_ret / g_ann_vol if g_ann_vol > 0 else np.nan
        st.caption(
            f"Global (incl. fonds euros) — Rendement: {fmt_pct_fr(g_ann_ret * 100)} | "
            f"Volatilité: {fmt_pct_fr(g_ann_vol * 100)} | Sharpe: {g_sharpe:.2f}"
            if g_sharpe == g_sharpe
            else f"Global (incl. fonds euros) — Rendement: {fmt_pct_fr(g_ann_ret * 100)} | "
            f"Volatilité: {fmt_pct_fr(g_ann_vol * 100)} | Sharpe: —"
        )

        st.markdown("**Heatmap corrélation (UC)**")
        if returns_selected.shape[1] >= 2:
            df_corr = corr.copy()
            df_corr["Ligne1"] = df_corr.index
            df_melt = df_corr.melt(id_vars="Ligne1", var_name="Ligne2", value_name="corr")
            heat = (
                alt.Chart(df_melt)
                .mark_rect()
                .encode(
                    x=alt.X("Ligne1:O", sort=None, title=""),
                    y=alt.Y("Ligne2:O", sort=None, title=""),
                    color=alt.Color("corr:Q", scale=alt.Scale(domain=[-1, 1])),
                    tooltip=[
                        alt.Tooltip("Ligne1:N", title="Ligne 1"),
                        alt.Tooltip("Ligne2:N", title="Ligne 2"),
                        alt.Tooltip("corr:Q", title="Corrélation", format=".2f"),
                    ],
                )
                .properties(height=260)
            )
            st.altair_chart(heat, use_container_width=True)
            st.caption("Corrélation proche de 1 = redondant • proche de 0 = décorrélé.")
        else:
            st.info("Corrélation indisponible (données insuffisantes).")

        st.download_button(
            "📥 Télécharger allocation (CSV)",
            data=df_alloc.to_csv(index=False).encode("utf-8"),
            file_name="allocation_portefeuille.csv",
            mime="text/csv",
        )

        avg_corr_uc = _avg_offdiag_corr(corr)
        st.markdown(
            f"**Fenêtre utilisée** : {fmt_date(opt_start)} → {fmt_date(opt_end)}"
        )
        if insufficient:
            st.markdown(
                "**Fonds exclus (historique insuffisant)** : "
                + ", ".join(insufficient)
            )
        st.info(
            "Pourquoi cette allocation ?\n"
            f"- Objectif appliqué : {objective_choice}\n"
            f"- Corrélation moyenne UC : {avg_corr_uc:.2f}\n"
            "- Contraintes respectées (fonds euros + 25% max par UC)"
        )
    except Exception as exc:
        st.error("Une erreur est survenue dans le builder. L'application reste utilisable.")
        st.exception(exc)


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

        st.session_state["EURO_RATE_PREVIEW"] = (EURO_RATE_A + EURO_RATE_B) / 2.0

        # Date des investissements initiaux (par portefeuille)
        st.header("Dates des investissements initiaux")
        st.session_state["INIT_A_DATE"] = st.date_input(
            "Portefeuille 1 (Client)",
            value=st.session_state.get("INIT_A_DATE", pd.Timestamp("2024-01-02").date()),
            key="INIT_A_DATE",
        )
        st.session_state["INIT_B_DATE"] = st.date_input(
            "Portefeuille 2 (Valority)",
            value=st.session_state.get("INIT_B_DATE", pd.Timestamp("2024-01-02").date()),
            key="INIT_B_DATE",
        )

        st.markdown("---")
        st.header("Frais d'entrée (%)")
        st.session_state["FEE_A"] = st.number_input(
            "Portefeuille 1 (Client)",
            0.0,
            10.0,
            st.session_state.get("FEE_A", 3.0),
            0.1,
            key="FEE_A",
        )
        st.session_state["FEE_B"] = st.number_input(
            "Portefeuille 2 (Valority)",
            0.0,
            10.0,
            st.session_state.get("FEE_B", 2.0),
            0.1,
            key="FEE_B",
        )

        st.markdown("---")

        st.header("Versements mensuels (brut)")
        st.session_state["M_A"] = st.number_input(
            "Portefeuille 1 (Client)",
            0.0,
            1_000_000.0,
            st.session_state.get("M_A", 0.0),
            step=50.0,
            key="M_A",
        )
        st.session_state["M_B"] = st.number_input(
            "Portefeuille 2 (Valority)",
            0.0,
            1_000_000.0,
            st.session_state.get("M_B", 0.0),
            step=50.0,
            key="M_B",
        )

        st.markdown("---")

        st.header("Versement ponctuel (brut)")
        st.session_state["ONE_A"] = st.number_input(
            "Montant (Portefeuille 1)",
            0.0,
            10_000_000.0,
            st.session_state.get("ONE_A", 0.0),
            step=1000.0,
            key="ONE_A",
        )
        st.session_state["ONE_A_DATE"] = st.date_input(
            "Date versement ponctuel (Portefeuille 1)",
            value=st.session_state.get("ONE_A_DATE", pd.Timestamp("2024-07-01").date()),
            key="ONE_A_DATE",
        )

        st.session_state["ONE_B"] = st.number_input(
            "Montant (Portefeuille 2)",
            0.0,
            10_000_000.0,
            st.session_state.get("ONE_B", 0.0),
            step=1000.0,
            key="ONE_B",
        )
        st.session_state["ONE_B_DATE"] = st.date_input(
            "Date versement ponctuel (Portefeuille 2)",
            value=st.session_state.get("ONE_B_DATE", pd.Timestamp("2024-07-01").date()),
            key="ONE_B_DATE",
        )

    # -------------------------------------------------------------------
    # Panneaux d'avertissements (dates invalides)
    # -------------------------------------------------------------------
    warnings = st.session_state.get("DATE_WARNINGS", [])
    if warnings:
        st.warning("⚠️ Dates d'achat antérieures à la première VL")
        with st.expander("Détails des incohérences temporelles", expanded=False):
            for w in warnings:
                st.markdown(f"```\n{w}\n```")
        st.session_state["DATE_WARNINGS"] = []

    # -------------------------------------------------------------------
    # Gestion des lignes (méthode accordéon)
    # -------------------------------------------------------------------

    # Portefeuille 1 — Client
    st.markdown("---")
    st.header("Portefeuille 1 — Client")

    tab1, tab2 = st.tabs(["Fonds recommandés", "Saisie libre (ISIN / Code)"])
    with tab1:
        _add_from_reco_block("A_lines", "Ajouter un fonds recommandé")
    with tab2:
        _add_line_form_free("A_lines", "Saisie libre (ISIN ou code)")

    st.markdown("### Lignes enregistrées")
    linesA = st.session_state.get("A_lines", [])
    if not linesA:
        st.info("Aucune ligne pour l'instant.")
    else:
        for idx, ln in enumerate(linesA):
            _line_card(ln, idx, "A_lines")

    # Portefeuille 2 — Valority
    st.markdown("---")
    st.header("Portefeuille 2 — Valority")

    tab3, tab4 = st.tabs(["Fonds recommandés", "Saisie libre (ISIN / Code)"])
    with tab3:
        _add_from_reco_block("B_lines", "Ajouter un fonds recommandé")
    with tab4:
        _add_line_form_free("B_lines", "Saisie libre (ISIN ou code)")

    st.markdown("### Lignes enregistrées")
    linesB = st.session_state.get("B_lines", [])
    if not linesB:
        st.info("Aucune ligne pour l'instant.")
    else:
        for idx, ln in enumerate(linesB):
            _line_card(ln, idx, "B_lines")

    # -------------------------------------------------------------------
    # Calcul des portefeuilles + affichage synthèse
    # -------------------------------------------------------------------
    st.markdown("---")
    st.header("Résultats globaux — Comparaison")

    # Portefeuille A
    dfA_val, brutA, netA, finalA, irrA, startA_min, startA_full = compute_portfolio(
        linesA,
        st.session_state["FEE_A"],
        st.session_state["EURO_RATE_A"],
        st.session_state["M_A"],
        st.session_state["ONE_A"],
        pd.Timestamp(st.session_state["ONE_A_DATE"]),
        st.session_state.get("ALLOC_MODE", "equal"),
        st.session_state.get("CUSTOM_WEIGHTS_MONTHLY_A"),
        st.session_state.get("CUSTOM_WEIGHTS_ONEOFF_A"),
        st.session_state.get("SINGLE_TARGET_A"),
        "Portefeuille Client",
    )
    # Portefeuille B
    dfB_val, brutB, netB, finalB, irrB, startB_min, startB_full = compute_portfolio(
        linesB,
        st.session_state["FEE_B"],
        st.session_state["EURO_RATE_B"],
        st.session_state["M_B"],
        st.session_state["ONE_B"],
        pd.Timestamp(st.session_state["ONE_B_DATE"]),
        st.session_state.get("ALLOC_MODE", "equal"),
        st.session_state.get("CUSTOM_WEIGHTS_MONTHLY_B"),
        st.session_state.get("CUSTOM_WEIGHTS_ONEOFF_B"),
        st.session_state.get("SINGLE_TARGET_B"),
        "Portefeuille Valority",
    )

    # Carte synthèse
    st.subheader("Synthèse chiffrée — Client vs Valority")

    def _kpi_card(label: str, val: float, brut: float, net: float, irr: Optional[float]):
        st.markdown(f"**{label}**")
        c1, c2, c3, c4, c5 = st.columns(5)
        with c1:
            st.metric("Valeur actuelle", to_eur(val))
        with c2:
            st.metric("Versé (brut)", to_eur(brut))
        with c3:
            st.metric("Net investi", to_eur(net))
        with c4:
            perf_pct = (val / net - 1.0) * 100.0 if net > 0 else np.nan
            st.metric("Rendement total", f"{perf_pct:,.2f} %".replace(",", " ").replace(".", ",") if perf_pct == perf_pct else "—")
        with c5:
            st.metric("Rendement annualisé (XIRR)", f"{irr:,.2f} %".replace(",", " ").replace(".", ",") if irr is not None and irr == irr else "—")

    _kpi_card("Portefeuille Client", finalA, brutA, netA, irrA)
    st.markdown("---")
    _kpi_card("Portefeuille Valority", finalB, brutB, netB, irrB)

    st.markdown("---")
    st.subheader("Comparaison Client vs Valority")
    c1, c2 = st.columns(2)
    with c1:
        delta_val = finalB - finalA
        st.metric("Différence de valeur finale (Valority – Client)", to_eur(delta_val))
    with c2:
        pA = (finalA / netA - 1.0) * 100.0 if netA > 0 else np.nan
        pB = (finalB / netB - 1.0) * 100.0 if netB > 0 else np.nan
        delta_perf = pB - pA if pA == pA and pB == pB else np.nan
        st.metric("Écart de performance totale (Valority – Client)", f"{delta_perf:,.2f} pp".replace(",", " ").replace(".", ",") if delta_perf == delta_perf else "—")

    st.markdown("---")
    st.subheader("Évolution des portefeuilles (ligne du temps)")

    if not dfA_val.empty and not dfB_val.empty:
        df_comp = pd.DataFrame(
            {
                "Date": dfA_val.index,
                "Client": dfA_val["Valeur"].values,
                "Valority": dfB_val["Valeur"].values,
            }
        )
        df_comp = df_comp.melt(id_vars="Date", var_name="Portefeuille", value_name="Valeur")

        chart = (
            alt.Chart(df_comp)
            .mark_line()
            .encode(
                x=alt.X("Date:T", title="Date"),
                y=alt.Y("Valeur:Q", title="Valeur (€)"),
                color=alt.Color("Portefeuille:N", legend=alt.Legend(title="Portefeuille")),
                tooltip=[
                    alt.Tooltip("Date:T", title="Date"),
                    alt.Tooltip("Portefeuille:N", title="Portefeuille"),
                    alt.Tooltip("Valeur:Q", title="Valeur (€)", format=",.2f"),
                ],
            )
            .properties(height=350)
            .interactive()
        )
        st.altair_chart(chart, use_container_width=True)
    else:
        st.info("Pas de données de valorisation à afficher.")

    st.markdown("---")
    st.subheader("Positions (détails par ligne)")

    positions_table("Portefeuille Client", "A_lines")
    st.markdown("---")
    positions_table("Portefeuille Valority", "B_lines")

    st.markdown("---")

    def _render_portfolio_pie(port_key: str, title: str):
        df_positions = build_positions_dataframe(port_key)
        df_pie = _prepare_pie_df(df_positions, max_items=8, min_pct=0.03)
        if df_pie.empty:
            st.info(f"Aucune donnée disponible pour {title}.")
            return

        chart = (
            alt.Chart(df_pie)
            .mark_arc()
            .encode(
                theta=alt.Theta("Valeur actuelle €:Q"),
                color=alt.Color("Nom:N", legend=alt.Legend(title="Fonds")),
                tooltip=[
                    alt.Tooltip("Nom:N", title="Fonds"),
                    alt.Tooltip("Part %:Q", title="Part %", format=".2f"),
                    alt.Tooltip("Valeur actuelle €:Q", title="Valeur (€)", format=",.2f"),
                ],
            )
            .properties(height=280, title=title)
        )
        st.altair_chart(chart, use_container_width=True)

    st.subheader("Répartition par fonds")
    col_a, col_b = st.columns(2)
    with col_a:
        _render_portfolio_pie("A_lines", "Portefeuille Client")
    with col_b:
        _render_portfolio_pie("B_lines", "Portefeuille Valority")

    # APP – Composition
    st.subheader("Composition du portefeuille (Valority)")

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

    _render_valority_composition_section()

    with st.expander("Aide rapide"):
        st.markdown(
            """
- Dans chaque portefeuille, vous pouvez **soit** ajouter des *fonds recommandés* (onglet dédié),
  **soit** utiliser la *saisie libre* avec ISIN / code.
- Pour le **fonds en euros**, utilisez le symbole **EUROFUND** (taux paramétrable dans la barre de gauche).
- Les frais d'entrée s'appliquent à chaque investissement.
- Le **rendement total** est la performance globale depuis l'origine (valeur actuelle / net investi).
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

        st.markdown("---")

        # Portefeuille Valority
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



def render_mode_router():
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    mode = st.radio(
        "Choisissez votre mode",
        ["Créer le portefeuille parfait", "Comparateur de portefeuilles"],
        horizontal=True,
    )
    if mode == "Créer le portefeuille parfait":
        render_portfolio_builder()
    else:
        render_app(run_page_config=False)


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
