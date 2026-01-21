from __future__ import annotations

import json
from datetime import date
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import streamlit as st
import altair as alt
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfgen import canvas

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
# Calcul par ligne (avec frais)
# ------------------------------------------------------------
def compute_line_metrics(
    line: Dict[str, Any], fee_pct: float, euro_rate: float
) -> Tuple[float, float, float]:
    amt_brut = float(line.get("amount_gross", 0.0))
    net_amt = amt_brut * (1.0 - fee_pct / 100.0)
    buy_ts = pd.Timestamp(line.get("buy_date"))
    px_manual = line.get("buy_px", None)

    dfp, _, _ = get_price_series(line.get("isin") or line.get("name"), buy_ts, euro_rate)
    if str(line.get("sym_used", "")).upper() == "EUROFUND" or str(line.get("isin", "")).upper() == "EUROFUND":
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
