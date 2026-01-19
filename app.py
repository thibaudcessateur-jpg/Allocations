from __future__ import annotations

import base64
import importlib
import importlib.util
import json
from io import BytesIO
from datetime import date
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
import streamlit as st
import altair as alt

PLT_AVAILABLE = importlib.util.find_spec("matplotlib.pyplot") is not None
plt = importlib.import_module("matplotlib.pyplot") if PLT_AVAILABLE else None

WEASYPRINT_AVAILABLE = importlib.util.find_spec("weasyprint") is not None
HTML = importlib.import_module("weasyprint").HTML if WEASYPRINT_AVAILABLE else None

# ------------------------------------------------------------
# Constantes & univers de fonds recommandés
# ------------------------------------------------------------
TODAY = pd.Timestamp.today().normalize()
APP_TITLE = "Comparateur de portefeuilles"

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

# Frais de gestion du contrat
MGMT_FEE_EURO_PCT = 0.9
MGMT_FEE_UC_PCT = 1.2

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


def contract_mgmt_fee_pct(isin_or_name: str) -> float:
    code = str(isin_or_name or "").strip().upper()
    if code == "EUROFUND":
        return MGMT_FEE_EURO_PCT
    return MGMT_FEE_UC_PCT


def apply_management_fee(df: pd.DataFrame, fee_pct: float) -> pd.DataFrame:
    if df.empty or fee_pct <= 0:
        return df
    df_adj = df.copy()
    days = (df_adj.index - df_adj.index[0]).days.astype(float)
    factor = np.power(1.0 - fee_pct / 100.0, days / 365.0)
    df_adj["Close"] = df_adj["Close"].astype(float) * factor
    return df_adj


def _fig_to_base64_png(fig: "plt.Figure") -> str:
    if not PLT_AVAILABLE or plt is None:
        return ""
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=150)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _portfolio_pie_chart_b64(df_lines: pd.DataFrame, title: str) -> str:
    if not PLT_AVAILABLE or plt is None:
        return ""
    if df_lines is None or df_lines.empty:
        return ""
    if "Nom" not in df_lines.columns or "Valeur actuelle €" not in df_lines.columns:
        return ""
    values = df_lines[["Nom", "Valeur actuelle €"]].copy()
    values = values[values["Valeur actuelle €"] > 0]
    if values.empty:
        return ""
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.pie(
        values["Valeur actuelle €"],
        labels=values["Nom"],
        autopct="%1.1f%%",
        startangle=90,
    )
    ax.set_title(title)
    ax.axis("equal")
    return _fig_to_base64_png(fig)


def _portfolio_value_chart_b64(df_client: pd.DataFrame, df_valority: pd.DataFrame) -> str:
    if not PLT_AVAILABLE or plt is None:
        return ""
    if (df_client is None or df_client.empty) and (df_valority is None or df_valority.empty):
        return ""
    fig, ax = plt.subplots(figsize=(7, 3))
    if df_client is not None and not df_client.empty:
        ax.plot(df_client.index, df_client["Valeur"], label="Client")
    if df_valority is not None and not df_valority.empty:
        ax.plot(df_valority.index, df_valority["Valeur"], label="Valority")
    ax.set_title("Évolution de la valeur des portefeuilles")
    ax.set_xlabel("Date")
    ax.set_ylabel("Valeur (€)")
    ax.legend()
    fig.autofmt_xdate()
    return _fig_to_base64_png(fig)


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

        return df, "EUROFUND", "{}"

    # ✅ Instruments EODHD — recherche candidates puis EOD daily
    cands = _symbol_candidates(val)
    debug["cands"] = cands

    for sym in cands:
        df = eodhd_prices_daily(sym)
        if not df.empty:
            if start is not None:
                df = df.loc[df.index >= start]
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

        df_raw, sym, _ = get_price_series(ln.get("isin") or ln.get("name"), None, euro_rate)
        df = apply_management_fee(df_raw, contract_mgmt_fee_pct(sym or ln.get("isin") or ln.get("name")))
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
        mgmt_fee = contract_mgmt_fee_pct(sym or isin_or_name)
        df = apply_management_fee(df_full, mgmt_fee)

        if d_buy in df.index:
            px_buy = float(df_full.loc[d_buy, "Close"])
            eff_dt = d_buy
        else:
            after = df_full.loc[df_full.index >= d_buy]
            if after.empty:
                px_buy = float(df_full.iloc[-1]["Close"])
                eff_dt = df_full.index[-1]
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

        dfl_raw, sym, _ = get_price_series(ln.get("isin") or ln.get("name"), None, euro_rate)
        dfl = apply_management_fee(dfl_raw, contract_mgmt_fee_pct(sym or ln.get("isin") or ln.get("name")))
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
        dfl_raw, sym, _ = get_price_series(ln.get("isin") or ln.get("name"), buy_ts, euro_rate)
        dfl = apply_management_fee(dfl_raw, contract_mgmt_fee_pct(sym or ln.get("isin") or ln.get("name")))

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

        df_raw, sym, _ = get_price_series(ln.get("isin") or ln.get("name"), None, euro_rate)
        df = apply_management_fee(df_raw, contract_mgmt_fee_pct(sym or ln.get("isin") or ln.get("name")))
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

# ------------------------------------------------------------
# Layout principal
# ------------------------------------------------------------
st.set_page_config(page_title=APP_TITLE, layout="wide")
st.title(APP_TITLE)
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

    st.caption("Le taux est appliqué annuellement sur la part investie en fonds euros (EUROFUND).")

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

single_target_A = id(st.session_state["A_lines"][0]) if st.session_state["A_lines"] else None
single_target_B = id(st.session_state["B_lines"][0]) if st.session_state["B_lines"] else None

alloc_mode_code = st.session_state.get("ALLOC_MODE", "equal")

custom_month_weights_A: Optional[Dict[int, float]] = None
custom_oneoff_weights_A: Optional[Dict[int, float]] = None
custom_month_weights_B: Optional[Dict[int, float]] = None
custom_oneoff_weights_B: Optional[Dict[int, float]] = None

if alloc_mode_code == "custom":
    cmA = st.session_state.get("CUSTOM_M_A", {}) or {}
    coA = st.session_state.get("CUSTOM_O_A", {}) or {}
    tot_mA = sum(v for v in cmA.values() if v > 0)
    tot_oA = sum(v for v in coA.values() if v > 0)
    if tot_mA > 0:
        custom_month_weights_A = {k: v / tot_mA for k, v in cmA.items() if v > 0}
    if tot_oA > 0:
        custom_oneoff_weights_A = {k: v / tot_oA for k, v in coA.items() if v > 0}

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

if mode in ("compare", "client"):
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

if mode in ("compare", "valority"):
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

mode = st.session_state.get("MODE_ANALYSE", "compare")

# Déterminer le start_plot uniquement sur les portefeuilles affichés
full_dates: List[pd.Timestamp] = []
if mode in ("compare", "client") and isinstance(fullA, pd.Timestamp):
    full_dates.append(fullA)
if mode in ("compare", "valority") and isinstance(fullB, pd.Timestamp):
    full_dates.append(fullB)

start_plot = max(full_dates) if full_dates else TODAY

idx = pd.bdate_range(start=start_plot, end=TODAY, freq="B")
chart_df = pd.DataFrame(index=idx)

if mode in ("compare", "client") and not dfA.empty:
    chart_df["Client"] = dfA.reindex(idx)["Valeur"].ffill()

if mode in ("compare", "valority") and not dfB.empty:
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

perf_tot_client = (valA / netA - 1.0) * 100.0 if netA > 0 else None
perf_tot_valority = (valB / netB - 1.0) * 100.0 if netB > 0 else None

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
    synthA = report.get("client_summary", {})
    synthB = report.get("valority_summary", {})
    comp = report.get("comparison", {})

    dfA_lines = report.get("df_client_lines")
    dfB_lines = report.get("df_valority_lines")
    dfA_val = report.get("dfA_val")
    dfB_val = report.get("dfB_val")
    chart_value_b64 = report.get("chart_value_b64", "")
    pie_client_b64 = report.get("pie_client_b64", "")
    pie_valority_b64 = report.get("pie_valority_b64", "")

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
img {{
  max-width: 100%;
  height: auto;
}}
</style>
</head>
<body>

<h1>Rapport de portefeuille</h1>
<p class="small">Date de génération : {as_of}</p>

<h2>1. Synthèse chiffrée</h2>

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

<h2>2. Composition du portefeuille</h2>

<h3>Client</h3>
{"<img src='data:image/png;base64," + pie_client_b64 + "' alt='Composition Client' />" if pie_client_b64 else "<p class='small'>Composition indisponible.</p>"}

<h3>Valority</h3>
{"<img src='data:image/png;base64," + pie_valority_b64 + "' alt='Composition Valority' />" if pie_valority_b64 else "<p class='small'>Composition indisponible.</p>"}

<h2>3. Détail des lignes</h2>

<h3>Portefeuille Client</h3>
{html_client_lines}

<h3>Portefeuille Valority</h3>
{html_valority_lines}

<h2>4. Historique de la valeur des portefeuilles</h2>

{"<img src='data:image/png;base64," + chart_value_b64 + "' alt='Évolution de la valeur' />" if chart_value_b64 else "<p class='small'>Graphique indisponible.</p>"}

<h3>Client – Valeur du portefeuille par date</h3>
{html_A_val}

<h3>Valority – Valeur du portefeuille par date</h3>
{html_B_val}

<p class="small">
Ce document est fourni à titre informatif uniquement et ne constitue pas un conseil en investissement
personnalisé.
</p>

</body>
</html>
"""
    return html


st.markdown("---")
st.subheader("📄 Rapport client")

df_client_lines = portfolio_summary_dataframe("A_lines")
df_valority_lines = portfolio_summary_dataframe("B_lines")

dfA_val = dfA.reset_index().rename(columns={"index": "Date"}) if not dfA.empty else None
dfB_val = dfB.reset_index().rename(columns={"index": "Date"}) if not dfB.empty else None

chart_value_b64 = _portfolio_value_chart_b64(dfA, dfB)
pie_client_b64 = _portfolio_pie_chart_b64(df_client_lines, "Composition — Client")
pie_valority_b64 = _portfolio_pie_chart_b64(df_valority_lines, "Composition — Valority")

report_data = {
    "as_of": fmt_date(TODAY),
    "client_summary": {
        "val": valA,
        "net": netA,
        "brut": brutA,
        "perf_tot_pct": perf_tot_client or 0.0,
        "irr_pct": xirrA or 0.0,
    },
    "valority_summary": {
        "val": valB,
        "net": netB,
        "brut": brutB,
        "perf_tot_pct": perf_tot_valority or 0.0,
        "irr_pct": xirrB or 0.0,
    },
    "comparison": {
        "delta_val": (valB - valA) if (valA is not None and valB is not None) else 0.0,
        "delta_perf_pct": (
            (perf_tot_valority - perf_tot_client)
            if (perf_tot_client is not None and perf_tot_valority is not None)
            else 0.0
        ),
    },
    "df_client_lines": df_client_lines,
    "df_valority_lines": df_valority_lines,
    "dfA_val": dfA_val,
    "dfB_val": dfB_val,
    "chart_value_b64": chart_value_b64,
    "pie_client_b64": pie_client_b64,
    "pie_valority_b64": pie_valority_b64,
}

html_report = build_html_report(report_data)
st.download_button(
    "📄 Télécharger le rapport complet (HTML)",
    data=html_report.encode("utf-8"),
    file_name="rapport_portefeuille_valority.html",
    mime="text/html",
)

if HTML is None:
    st.warning(
        "La génération PDF est indisponible dans cet environnement. "
        "Installez WeasyPrint pour activer l'export PDF.",
    )
else:
    try:
        pdf_report = HTML(string=html_report).write_pdf()
        st.download_button(
            "📕 Télécharger le rapport complet (PDF)",
            data=pdf_report,
            file_name="rapport_portefeuille_valority.pdf",
            mime="application/pdf",
        )
    except Exception:
        st.warning(
            "La génération PDF est indisponible dans cet environnement. "
            "Vérifiez les dépendances système de WeasyPrint.",
        )

# ------------------------------------------------------------
# Bloc final : Comparaison OU "Frais & valeur créée"
# ------------------------------------------------------------
mode = st.session_state.get("MODE_ANALYSE", "compare")

def _years_between(d0: pd.Timestamp, d1: pd.Timestamp) -> float:
    return max(0.0, (d1 - d0).days / 365.25)

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
positions_table("Portefeuille 1 — Client", "A_lines")
positions_table("Portefeuille 2 — Valority", "B_lines")

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
            
