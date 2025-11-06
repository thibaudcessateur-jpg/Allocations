# =========================================
# 00) APP CONFIG
# =========================================
import os, re, math, json, requests, calendar
from datetime import datetime, date
from typing import Optional, List, Dict, Any, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

st.set_page_config(page_title="Comparateur de Portefeuilles — CGP", page_icon="🦉", layout="wide")
TODAY = pd.Timestamp.today().normalize()
TRADING_DAYS = 252


# =========================================
# 01) HELPERS GÉNÉRAUX
# =========================================
def Secret_Token(name: str) -> str:
    v = os.getenv(name) or str(st.secrets.get(name, "")).strip()  # type: ignore[attr-defined]
    if not v:
        raise RuntimeError(f"Secret manquant: {name}")
    return v

def to_eur(x: float) -> str:
    try:
        if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
            return ""
        s = f"{float(x):,.2f}".replace(",", "X").replace(".", ",").replace("X", " ")
        return f"{s} €"
    except Exception:
        return ""

def to_pct(x: float) -> str:
    try:
        if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
            return ""
        s = f"{float(x):+.2f}".replace(",", "X").replace(".", ",").replace("X", " ")
        return f"{s}%"
    except Exception:
        return ""


# =========================================
# 02) EODHD CLIENT (prix quotidiens / VL)
# =========================================
EODHD_BASE = "https://eodhd.com/api"

def eodhd_get(path: str, params: Optional[Dict[str, Any]] = None) -> Any:
    url = f"{EODHD_BASE.rstrip('/')}{path}"
    p = {"api_token": Secret_Token("EODHD_API_KEY"), "fmt": "json"}
    if params:
        p.update(params)
    r = requests.get(url, params=p, timeout=25)
    r.raise_for_status()
    js = r.json()
    if isinstance(js, str) and js.lower().startswith("error"):
        raise requests.HTTPError(js)
    return js

@st.cache_data(ttl=6*3600, show_spinner=False)
def eod_search(query: str) -> List[Dict[str, Any]]:
    try:
        js = eodhd_get(f"/search/{query.upper()}", params={})
        return js if isinstance(js, list) else []
    except Exception:
        return []

def _looks_like_isin(s: str) -> bool:
    return bool(re.fullmatch(r"[A-Z]{2}[A-Z0-9]{9}[0-9]", s.strip().upper()))

@st.cache_data(ttl=24*3600, show_spinner=False)
def resolve_symbol(q: str) -> Optional[str]:
    """
    Résout un ISIN / nom vers un symbole EODHD.
    - Si ISIN et Exchange EUFUND => retourne "ISIN.EUFUND" (fonds/OPCVM)
    - Sinon renvoie le 'Code' trouvé, ou tente des suffixes boursiers courants.
    """
    q = q.strip()
    if _looks_like_isin(q):
        res = eod_search(q)
        if res:
            for it in res:
                if str(it.get("Exchange", "")).upper() == "EUFUND":
                    return f"{q}.EUFUND"
            code = res[0].get("Code")
            if code:
                return str(code)
        for suf in [".EUFUND", ".PA", ".AS", ".MI", ".DE", ".LSE"]:
            try_sym = f"{q}{suf}"
            try:
                js = eodhd_get(f"/eod/{try_sym}", params={"period": "d"})
                if isinstance(js, list) and js:
                    return try_sym
            except Exception:
                continue
        return None
    else:
        res = eod_search(q)
        if res:
            code = res[0].get("Code")
            if code:
                if str(res[0].get("Exchange", "")).upper() == "EUFUND":
                    isin = res[0].get("ISIN")
                    if isinstance(isin, str) and _looks_like_isin(isin):
                        return f"{isin}.EUFUND"
                return str(code)
    return None

@st.cache_data(ttl=3*3600, show_spinner=False)
def eod_prices(symbol: str, from_dt: Optional[str] = None) -> pd.DataFrame:
    params = {"period": "d"}
    if from_dt:
        params["from"] = from_dt
    js = eodhd_get(f"/eod/{symbol}", params=params)
    df = pd.DataFrame(js)
    if df.empty:
        return pd.DataFrame()
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    return df[["close"]].rename(columns={"close": "Close"})

def get_close_on(df: pd.DataFrame, dt: pd.Timestamp) -> Optional[float]:
    if df.empty:
        return None
    if dt in df.index:
        v = df.loc[dt, "Close"]
        return float(v) if pd.notna(v) else None
    before = df.loc[df.index <= dt]
    if before.empty:
        return None
    v = before["Close"].iloc[-1]
    return float(v) if pd.notna(v) else None


# =========================================
# 03) FINANCE — XIRR (money-weighted)
# =========================================
def xnpv(rate: float, cash_flows: List[Tuple[pd.Timestamp, float]]) -> float:
    t0 = cash_flows[0][0]
    return sum(cf / ((1 + rate) ** ((t - t0).days / 365.2425)) for t, cf in cash_flows)

def xirr(cash_flows: List[Tuple[pd.Timestamp, float]], guess: float = 0.1) -> Optional[float]:
    if not cash_flows or len(cash_flows) < 2:
        return None
    low, high = -0.9999, 10.0
    for _ in range(100):
        mid = (low + high) / 2
        val = xnpv(mid, cash_flows)
        if abs(val) < 1e-6:
            return mid
        val_low = xnpv(low, cash_flows)
        if val_low * val < 0:
            high = mid
        else:
            low = mid
    return None


# =========================================
# 04) UI — COMPARATEUR A vs B (saisie)
# =========================================
st.title("🟣 Comparateur de Portefeuilles")
st.caption("Comparez A (client) vs B (vous). Dates et prix d’achat pris en compte.")

def parse_date(s: str) -> Optional[pd.Timestamp]:
    s = str(s).strip()
    if not s:
        return None
    try:
        return pd.Timestamp(s)
    except Exception:
        return None

def parse_float(s: Any) -> Optional[float]:
    if s is None or s == "":
        return None
    try:
        return float(str(s).replace(",", "."))
    except Exception:
        return None

defaultA = pd.DataFrame([
    {"Instrument (ISIN ou Nom)": "FR0011253624", "Quantité": 10, "Date d’achat (YYYY-MM-DD)": "2023-09-15", "Prix d’achat (optionnel)": ""},
    {"Instrument (ISIN ou Nom)": "IE00B4L5Y983", "Quantité": 5,  "Date d’achat (YYYY-MM-DD)": "2024-02-01", "Prix d’achat (optionnel)": ""},
])
defaultB = pd.DataFrame([
    {"Instrument (ISIN ou Nom)": "FR0010148981", "Quantité": 8, "Date d’achat (YYYY-MM-DD)": "2023-11-10", "Prix d’achat (optionnel)": ""},
    {"Instrument (ISIN ou Nom)": "IE00B6R52259", "Quantité": 4, "Date d’achat (YYYY-MM-DD)": "2024-03-20", "Prix d’achat (optionnel)": ""},
])

cA, cB = st.columns(2)
with cA:
    st.subheader("Portefeuille A (client)")
    dfA = st.data_editor(defaultA, num_rows="dynamic", use_container_width=True)
with cB:
    st.subheader("Portefeuille B (vous)")
    dfB = st.data_editor(defaultB, num_rows="dynamic", use_container_width=True)

run_ab = st.button("📊 Comparer A vs B", type="primary")


# =========================================
# 05) CALCUL — PAR PORTEFEUILLE
# =========================================
@st.cache_data(ttl=3600, show_spinner=False)
def load_price_series_cached(symbol: str, from_dt: str) -> pd.DataFrame:
    return eod_prices(symbol, from_dt=from_dt)

def compute_portfolio(df_input: pd.DataFrame, label: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    rows = []
    cash_flows: List[Tuple[pd.Timestamp, float]] = []
    min_buy_dt: Optional[pd.Timestamp] = None
    total_investi, total_valeur = 0.0, 0.0

    for _, row in df_input.iterrows():
        instr = str(row.get("Instrument (ISIN ou Nom)", "")).strip()
        qty = parse_float(row.get("Quantité", 0))
        d_buy = parse_date(row.get("Date d’achat (YYYY-MM-DD)", ""))
        px_buy_opt = parse_float(row.get("Prix d’achat (optionnel)", ""))

        if not instr or not qty or qty <= 0 or not d_buy:
            continue

        sym = resolve_symbol(instr)
        if not sym:
            rows.append({"Instrument": instr, "Symbole": "—", "Quantité": qty, "Date achat": d_buy.date(),
                         "Prix achat": px_buy_opt if px_buy_opt else "—", "Dernier cours": "ND",
                         "Investi €": np.nan, "Valeur actuelle €": np.nan, "P&L €": np.nan,
                         "Perf % depuis achat": np.nan})
            continue

        dfp = load_price_series_cached(sym, d_buy.strftime("%Y-%m-%d"))
        last_close = dfp["Close"].iloc[-1] if not dfp.empty else np.nan

        if px_buy_opt is None:
            px_buy = get_close_on(dfp, d_buy)
            if px_buy is None:
                after = dfp.loc[dfp.index >= d_buy]
                px_buy = float(after["Close"].iloc[0]) if not after.empty else np.nan
        else:
            px_buy = px_buy_opt

        investi = float(qty) * float(px_buy) if pd.notna(px_buy) else np.nan
        valeur = float(qty) * float(last_close) if pd.notna(last_close) else np.nan
        pl_eur = valeur - investi if (pd.notna(valeur) and pd.notna(investi)) else np.nan
        perf_pct = (valeur / investi - 1.0) * 100.0 if (pd.notna(valeur) and pd.notna(investi) and investi != 0) else np.nan

        rows.append({
            "Instrument": instr, "Symbole": sym, "Quantité": qty, "Date achat": d_buy.date(),
            "Prix achat": px_buy if pd.notna(px_buy) else "—", "Dernier cours": last_close if pd.notna(last_close) else "ND",
            "Investi €": investi, "Valeur actuelle €": valeur, "P&L €": pl_eur, "Perf % depuis achat": perf_pct
        })

        if pd.notna(investi):
            total_investi += investi
            cash_flows.append((d_buy, -investi))
            if (min_buy_dt is None) or (d_buy < min_buy_dt):
                min_buy_dt = d_buy
        if pd.notna(valeur):
            total_valeur += valeur

    if total_valeur > 0 and cash_flows:
        cash_flows.append((TODAY, total_valeur))

    df_lines = pd.DataFrame(rows)
    perf_simple = (total_valeur / total_investi - 1.0) if total_investi > 0 else np.nan
    irr = xirr(cash_flows) if cash_flows else None
    agg = {
        "label": label,
        "investi_total": total_investi,
        "valeur_totale": total_valeur,
        "pl_total": total_valeur - total_investi if total_investi > 0 else np.nan,
        "perf_simple_pct": perf_simple * 100.0 if pd.notna(perf_simple) else np.nan,
        "xirr_pct": irr * 100.0 if irr is not None else np.nan,
        "min_buy_dt": min_buy_dt.date() if min_buy_dt else None,
    }
    return df_lines, agg


# =========================================
# 06) RUN — A vs B
# =========================================
if run_ab:
    st.divider()
    st.subheader("Résultats A vs B")

    ca, cb = st.columns(2)
    with ca:
        st.markdown("### 📁 Portefeuille A (client)")
        dfA_lines, aggA = compute_portfolio(dfA, "Portefeuille A")
        st.dataframe(
            dfA_lines.style.format({
                "Prix achat": to_eur, "Dernier cours": to_eur,
                "Investi €": to_eur, "Valeur actuelle €": to_eur,
                "P&L €": to_eur, "Perf % depuis achat": "{:.2f}%"
            }, na_rep=""),
            use_container_width=True, hide_index=True
        )
        st.write(
            f"- **Investi** : {to_eur(aggA['investi_total'])}  "
            f"- **Valeur** : {to_eur(aggA['valeur_totale'])}  "
            f"- **P&L** : {to_eur(aggA['pl_total'])}  "
            f"- **Perf simple** : {to_pct(aggA['perf_simple_pct'])}  "
            f"- **XIRR** : {to_pct(aggA['xirr_pct'])}  "
            f"- **Depuis** : {aggA['min_buy_dt']}"
        )

    with cb:
        st.markdown("### 🟣 Portefeuille B (vous)")
        dfB_lines, aggB = compute_portfolio(dfB, "Portefeuille B")
        st.dataframe(
            dfB_lines.style.format({
                "Prix achat": to_eur, "Dernier cours": to_eur,
                "Investi €": to_eur, "Valeur actuelle €": to_eur,
                "P&L €": to_eur, "Perf % depuis achat": "{:.2f}%"
            }, na_rep=""),
            use_container_width=True, hide_index=True
        )
        st.write(
            f"- **Investi** : {to_eur(aggB['investi_total'])}  "
            f"- **Valeur** : {to_eur(aggB['valeur_totale'])}  "
            f"- **P&L** : {to_eur(aggB['pl_total'])}  "
            f"- **Perf simple** : {to_pct(aggB['perf_simple_pct'])}  "
            f"- **XIRR** : {to_pct(aggB['xirr_pct'])}  "
            f"- **Depuis** : {aggB['min_buy_dt']}"
        )

    st.markdown("### ⚖️ Comparatif synthétique")
    comp = pd.DataFrame([
        {"Portefeuille": "A (client)", "Investi €": aggA["investi_total"], "Valeur €": aggA["valeur_totale"],
         "P&L €": aggA["pl_total"], "Perf simple %": aggA["perf_simple_pct"], "XIRR %": aggA["xirr_pct"]},
        {"Portefeuille": "B (vous)",   "Investi €": aggB["investi_total"], "Valeur €": aggB["valeur_totale"],
         "P&L €": aggB["pl_total"], "Perf simple %": aggB["perf_simple_pct"], "XIRR %": aggB["xirr_pct"]},
    ])
    st.dataframe(
        comp.style.format({
            "Investi €": to_eur, "Valeur €": to_eur, "P&L €": to_eur,
            "Perf simple %": "{:.2f}%", "XIRR %": "{:.2f}%"
        }, na_rep=""),
        use_container_width=True, hide_index=True
    )

    fig = px.bar(comp, x="Portefeuille", y="Valeur €", text="Valeur €", title="Valeur actuelle par portefeuille")
    st.plotly_chart(fig, use_container_width=True)


# =========================================
# 07) PORTFEUILLE DE RÉFÉRENCE (Generali)
# =========================================
st.divider()
st.header("🟣 Portefeuille de référence — Generali (Core + Défensifs)")

UNIVERSE_GENERALI = [
    {"name": "R-co Valor C EUR", "isin": "FR0011253624", "type": "Actions Monde"},
    {"name": "Vivalor International", "isin": "FR0014001LS1", "type": "Actions Monde"},
    {"name": "CARMIGNAC Investissement A EUR", "isin": "FR0010148981", "type": "Actions Monde"},
    {"name": "FIDELITY Funds - World Fund", "isin": "LU0069449576", "type": "Actions Monde"},
    {"name": "CLARTAN Valeurs C", "isin": "LU1100076550", "type": "Actions Europe"},
    {"name": "CARMIGNAC Patrimoine", "isin": "FR0010135103", "type": "Diversifié patrimonial"},
    {"name": "SYCOYIELD 2030 RC", "isin": "FR001400MCQ6", "type": "Obligataire échéance"},
    {"name": "R-Co Target 2029 HY", "isin": "FR001400AWH8", "type": "Obligataire haut rendement"},  # confirme ISIN
    {"name": "Fonds en euros Generali", "isin": None, "type": "Fonds Euro"},
]
DF_UNI = pd.DataFrame(UNIVERSE_GENERALI)

_w_init = pd.DataFrame(
    [{"Fonds": r["name"], "ISIN": r["isin"], "Type": r["type"], "Poids %": 0.0} for r in UNIVERSE_GENERALI]
)
st.markdown("Saisis les **poids (%)** (0% possible) — la somme sera **normalisée**.")
df_weights = st.data_editor(_w_init, use_container_width=True, num_rows="fixed")
df_weights["Poids %"] = pd.to_numeric(df_weights["Poids %"], errors="coerce").fillna(0.0)

c1, c2, c3 = st.columns(3)
with c1:
    ref_initial_date = st.date_input("Date d’investissement initial", value=date(2024, 1, 2))
with c2:
    ref_initial_amount = st.number_input("Montant initial (€)", min_value=0.0, value=10000.0, step=500.0, format="%.2f")
with c3:
    euro_rate = st.number_input("Rendement annuel Fonds € (%)", min_value=0.0, value=2.00, step=0.10, format="%.2f")

st.markdown("### Versements Libres Programmés (VLP) — *optionnels*")
d1, d2, d3, d4 = st.columns(4)
with d1:
    vlp_enable = st.checkbox("Activer VLP mensuels", value=False)
with d2:
    vlp_amount = st.number_input("Montant VLP mensuel (€)", min_value=0.0, value=300.0, step=50.0, format="%.2f", disabled=not vlp_enable)
with d3:
    vlp_start = st.date_input("Début VLP", value=ref_initial_date, disabled=not vlp_enable)
with d4:
    vlp_end = st.date_input("Fin VLP", value=date.today(), disabled=not vlp_enable)

run_ref = st.button("🚀 Calculer la performance du portefeuille de référence", type="primary")

def month_end(dt: date) -> date:
    last_day = calendar.monthrange(dt.year, dt.month)[1]
    return date(dt.year, dt.month, last_day)

def monthly_schedule(start: date, end: date) -> list[date]:
    if end < start:
        return []
    # Avance mois par mois (sécurisé jour<=28)
    d = date(start.year, start.month, min(start.day, 28))
    out = []
    while d <= end:
        out.append(month_end(d))
        y = d.year + (d.month // 12)
        m = 1 if d.month == 12 else d.month + 1
        d = date(y, m, min(d.day, 28))
    return out

def to_ts(d: date) -> pd.Timestamp:
    return pd.Timestamp(d)

@st.cache_data(ttl=6*3600, show_spinner=False)
def euro_fund_series(start_dt: date, end_dt: date, annual_rate: float) -> pd.Series:
    idx = pd.bdate_range(start=to_ts(start_dt), end=to_ts(end_dt), freq="B")
    if len(idx) == 0:
        return pd.Series(dtype=float)
    daily_rate = (1 + annual_rate/100.0) ** (1/252) - 1
    vals = (1 + daily_rate) ** np.arange(len(idx))
    s = pd.Series(vals, index=idx, name="Fonds € (synthetic)")
    return s * 100.0

@st.cache_data(ttl=6*3600, show_spinner=False)
def load_price_series_generic(isin: str | None, name: str, from_dt: date) -> pd.Series:
    if isin:
        sym = resolve_symbol(isin)
    else:
        sym = None
    if sym:
        dfp = eod_prices(sym, from_dt.strftime("%Y-%m-%d"))
        if not dfp.empty:
            s = dfp["Close"].copy()
            s.name = name
            return s
    if isin is None and "Fonds en euros" in name:
        return euro_fund_series(from_dt, date.today(), euro_rate).rename(name)
    return pd.Series(dtype=float)

def simulate_reference_portfolio(dfw: pd.DataFrame,
                                 initial_dt: date,
                                 initial_amt: float,
                                 vlp_on: bool,
                                 vlp_amt: float,
                                 vlp_dt_start: date,
                                 vlp_dt_end: date) -> dict:
    dfw = dfw.copy()
    dfw["Poids %"] = pd.to_numeric(dfw["Poids %"], errors="coerce").fillna(0.0)
    total = dfw["Poids %"].sum()
    if total <= 0:
        st.error("Tous les poids sont à 0%. Mets au moins un poids > 0.")
        return {}
    dfw["poids_n"] = dfw["Poids %"] / total

    series_map: dict[str, pd.Series] = {}
    not_found = []
    for _, r in dfw.iterrows():
        if r["poids_n"] <= 0:
            continue
        s = load_price_series_generic(r["ISIN"], r["Fonds"], initial_dt)
        if s.empty:
            not_found.append(r["Fonds"])
        else:
            series_map[r["Fonds"]] = s

    if not_found:
        st.warning("Données manquantes pour: " + ", ".join(not_found))
    if not series_map:
        st.error("Aucune VL disponible pour les fonds sélectionnés.")
        return {}

    prices = pd.concat(series_map, axis=1).sort_index().ffill()
    prices = prices.loc[prices.index >= to_ts(initial_dt)]
    if prices.empty:
        st.error("Pas de données de VL après la date d’investissement.")
        return {}

    cashflows: list[tuple[pd.Timestamp, float]] = []
    contribs: list[tuple[pd.Timestamp, float]] = []
    cashflows.append((to_ts(initial_dt), -float(initial_amt)))
    contribs.append((to_ts(initial_dt), float(initial_amt)))

    if vlp_on and vlp_amt > 0:
        for d in monthly_schedule(vlp_dt_start, vlp_dt_end):
            ts = to_ts(d)
            if ts < prices.index.min():
                continue
            cashflows.append((ts, -float(vlp_amt)))
            contribs.append((ts, float(vlp_amt)))

    parts = {f: 0.0 for f in prices.columns}

    def px_on(fund: str, dts: pd.Timestamp) -> float | None:
        s = prices[fund]
        if dts in s.index:
            v = s.loc[dts]
            return float(v) if pd.notna(v) else None
        bef = s.loc[s.index <= dts]
        if not bef.empty:
            return float(bef.iloc[-1])
        aft = s.loc[s.index >= dts]
        if not aft.empty:
            return float(aft.iloc[0])
        return None

    for ts, amt in contribs:
        for _, r in dfw.iterrows():
            if r["poids_n"] <= 0:
                continue
            fund = r["Fonds"]
            if fund not in prices.columns:
                continue
            px = px_on(fund, ts)
            if (px is None) or (px <= 0):
                continue
            parts[fund] += (amt * r["poids_n"]) / px

    vals = []
    for dts in prices.index:
        val = 0.0
        for f in prices.columns:
            px = prices.loc[dts, f]
            if pd.isna(px):
                continue
            val += parts.get(f, 0.0) * float(px)
        vals.append((dts, val))
    df_val = pd.DataFrame(vals, columns=["Date", "Valeur"]).set_index("Date")

    total_investi = -sum(cf for _, cf in cashflows if cf < 0)
    final_val = float(df_val["Valeur"].iloc[-1])
    irr = xirr(cashflows + [(df_val.index[-1], final_val)])

    out = {
        "prices": prices,
        "values": df_val,
        "investi": total_investi,
        "final": final_val,
        "pl": final_val - total_investi,
        "perf_pct": (final_val / total_investi - 1.0) * 100.0 if total_investi > 0 else np.nan,
        "xirr_pct": (irr * 100.0) if irr is not None else np.nan,
        "parts": parts,
    }
    return out

if run_ref:
    res = simulate_reference_portfolio(
        df_weights, ref_initial_date, ref_initial_amount,
        vlp_enable, vlp_amount, vlp_start, vlp_end
    )
    if res:
        st.subheader("📈 Valeur du portefeuille (VL agrégées)")
        fig = px.line(res["values"].reset_index(), x="Date", y="Valeur", title="Portefeuille de référence (valeur)")
        st.plotly_chart(fig, use_container_width=True)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Investi total", to_eur(res["investi"]))
        c2.metric("Valeur actuelle", to_eur(res["final"]))
        c3.metric("P&L", to_eur(res["pl"]))
        c4.metric("Perf depuis origine", f"{res['perf_pct']:.2f}%")
        st.markdown(f"**XIRR (annualisé)** : {'' if pd.isna(res['xirr_pct']) else f'{res['xirr_pct']:.2f}%'}")

        last_date = res["values"].index[-1]
        last_px = res["prices"].loc[last_date]
        tbl = []
        for f in res["prices"].columns:
            qty = res["parts"].get(f, 0.0)
            px = float(last_px.get(f, np.nan))
            val = qty * px if pd.notna(px) else np.nan
            tbl.append({"Fonds": f, "Parts détenues": qty, "Dernier prix": px, "Valeur €": val})
        st.subheader("📄 Détail positions (estimées)")
        st.dataframe(
            pd.DataFrame(tbl).style.format({"Parts détenues": "{:.4f}", "Dernier prix": to_eur, "Valeur €": to_eur}, na_rep=""),
            use_container_width=True, hide_index=True
        )

        base100 = res["values"] / res["values"].iloc[0] * 100.0
        st.subheader("📊 Performance base 100")
        st.line_chart(base100.rename(columns={"Valeur": "Référence (base 100)"}))
else:
    st.info("Remplis les poids, la date (et VLP si besoin), puis clique **Calculer**.")

# =========================================
# 08) PORTEFEUILLES PAR POIDS — A (client) vs B (toi)
#     - même logique que le portefeuille de référence
#     - poids (0% possible), date, montant initial, VLP
# =========================================

st.divider()
st.header("⚖️ Comparaison par composition (poids) — Portefeuille A vs Portefeuille B")

# --- Table poids A & B (pré-remplies avec l'univers Generali)
_w_template = pd.DataFrame(
    [{"Fonds": r["name"], "ISIN": r["isin"], "Type": r["type"], "Poids %": 0.0} for r in UNIVERSE_GENERALI]
)

cA, cB = st.columns(2)
with cA:
    st.subheader("Portefeuille A (client) — Composition")
    df_weights_A = st.data_editor(_w_template.copy(), use_container_width=True, num_rows="fixed", key="weights_A")
with cB:
    st.subheader("Portefeuille B (toi) — Composition")
    df_weights_B = st.data_editor(_w_template.copy(), use_container_width=True, num_rows="fixed", key="weights_B")

# --- Paramètres d’investissement A & B
colA1, colA2, colA3 = st.columns(3)
with colA1:
    A_initial_date = st.date_input("A — Date d’investissement initial", value=date(2024, 1, 2), key="A_dt")
with colA2:
    A_initial_amount = st.number_input("A — Montant initial (€)", min_value=0.0, value=10000.0, step=500.0, format="%.2f", key="A_amt")
with colA3:
    A_euro_rate = st.number_input("A — Rendement annuel Fonds € (%)", min_value=0.0, value=euro_rate, step=0.10, format="%.2f", key="A_euro")

colA4, colA5, colA6, colA7 = st.columns(4)
with colA4:
    A_vlp_enable = st.checkbox("A — Activer VLP mensuels", value=False, key="A_vlp_on")
with colA5:
    A_vlp_amount = st.number_input("A — Montant VLP mensuel (€)", min_value=0.0, value=300.0, step=50.0, format="%.2f", disabled=not A_vlp_enable, key="A_vlp_amt")
with colA6:
    A_vlp_start = st.date_input("A — Début VLP", value=A_initial_date, disabled=not A_vlp_enable, key="A_vlp_start")
with colA7:
    A_vlp_end = st.date_input("A — Fin VLP", value=date.today(), disabled=not A_vlp_enable, key="A_vlp_end")

colB1, colB2, colB3 = st.columns(3)
with colB1:
    B_initial_date = st.date_input("B — Date d’investissement initial", value=date(2024, 1, 2), key="B_dt")
with colB2:
    B_initial_amount = st.number_input("B — Montant initial (€)", min_value=0.0, value=10000.0, step=500.0, format="%.2f", key="B_amt")
with colB3:
    B_euro_rate = st.number_input("B — Rendement annuel Fonds € (%)", min_value=0.0, value=euro_rate, step=0.10, format="%.2f", key="B_euro")

colB4, colB5, colB6, colB7 = st.columns(4)
with colB4:
    B_vlp_enable = st.checkbox("B — Activer VLP mensuels", value=False, key="B_vlp_on")
with colB5:
    B_vlp_amount = st.number_input("B — Montant VLP mensuel (€)", min_value=0.0, value=300.0, step=50.0, format="%.2f", disabled=not B_vlp_enable, key="B_vlp_amt")
with colB6:
    B_vlp_start = st.date_input("B — Début VLP", value=B_initial_date, disabled=not B_vlp_enable, key="B_vlp_start")
with colB7:
    B_vlp_end = st.date_input("B — Fin VLP", value=date.today(), disabled=not B_vlp_enable, key="B_vlp_end")

# ⚠️ Important : si le "Fonds en euros" est présent, on réutilise la série synthétique avec le taux saisi
# on met à jour euro_rate dynamique avant calcul
def _set_euro_rate(rate: float):
    # simple setter local: la fonction euro_fund_series lit euro_rate via la closure de load_price_series_generic (si tu préfères, passe rate en param)
    # Ici on écrase la variable globale 'euro_rate' si elle existe
    try:
        globals()["euro_rate"] = rate
    except Exception:
        pass

run_comp = st.button("🚀 Calculer A (client) vs B (toi) — par composition", type="primary")

if run_comp:
    # normalise les poids et calcule chaque portefeuille via simulate_reference_portfolio
    # (on réutilise la logique de la section Référence)
    # A
    _set_euro_rate(A_euro_rate)
    resA = simulate_reference_portfolio(
        df_weights_A, A_initial_date, A_initial_amount,
        A_vlp_enable, A_vlp_amount, A_vlp_start, A_vlp_end
    )
    # B
    _set_euro_rate(B_euro_rate)
    resB = simulate_reference_portfolio(
        df_weights_B, B_initial_date, B_initial_amount,
        B_vlp_enable, B_vlp_amount, B_vlp_start, B_vlp_end
    )

    if resA and resB:
        st.subheader("📈 Valeur des portefeuilles (VL agrégées)")
        left, right = st.columns(2)
        with left:
            st.markdown("**Portefeuille A (client)**")
            figA = px.line(resA["values"].reset_index(), x="Date", y="Valeur", title="A — Valeur du portefeuille")
            st.plotly_chart(figA, use_container_width=True)
        with right:
            st.markdown("**Portefeuille B (toi)**")
            figB = px.line(resB["values"].reset_index(), x="Date", y="Valeur", title="B — Valeur du portefeuille")
            st.plotly_chart(figB, use_container_width=True)

        # Métriques côte-à-côte
        st.subheader("📊 Synthèse")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("### A (client)")
            ca1, ca2, ca3, ca4 = st.columns(4)
            ca1.metric("Investi", to_eur(resA["investi"]))
            ca2.metric("Valeur", to_eur(resA["final"]))
            ca3.metric("P&L", to_eur(resA["pl"]))
            ca4.metric("Perf", f"{resA['perf_pct']:.2f}%")
            st.markdown(f"**XIRR (annualisé)** : {'' if pd.isna(resA['xirr_pct']) else f'{resA['xirr_pct']:.2f}%'}")

        with c2:
            st.markdown("### B (toi)")
            cb1, cb2, cb3, cb4 = st.columns(4)
            cb1.metric("Investi", to_eur(resB["investi"]))
            cb2.metric("Valeur", to_eur(resB["final"]))
            cb3.metric("P&L", to_eur(resB["pl"]))
            cb4.metric("Perf", f"{resB['perf_pct']:.2f}%")
            st.markdown(f"**XIRR (annualisé)** : {'' if pd.isna(resB['xirr_pct']) else f'{resB['xirr_pct']:.2f}%'}")

        # Base 100 comparée
        st.subheader("📎 Comparaison directe — Base 100")
        base = pd.concat(
            [
                resA["values"].rename(columns={"Valeur": "A (client)"}),
                resB["values"].rename(columns={"Valeur": "B (toi)"}),
            ],
            axis=1
        ).dropna(how="any")
        if not base.empty:
            base100 = base / base.iloc[0] * 100.0
            fig_comp = px.line(base100.reset_index(), x="Date", y=base100.columns, title="A vs B — Base 100")
            st.plotly_chart(fig_comp, use_container_width=True)
        else:
            st.info("Périodes non chevauchantes : impossible de tracer la base 100 commune. Essaie de rapprocher les dates d’investissement.")

        # Détail positions au dernier point (optionnel)
        with st.expander("📄 Détail des positions (dernier jour)"):
            # A
            lastA = resA["values"].index[-1]
            last_px_A = resA["prices"].loc[lastA]
            tblA = []
            for f in resA["prices"].columns:
                qty = resA["parts"].get(f, 0.0)
                px = float(last_px_A.get(f, np.nan))
                val = qty * px if pd.notna(px) else np.nan
                tblA.append({"Portefeuille": "A", "Fonds": f, "Parts détenues": qty, "Dernier prix": px, "Valeur €": val})
            # B
            lastB = resB["values"].index[-1]
            last_px_B = resB["prices"].loc[lastB]
            tblB = []
            for f in resB["prices"].columns:
                qty = resB["parts"].get(f, 0.0)
                px = float(last_px_B.get(f, np.nan))
                val = qty * px if pd.notna(px) else np.nan
                tblB.append({"Portefeuille": "B", "Fonds": f, "Parts détenues": qty, "Dernier prix": px, "Valeur €": val})

            df_pos = pd.DataFrame(tblA + tblB)
            st.dataframe(
                df_pos.style.format({"Parts détenues": "{:.4f}", "Dernier prix": to_eur, "Valeur €": to_eur}, na_rep=""),
                use_container_width=True, hide_index=True
            )
    else:
        st.warning("Renseigne des poids > 0 et vérifie que des VL sont disponibles sur la période.")
