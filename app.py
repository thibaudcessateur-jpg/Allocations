# =========================================
# app.py — Comparateur de portefeuilles (propre, sans tableau Excel)
# =========================================
import os, re, math, requests, calendar
from datetime import date
from typing import Optional, List, Dict, Any, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

st.set_page_config(page_title="Comparateur Portefeuilles CGP", page_icon="🦉", layout="wide")
TODAY = pd.Timestamp.today().normalize()
TRADING_DAYS = 252


# =========================================
# 1) SECRETS / FORMATS
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
# 2) EODHD CLIENT (recherche & prix)
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
    Résout un ISIN / nom vers symbole EODHD.
    - Si ISIN et Exchange EUFUND => 'ISIN.EUFUND'
    - Sinon renvoie 'Code' trouvé ou tente suffixes.
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
# 3) FINANCE — XIRR (money-weighted)
# =========================================
def xnpv(rate: float, cash_flows: List[Tuple[pd.Timestamp, float]]) -> float:
    t0 = cash_flows[0][0]
    return sum(cf / ((1 + rate) ** ((t - t0).days / 365.2425)) for t, cf in cash_flows)

def xirr(cash_flows: List[Tuple[pd.Timestamp, float]]) -> Optional[float]:
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
# 4) UNIVERS VALORITY / GENERALI — Core + Défensifs
# =========================================
UNIVERSE_GENERALI = [
    {"name": "R-co Valor C EUR", "isin": "FR0011253624", "type": "Actions Monde"},
    {"name": "Vivalor International", "isin": "FR0014001LS1", "type": "Actions Monde"},
    {"name": "CARMIGNAC Investissement A EUR", "isin": "FR0010148981", "type": "Actions Monde"},
    {"name": "FIDELITY Funds - World Fund", "isin": "LU0069449576", "type": "Actions Monde"},
    {"name": "CLARTAN Valeurs C", "isin": "LU1100076550", "type": "Actions Europe"},
    {"name": "CARMIGNAC Patrimoine", "isin": "FR0010135103", "type": "Diversifié patrimonial"},
    {"name": "SYCOYIELD 2030 RC", "isin": "FR001400MCQ6", "type": "Obligataire échéance"},
    {"name": "R-Co Target 2029 HY", "isin": "FR001400AWH8", "type": "Obligataire haut rendement"},  # vérifie l'ISIN
]
DF_UNI = pd.DataFrame(UNIVERSE_GENERALI)
UNI_OPTIONS = ["— Saisie libre —"] + [f"{r['name']} — {r['isin']}" for r in UNIVERSE_GENERALI]


# =========================================
# 5) UI — Constructeur de portefeuille (cartes)
# =========================================
st.title("🟣 Comparer deux portefeuilles (Client vs Vous)")
st.caption("Ajoutez les lignes **proprement** : fonds → ISIN auto, quantité, date d’achat, prix d’achat optionnel.")

# Init state
for key in ["A_lines", "B_lines"]:
    if key not in st.session_state:
        st.session_state[key] = []  # chaque ligne: {name, isin, qty, buy_date, buy_px_opt}

def _parse_float(x: Any) -> Optional[float]:
    if x in (None, "", "—"): return None
    try: return float(str(x).replace(",", "."))
    except: return None

def _resolve_line_symbol(line: Dict[str, Any]) -> Optional[str]:
    # priorise ISIN si fourni
    if line.get("isin"):
        sym = resolve_symbol(str(line["isin"]))
        if sym: return sym
    if line.get("name"):
        sym = resolve_symbol(str(line["name"]))
        if sym: return sym
    return None

def _line_card(line: Dict[str, Any], idx: int, port_key: str):
    col1, col2, col3, col4, col5 = st.columns([3,2,1.4,1.6,0.8])
    with col1:
        st.markdown(f"**{line.get('name','?')}**")
        st.caption(f"ISIN : `{line.get('isin','—')}`")
    with col2:
        st.markdown(f"**Quantité :** {line.get('qty','—')}")
        st.caption(f"Achat : {line.get('buy_date')}")
    with col3:
        st.markdown("Prix achat")
        st.markdown(f"{to_eur(line.get('buy_px_opt')) if line.get('buy_px_opt') else '—'}")
    with col4:
        sym = _resolve_line_symbol(line) or "—"
        st.caption("Symbole EODHD")
        st.code(sym)
    with col5:
        if st.button("🗑️", key=f"del_{port_key}_{idx}", help="Supprimer cette ligne"):
            st.session_state[port_key].pop(idx)
            st.experimental_rerun()


def _add_line_ui(port_key: str, title: str):
    st.subheader(title)

    with st.container(border=True):
        c1, c2 = st.columns([2,1])
        with c1:
            sel = st.selectbox("Choisir un fonds (ou saisie libre) :", UNI_OPTIONS, key=f"{port_key}_select")
        with c2:
            qty = st.number_input("Quantité", min_value=0.0, value=0.0, step=1.0, key=f"{port_key}_qty")

        c3, c4 = st.columns(2)
        with c3:
            dt = st.date_input("Date d’achat", value=date(2024,1,2), key=f"{port_key}_date")
        with c4:
            px_opt = st.text_input("Prix d’achat (optionnel)", value="", key=f"{port_key}_px")

        # Pré-remplissage name/isin
        if sel != "— Saisie libre —":
            name, isin = sel.split(" — ")
        else:
            name = st.text_input("Nom du fonds / Instrument (saisie libre)", key=f"{port_key}_name")
            isin = st.text_input("ISIN (optionnel, conseillé)", key=f"{port_key}_isin")

        # Ajout
        if st.button("➕ Ajouter la ligne", type="primary", key=f"{port_key}_add"):
            if (sel != "— Saisie libre —" and qty > 0) or (name and qty > 0):
                line = {
                    "name": name.strip() if name else "",
                    "isin": isin.strip() if isin else "",
                    "qty": float(qty),
                    "buy_date": pd.Timestamp(dt),
                    "buy_px_opt": _parse_float(px_opt),
                }
                st.session_state[port_key].append(line)
                st.success("Ligne ajoutée.")
            else:
                st.warning("Indique au minimum le fonds et une quantité > 0.")

        # Reset
        st.button("♻️ Vider le portefeuille", key=f"{port_key}_reset", on_click=lambda: st.session_state.update({port_key: []}))

    # Affichage des lignes en cartes
    if st.session_state[port_key]:
        st.markdown("#### Lignes du portefeuille")
        for i, ln in enumerate(st.session_state[port_key]):
            _line_card(ln, i, port_key)
    else:
        st.info("Aucune ligne pour l’instant.")

# UI pour A et B
tabA, tabB = st.tabs(["📁 Portefeuille 1 — Client", "🟣 Portefeuille 2 — Vous"])
with tabA:
    _add_line_ui("A_lines", "Portefeuille 1 — Client")
with tabB:
    _add_line_ui("B_lines", "Portefeuille 2 — Vous")


# =========================================
# 6) CALCUL DES PERFORMANCES
# =========================================
@st.cache_data(ttl=3600, show_spinner=False)
def load_price_series_cached(symbol: str, from_dt: str) -> pd.DataFrame:
    return eod_prices(symbol, from_dt=from_dt)

def compute_portfolio_from_lines(lines: List[Dict[str, Any]], label: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    rows = []
    cash_flows: List[Tuple[pd.Timestamp, float]] = []
    min_buy_dt: Optional[pd.Timestamp] = None
    total_investi, total_valeur = 0.0, 0.0

    for ln in lines:
        name = str(ln.get("name","")).strip()
        isin = str(ln.get("isin","")).strip() or None
        qty  = float(ln.get("qty", 0.0))
        d_buy: pd.Timestamp = ln.get("buy_date")
        px_buy_opt = ln.get("buy_px_opt")

        if not name and not isin:  # rien à résoudre
            continue
        if qty <= 0 or d_buy is None:
            continue

        # Résolution symbole (priorise ISIN)
        sym = resolve_symbol(isin) if isin else resolve_symbol(name)
        if not sym:
            rows.append({
                "Fonds": name or "—", "ISIN": isin or "—", "Symbole": "—",
                "Quantité": qty, "Date achat": d_buy.date(), "Prix achat": px_buy_opt if px_buy_opt else "—",
                "Dernier cours": "ND", "Investi €": np.nan, "Valeur actuelle €": np.nan, "P&L €": np.nan,
                "Perf % depuis achat": np.nan
            })
            continue

        dfp = load_price_series_cached(sym, d_buy.strftime("%Y-%m-%d"))
        last_close = dfp["Close"].iloc[-1] if not dfp.empty else np.nan

        # prix d’achat: optionnel sinon dernier <= date achat
        if px_buy_opt is None:
            px_buy = get_close_on(dfp, d_buy)
            if px_buy is None:
                after = dfp.loc[dfp.index >= d_buy]
                px_buy = float(after["Close"].iloc[0]) if not after.empty else np.nan
        else:
            px_buy = float(px_buy_opt)

        investi = float(qty) * float(px_buy) if pd.notna(px_buy) else np.nan
        valeur  = float(qty) * float(last_close) if pd.notna(last_close) else np.nan
        pl_eur  = valeur - investi if (pd.notna(valeur) and pd.notna(investi)) else np.nan
        perf_pct = (valeur / investi - 1.0) * 100.0 if (pd.notna(valeur) and pd.notna(investi) and investi != 0) else np.nan

        rows.append({
            "Fonds": name or "—",
            "ISIN": isin or "—",
            "Symbole": sym,
            "Quantité": qty,
            "Date achat": d_buy.date(),
            "Prix achat": px_buy if pd.notna(px_buy) else "—",
            "Dernier cours": last_close if pd.notna(last_close) else "ND",
            "Investi €": investi,
            "Valeur actuelle €": valeur,
            "P&L €": pl_eur,
            "Perf % depuis achat": perf_pct
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
# 7) ACTION — COMPARER
# =========================================
st.divider()
run = st.button("🚀 Comparer Portefeuille 1 (Client) vs Portefeuille 2 (Vous)", type="primary")

if run:
    # Calculs
    dfA_lines, aggA = compute_portfolio_from_lines(st.session_state["A_lines"], "Portefeuille 1 — Client")
    dfB_lines, aggB = compute_portfolio_from_lines(st.session_state["B_lines"], "Portefeuille 2 — Vous")

    # Résumé cartes
    st.subheader("📊 Synthèse")
    ca, cb = st.columns(2)
    with ca:
        st.markdown("### 📁 Portefeuille 1 — Client")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Investi", to_eur(aggA["investi_total"]))
        c2.metric("Valeur", to_eur(aggA["valeur_totale"]))
        c3.metric("P&L", to_eur(aggA["pl_total"]))
        c4.metric("Perf", f"{aggA['perf_simple_pct']:.2f}%" if pd.notna(aggA["perf_simple_pct"]) else "—")
        st.markdown(f"**XIRR (annualisé)** : {'' if pd.isna(aggA['xirr_pct']) else f'{aggA['xirr_pct']:.2f}%'}")
        st.caption(f"Depuis : {aggA['min_buy_dt']}")

    with cb:
        st.markdown("### 🟣 Portefeuille 2 — Vous")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Investi", to_eur(aggB["investi_total"]))
        c2.metric("Valeur", to_eur(aggB["valeur_totale"]))
        c3.metric("P&L", to_eur(aggB["pl_total"]))
        c4.metric("Perf", f"{aggB['perf_simple_pct']:.2f}%" if pd.notna(aggB["perf_simple_pct"]) else "—")
        st.markdown(f"**XIRR (annualisé)** : {'' if pd.isna(aggB['xirr_pct']) else f'{aggB['xirr_pct']:.2f}%'}")
        st.caption(f"Depuis : {aggB['min_buy_dt']}")

    # Détail par portefeuille (en cartes + expander table)
    st.subheader("📄 Détail des positions")
    d1, d2 = st.columns(2)

    def _detail_port(df_lines: pd.DataFrame, title: str):
        st.markdown(f"#### {title}")
        if df_lines.empty:
            st.info("Aucune ligne calculable.")
            return
        for _, r in df_lines.iterrows():
            with st.container(border=True):
                c1, c2, c3, c4 = st.columns([3,2,2,2])
                with c1:
                    st.markdown(f"**{r['Fonds']}**")
                    st.caption(f"ISIN : `{r['ISIN']}` • Symbole : `{r['Symbole']}`")
                with c2:
                    st.markdown(f"**Quantité :** {r['Quantité']}")
                    st.caption(f"Achat : {r['Date achat']}")
                with c3:
                    st.markdown(f"Prix achat\n\n**{to_eur(r['Prix achat']) if isinstance(r['Prix achat'], (int,float)) else r['Prix achat']}**")
                    st.caption(f"Dernier : **{to_eur(r['Dernier cours']) if isinstance(r['Dernier cours'], (int,float)) else r['Dernier cours']}**")
                with c4:
                    st.markdown(f"Valeur : **{to_eur(r['Valeur actuelle €'])}**")
                    st.caption(f"P&L : **{to_eur(r['P&L €'])}** ({'' if pd.isna(r['Perf % depuis achat']) else f'{r['Perf % depuis achat']:.2f}%'} )")
        with st.expander("Voir le tableau récapitulatif"):
            st.dataframe(
                df_lines.style.format({
                    "Prix achat": to_eur, "Dernier cours": to_eur,
                    "Investi €": to_eur, "Valeur actuelle €": to_eur,
                    "P&L €": to_eur, "Perf % depuis achat": "{:.2f}%"
                }, na_rep=""),
                use_container_width=True, hide_index=True
            )

    with d1:
        _detail_port(dfA_lines, "Portefeuille 1 — Client")
    with d2:
        _detail_port(dfB_lines, "Portefeuille 2 — Vous")

    # Comparatif visuel
    st.subheader("📈 Comparatif visuel")
    comp = pd.DataFrame([
        {"Portefeuille": "Client", "Investi €": aggA["investi_total"], "Valeur €": aggA["valeur_totale"], "P&L €": aggA["pl_total"], "Perf %": aggA["perf_simple_pct"], "XIRR %": aggA["xirr_pct"]},
        {"Portefeuille": "Vous",   "Investi €": aggB["investi_total"], "Valeur €": aggB["valeur_totale"], "P&L €": aggB["pl_total"], "Perf %": aggB["perf_simple_pct"], "XIRR %": aggB["xirr_pct"]},
    ])
    fig = px.bar(comp, x="Portefeuille", y="Valeur €", text="Valeur €", title="Valeur actuelle par portefeuille")
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(
        comp.style.format({
            "Investi €": to_eur, "Valeur €": to_eur, "P&L €": to_eur,
            "Perf %": "{:.2f}%", "XIRR %": "{:.2f}%"
        }, na_rep=""),
        use_container_width=True, hide_index=True
    )

else:
    st.info("Ajoute des lignes dans chacun des onglets puis clique **Comparer**.")
