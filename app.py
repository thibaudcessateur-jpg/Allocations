# =========================================
# app.py — Comparateur Portefeuilles CGP
# Saisie par MONTANT investi (€), quantité auto = montant / prix d'achat
# =========================================
import os, re, math, requests
from datetime import date
from typing import Optional, List, Dict, Any, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

st.set_page_config(page_title="Comparateur Portefeuilles CGP", page_icon="🦉", layout="wide")
TODAY = pd.Timestamp.today().normalize()


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
# 2) EODHD CLIENT (recherche & prix) — ISIN-first + fallbacks
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
    return bool(re.fullmatch(r"[A-Z]{2}[A-Z0-9]{9}[0-9]", (s or "").strip().upper()))

def _eod_ok(sym: str) -> bool:
    try:
        js = eodhd_get(f"/eod/{sym}", params={"period": "d"})
        return isinstance(js, list) and len(js) > 0
    except Exception:
        return False

@st.cache_data(ttl=24*3600, show_spinner=False)
def resolve_symbol(q: str) -> Optional[str]:
    """
    Résolution centrée ISIN (approche Valeris) :
      1) ISIN -> tester .EUFUND, .FUND, .USFUND
      2) /search(ISIN) — ne garder que l’ISIN exact, priorité {EUFUND, FUND, USFUND}, tester ISIN.EXCH puis Code
      3) Nom -> /search(nom), même logique (ISIN propre si dispo sinon Code)
      4) Dernier recours ISIN -> suffixes actions EU (.PA, .AS, .MI, .DE, .LSE)
    """
    q = (q or "").strip()
    if not q:
        return None

    # ---- 1) ISIN direct
    if _looks_like_isin(q):
        base = q.upper()
        for suf in [".EUFUND", ".FUND", ".USFUND"]:
            cand = f"{base}{suf}"
            if _eod_ok(cand):
                return cand

        # ---- 2) Recherche stricte ISIN exact
        res = eod_search(base)
        if res:
            preferred = ["EUFUND", "FUND", "USFUND"]
            exact = [it for it in res if str(it.get("ISIN", "")).upper() == base]
            for exch in preferred:
                for it in exact:
                    ex = str(it.get("Exchange", "")).upper()
                    if ex == exch:
                        cand = f"{base}.{ex}"
                        if _eod_ok(cand):
                            return cand
                        code = str(it.get("Code", "")).strip()
                        if code and _eod_ok(code):
                            return code
            for it in exact:
                code = str(it.get("Code", "")).strip()
                if code and _eod_ok(code):
                    return code

        for suf in [".PA", ".AS", ".MI", ".DE", ".LSE"]:
            cand = f"{base}{suf}"
            if _eod_ok(cand):
                return cand
        return None

    # ---- 3) Nom
    res = eod_search(q)
    if res:
        preferred = ["EUFUND", "FUND", "USFUND"]
        for exch in preferred:
            for it in res:
                ex = str(it.get("Exchange", "")).upper()
                if ex == exch:
                    isin = str(it.get("ISIN", "")).upper()
                    if _looks_like_isin(isin):
                        cand = f"{isin}.{ex}"
                        if _eod_ok(cand):
                            return cand
        for it in res:
            code = str(it.get("Code", "")).strip()
            if code and _eod_ok(code):
                return code
    return None

# ------- Récupération de VL: super fallback
PREFERRED_FUND_EXCH = [".EUFUND", ".FUND", ".USFUND"]
ALT_EQUITY_EXCH = [".PA", ".AS", ".MI", ".DE", ".LSE"]

@st.cache_data(ttl=3*3600, show_spinner=False)
def eod_prices_any(symbol_or_isin: str, start_dt: Optional[pd.Timestamp] = None) -> Tuple[pd.DataFrame, str, str]:
    """
    Retourne (df_prices, symbol_used, note)
    df_prices: colonnes ['Close'] indexées par date
    Stratégie:
      1) /eod sans 'from', puis avec 'from'
      2) Variantes .EUFUND/.FUND/.USFUND, puis suffixes actions
      3) Fallback: dernier prix depuis /search en série 1 point
    """
    q = (symbol_or_isin or "").strip().upper()
    note = ""

    def _fetch(sym: str, from_dt: Optional[str]) -> pd.DataFrame:
        params = {"period": "d"}
        if from_dt:
            params["from"] = from_dt
        js = eodhd_get(f"/eod/{sym}", params=params)
        df = pd.DataFrame(js)
        if df.empty:
            return pd.DataFrame()
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date").sort_index()
        df["close"] = pd.to_numeric(df["close"], errors="coerce")
        return df[["close"]].rename(columns={"close": "Close"})

    def _try_with_without_from(sym: str, from_ts: Optional[pd.Timestamp]) -> Optional[pd.DataFrame]:
        df = _fetch(sym, None)
        if not df.empty:
            if from_ts is not None:
                df = df.loc[df.index >= from_ts]
            if not df.empty:
                return df
        f = from_ts.strftime("%Y-%m-%d") if from_ts is not None else None
        df = _fetch(sym, f)
        if not df.empty:
            return df
        return None

    # déjà un symbole ?
    if "." in q and not _looks_like_isin(q.split(".")[0]):
        df = _try_with_without_from(q, start_dt)
        if df is not None:
            return df, q, note

    base = q.split(".")[0]
    if _looks_like_isin(base):
        sym_res = resolve_symbol(base)
        if sym_res:
            df = _try_with_without_from(sym_res, start_dt)
            if df is not None:
                return df, sym_res, note

        for suf in PREFERRED_FUND_EXCH:
            sym = f"{base}{suf}"
            df = _try_with_without_from(sym, start_dt)
            if df is not None:
                return df, sym, note

        for suf in ALT_EQUITY_EXCH:
            sym = f"{base}{suf}"
            df = _try_with_without_from(sym, start_dt)
            if df is not None:
                return df, sym, note

    # Fallback: dernier cours via /search
    srch = eod_search(base if _looks_like_isin(base) else q)
    last_px, last_dt = None, None
    if srch:
        choices = srch
        if _looks_like_isin(base):
            choices = [it for it in srch if str(it.get("ISIN", "")).upper() == base] or srch
        it0 = choices[0]
        last_px = it0.get("previousClose")
        last_dt = it0.get("previousCloseDate")
    if last_px is not None:
        try:
            last_px = float(last_px)
            last_dt = pd.to_datetime(last_dt).normalize() if last_dt else TODAY
            df = pd.DataFrame({"Close": [last_px]}, index=[last_dt])
            note = "⚠️ Historique VL indisponible via /eod — utilisation du dernier cours (/search)."
            used = resolve_symbol(base) or base
            return df, used, note
        except Exception:
            pass

    return pd.DataFrame(), q, "⚠️ Aucune VL récupérée (symbole introuvable ou non couvert par l’API)."

@st.cache_data(ttl=3600, show_spinner=False)
def load_price_series_any(symbol_or_isin: str, from_dt: Optional[pd.Timestamp]) -> Tuple[pd.DataFrame, str, str]:
    return eod_prices_any(symbol_or_isin, from_dt)


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
# 4) UNIVERS — Core + Défensifs (pré-sélection)
# =========================================
UNIVERSE_GENERALI = [
    {"name": "R-co Valor C EUR", "isin": "FR0011253624", "type": "Actions Monde"},
    {"name": "Vivalor International", "isin": "FR0014001LS1", "type": "Actions Monde"},
    {"name": "CARMIGNAC Investissement A EUR", "isin": "FR0010148981", "type": "Actions Monde"},
    {"name": "FIDELITY Funds - World Fund", "isin": "LU0069449576", "type": "Actions Monde"},
    {"name": "CLARTAN Valeurs C", "isin": "LU1100076550", "type": "Actions Europe"},
    {"name": "CARMIGNAC Patrimoine", "isin": "FR0010135103", "type": "Diversifié patrimonial"},
    {"name": "SYCOYIELD 2030 RC", "isin": "FR001400MCQ6", "type": "Obligataire échéance"},
    {"name": "R-Co Target 2029 HY", "isin": "FR001400AWH8", "type": "Obligataire haut rendement"},  # à confirmer
]
UNI_OPTIONS = ["— Saisie libre —"] + [f"{r['name']} — {r['isin']}" for r in UNIVERSE_GENERALI]


# =========================================
# 5) UI — Constructeur de portefeuilles (cartes)
# =========================================
st.title("🟣 Comparer deux portefeuilles (Client vs Vous)")
st.caption("**Montant investi (€)** par ligne (la quantité est calculée automatiquement). ISIN seul accepté.")

for key in ["A_lines", "B_lines"]:
    if key not in st.session_state:
        st.session_state[key] = []  # {name, isin, amount, buy_date, buy_px_opt, sym_used, note}

def _parse_float(x: Any) -> Optional[float]:
    if x in (None, "", "—"): return None
    try: return float(str(x).replace(",", "."))
    except: return None

def _line_card(line: Dict[str, Any], idx: int, port_key: str):
    col1, col2, col3, col4, col5, col6 = st.columns([3,1.6,1.6,1.6,1.4,0.8])
    with col1:
        st.markdown(f"**{line.get('name','?')}**")
        st.caption(f"ISIN : `{line.get('isin','—')}`")
    with col2:
        st.markdown(f"**Montant investi :** {to_eur(line.get('amount')) if line.get('amount') else '—'}")
        st.caption(f"Achat : {line.get('buy_date')}")
    with col3:
        st.markdown("Prix achat")
        st.markdown(f"{to_eur(line.get('buy_px_opt')) if line.get('buy_px_opt') else 'auto (VL)'}")
    with col4:
        st.markdown("Quantité (calculée)")
        q = line.get("qty_calc")
        st.markdown(f"{q:.6f}" if isinstance(q, (int,float)) and not pd.isna(q) else "—")
    with col5:
        st.caption("Symbole utilisé")
        st.code(line.get("sym_used", "—"))
    with col6:
        if st.button("🗑️", key=f"del_{port_key}_{idx}", help="Supprimer cette ligne"):
            st.session_state[port_key].pop(idx)
            st.experimental_rerun()

def _add_line_ui(port_key: str, title: str):
    st.subheader(title)

    with st.form(key=f"{port_key}_form", clear_on_submit=False):
        c1, c2 = st.columns([2,1])
        with c1:
            sel = st.selectbox("Choisir un fonds (ou saisie libre) :", UNI_OPTIONS, key=f"{port_key}_select")
        with c2:
            amount = st.text_input("Montant investi (€)", value="", key=f"{port_key}_amount", help="Ex: 10 000 ou 10000,00")

        c3, c4 = st.columns(2)
        with c3:
            dt = st.date_input("Date d’achat", value=date(2024,1,2), key=f"{port_key}_date")
        with c4:
            px_opt = st.text_input("Prix d’achat (optionnel, sinon VL)", value="", key=f"{port_key}_px")

        if sel != "— Saisie libre —":
            name, isin = sel.split(" — ")
            name = name.strip()
            isin = isin.strip().upper()
        else:
            name = st.text_input("Nom du fonds / Instrument (facultatif)", key=f"{port_key}_name").strip()
            isin = st.text_input("ISIN (recommandé)", key=f"{port_key}_isin").strip().upper()

        submitted = st.form_submit_button("➕ Ajouter la ligne", type="primary")

    if submitted:
        amt = _parse_float(amount)
        if not amt or amt <= 0:
            st.warning("Entre un **montant investi (€)** strictement positif.")
            st.stop()

        valid_free = (bool(isin) or bool(name))
        valid_list = (sel != "— Saisie libre —")
        if not (valid_free or valid_list):
            st.warning("Indique au minimum le fonds (ISIN ou Nom).")
            st.stop()

        try:
            buy_px_opt = float(str(px_opt).replace(",", ".")) if px_opt else None
        except Exception:
            buy_px_opt = None

        # On charge la série dès maintenant (multi-fallback) pour:
        #  - récupérer la VL d'achat
        #  - calculer la quantité
        from_ts = pd.Timestamp(dt)
        dfp, sym_used, note = load_price_series_any(isin or name, from_ts)
        if dfp.empty:
            st.error("Impossible de récupérer des VL via EODHD pour ce fonds.")
            st.info("Astuce: vérifie l’ISIN exact. L’app teste .EUFUND, .FUND, .USFUND et quelques places actions.")
            st.stop()

        # prix d’achat
        if buy_px_opt is None:
            # tente VL du jour d'achat, sinon première après
            if from_ts in dfp.index:
                px_buy = float(dfp.loc[from_ts, "Close"])
            else:
                after = dfp.loc[dfp.index >= from_ts]
                px_buy = float(after["Close"].iloc[0]) if not after.empty else float(dfp["Close"].iloc[0])
        else:
            px_buy = float(buy_px_opt)

        if px_buy <= 0 or pd.isna(px_buy):
            st.error("Prix d’achat non déterminable (VL indisponible).")
            st.stop()

        qty_calc = float(amt) / float(px_buy)

        line = {
            "name": name or "—",
            "isin": isin or "",
            "amount": float(amt),
            "qty_calc": qty_calc,
            "buy_date": pd.Timestamp(dt),
            "buy_px_opt": buy_px_opt,
            "sym_used": sym_used,
            "note": note,
        }
        st.session_state[port_key].append(line)
        st.success(f"Ligne ajoutée ({sym_used}).")
        if note:
            st.caption(note)

    if st.session_state[port_key]:
        st.markdown("#### Lignes du portefeuille")
        for i, ln in enumerate(st.session_state[port_key]):
            _line_card(ln, i, port_key)
            if ln.get("note"):
                st.caption(ln["note"])
    else:
        st.info("Aucune ligne pour l’instant.")


# UI pour A et B
tabA, tabB = st.tabs(["📁 Portefeuille 1 — Client", "🟣 Portefeuille 2 — Vous"])
with tabA:
    _add_line_ui("A_lines", "Portefeuille 1 — Client")
with tabB:
    _add_line_ui("B_lines", "Portefeuille 2 — Vous")


# =========================================
# 6) CALCUL DES PERFORMANCES (avec 'amount' prioritaire)
# =========================================
@st.cache_data(ttl=3600, show_spinner=False)
def get_series_for_line(line: Dict[str, Any]) -> pd.DataFrame:
    dfp, _, _ = load_price_series_any(line.get("isin") or line.get("name"), pd.Timestamp(line["buy_date"]))
    return dfp

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

def compute_portfolio_from_lines(lines: List[Dict[str, Any]], label: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    rows = []
    cash_flows: List[Tuple[pd.Timestamp, float]] = []
    min_buy_dt: Optional[pd.Timestamp] = None
    total_investi, total_valeur = 0.0, 0.0

    for ln in lines:
        name = str(ln.get("name","")).strip()
        isin = str(ln.get("isin","")).strip() or None
        amount = float(ln.get("amount", 0.0) or 0.0)   # ✅ montant investi prioritaire
        qty_calc = float(ln.get("qty_calc") or 0.0)
        d_buy: pd.Timestamp = ln.get("buy_date")
        px_buy_opt = ln.get("buy_px_opt")
        sym_used = ln.get("sym_used", "—")

        if (not name and not isin) or amount <= 0 or d_buy is None:
            continue

        dfp = get_series_for_line(ln)
        last_close = dfp["Close"].iloc[-1] if not dfp.empty else np.nan

        # prix d'achat réel (si px_buy_opt non fourni)
        if px_buy_opt is None:
            px_buy = get_close_on(dfp, d_buy)
            if px_buy is None:
                after = dfp.loc[dfp.index >= d_buy]
                px_buy = float(after["Close"].iloc[0]) if not after.empty else float(dfp["Close"].iloc[0])
        else:
            px_buy = float(px_buy_opt)

        # quantité utilisée pour la valorisation : recompute si besoin pour robustesse
        qty_eff = qty_calc if qty_calc > 0 else (amount / px_buy if px_buy else np.nan)

        investi = amount  # ✅ on respecte exactement le montant saisi
        valeur  = float(qty_eff) * float(last_close) if (pd.notna(last_close) and not pd.isna(qty_eff)) else np.nan
        pl_eur  = valeur - investi if (pd.notna(valeur) and pd.notna(investi)) else np.nan
        perf_pct = (valeur / investi - 1.0) * 100.0 if (pd.notna(valeur) and pd.notna(investi) and investi != 0) else np.nan

        rows.append({
            "Fonds": name or "—",
            "ISIN": isin or "—",
            "Symbole": sym_used,
            "Montant investi €": investi,
            "Quantité": qty_eff,
            "Date achat": d_buy.date(),
            "Prix achat": px_buy if pd.notna(px_buy) else "—",
            "Dernier cours": last_close if pd.notna(last_close) else "ND",
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
    dfA_lines, aggA = compute_portfolio_from_lines(st.session_state["A_lines"], "Portefeuille 1 — Client")
    dfB_lines, aggB = compute_portfolio_from_lines(st.session_state["B_lines"], "Portefeuille 2 — Vous")

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

    st.subheader("📄 Détail des positions")
    d1, d2 = st.columns(2)

    def _detail_port(df_lines: pd.DataFrame, title: str):
        st.markdown(f"#### {title}")
        if df_lines.empty:
            st.info("Aucune ligne calculable.")
            return
        for _, r in df_lines.iterrows():
            with st.container(border=True):
                c1, c2, c3, c4 = st.columns([3,2.2,2.2,2.2])
                with c1:
                    st.markdown(f"**{r['Fonds']}**")
                    st.caption(f"ISIN : `{r['ISIN']}` • Symbole : `{r['Symbole']}`")
                with c2:
                    st.markdown(f"Investi\n\n**{to_eur(r['Montant investi €'])}**")
                    st.caption(f"Quantité : **{(r['Quantité'] if pd.notna(r['Quantité']) else '—')}**")
                with c3:
                    st.markdown(f"Prix achat\n\n**{to_eur(r['Prix achat']) if isinstance(r['Prix achat'], (int,float)) else r['Prix achat']}**")
                    st.caption(f"Dernier : **{to_eur(r['Dernier cours']) if isinstance(r['Dernier cours'], (int,float)) else r['Dernier cours']}**")
                with c4:
                    st.markdown(f"Valeur : **{to_eur(r['Valeur actuelle €'])}**")
                    st.caption(f"P&L : **{to_eur(r['P&L €'])}** ({'' if pd.isna(r['Perf % depuis achat']) else f'{r['Perf % depuis achat']:.2f}%'} )")
        with st.expander("Voir le tableau récapitulatif"):
            st.dataframe(
                df_lines.style.format({
                    "Montant investi €": to_eur, "Quantité": "{:.6f}",
                    "Prix achat": to_eur, "Dernier cours": to_eur,
                    "Valeur actuelle €": to_eur, "P&L €": to_eur,
                    "Perf % depuis achat": "{:.2f}%"
                }, na_rep=""),
                use_container_width=True, hide_index=True
            )

    with d1:
        _detail_port(dfA_lines, "Portefeuille 1 — Client")
    with d2:
        _detail_port(dfB_lines, "Portefeuille 2 — Vous")

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
    st.info("Ajoute des lignes avec un **montant investi (€)**, puis clique **Comparer**.")


# =========================================
# 8) Debug (optionnel)
# =========================================
with st.expander("🔧 Debug EODHD (optionnel)"):
    test_q = st.text_input("Tester une recherche (ISIN ou nom)")
    if test_q:
        st.write("Résultat /search :", eod_search(test_q))
        df_dbg, sym_dbg, note_dbg = load_price_series_any(test_q, None)
        st.write("Symbole testé :", sym_dbg)
        if note_dbg:
            st.caption(note_dbg)
        if not df_dbg.empty:
            st.dataframe(df_dbg.tail(5))
        else:
            st.warning("Aucune VL trouvée.")
