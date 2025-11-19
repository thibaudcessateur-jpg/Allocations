# app.py
# ------------------------------------------------------------
# Comparateur de deux portefeuilles (Client vs Vous)
# - Ajout/modif de lignes (ISIN, montant brut, date, px achat optionnel)
# - Coller un tableau (Nom;ISIN;Montant)
# - Fonds en euros avec taux paramétrable (symbole: EUROFUND)
# - Frais d'entrée propres à chaque portefeuille
# - Versements (mensuels + ponctuel, allocation configurable)
# - Moteur de valorisation vectorisé (prix x quantités)
# - Rendu graphique + XIRR + synthèse & "Et si c'était avec nous ?"
# - Récupération VL via EODHD (recherche robuste)
# ------------------------------------------------------------

from __future__ import annotations

import os
import math
import json
from datetime import date, datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
import streamlit as st
import altair as alt

# ----------------------- Constantes -------------------------
TODAY = pd.Timestamp.today().normalize()
APP_TITLE = "Comparateur de portefeuilles"

# ----------------------- Utils format -----------------------
def to_eur(x: Any) -> str:
    try:
        v = float(x)
    except Exception:
        return "—"
    # format fr: espace groupe + virgule
    s = f"{v:,.2f}".replace(",", "X").replace(".", ",").replace("X", " ")
    return s + " €"

def fmt_date(x: Any) -> str:
    try:
        return pd.Timestamp(x).strftime("%d/%m/%Y")
    except Exception:
        return "—"

# ----------------------- XIRR -------------------------------
def _npv(rate: float, cfs: List[Tuple[pd.Timestamp, float]]) -> float:
    t0 = cfs[0][0]
    return sum(cf / ((1 + rate) ** ((t - t0).days / 365.25)) for t, cf in cfs)

def xirr(cash_flows: List[Tuple[pd.Timestamp, float]], guess: float = 0.1) -> Optional[float]:
    if not cash_flows:
        return None
    cfs = sorted(cash_flows, key=lambda x: x[0])
    # Brent simple / Newton hybrid
    try:
        r = guess
        for _ in range(100):
            f = _npv(r, cfs)
            # dérivée numérique
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

# ----------------------- EODHD API --------------------------
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
    """Télécharge les cours journaliers d'un symbole EODHD."""
    try:
        js = eodhd_get(f"/eod/{symbol}", params={"period": "d"})
        if not isinstance(js, list) or len(js) == 0:
            return pd.DataFrame()
        df = pd.DataFrame(js)
        df["date"] = pd.to_datetime(df["date"])
        df.set_index("date", inplace=True)
        # Normalise la colonne Close
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
    """
    Stratégie robuste:
      - EUROFUND => prix 1.0 synthétique
      - ISIN.EUFUND / .FUND / .PA / .MI etc. via /search
      - Fallback: essayer ISIN.EUFUND et ISIN tout court
    """
    val = str(isin_or_name).strip()
    if not val:
        return []
    if val.upper() == "EUROFUND":
        return ["EUROFUND"]

    # Si ISIN pur
    candidates = [f"{val}.EUFUND", f"{val}.FUND", val]

    # Recherche EODHD
    try:
        res = eodhd_search(val)
        # priorité aux fonds
        for it in res:
            code = it.get("Code")
            exch = it.get("Exchange")
            if code and exch:
                candidates.append(f"{code}.{exch}")
    except Exception:
        pass

    # unicité
    seen = set()
    uniq = []
    for c in candidates:
        if c not in seen:
            seen.add(c)
            uniq.append(c)
    return uniq

def _get_close_on(df: pd.DataFrame, d: pd.Timestamp) -> float:
    """Prix d'achat au jour d'achat (ou 1er jour >= achat)."""
    if df.empty:
        return np.nan
    if d in df.index:
        return float(df.loc[d, "Close"])
    after = df.loc[df.index >= d]
    if not after.empty:
        return float(after.iloc[0]["Close"])
    return float(df.iloc[-1]["Close"])

@st.cache_data(show_spinner=False, ttl=3600)
def get_price_series(isin_or_name: str, start: Optional[pd.Timestamp], euro_rate: float) -> Tuple[pd.DataFrame, str, str]:
    """
    Retourne (df Close, symbol_utilisé, debug) pour un ISIN/Nom.
    EUROFUND => prix constant 1.0.
    """
    debug = {"cands": []}
    val = str(isin_or_name).strip()
    if not val:
        return pd.DataFrame(), "", json.dumps(debug)

    if val.upper() == "EUROFUND":
        idx = pd.bdate_range(start=pd.Timestamp("2000-01-03"), end=TODAY, freq="B")
        df = pd.DataFrame({"Close": 1.0}, index=idx)
        if start is not None:
            df = df.loc[df.index >= start]
        return df, "EUROFUND", json.dumps(debug)

    cands = _symbol_candidates(val)
    debug["cands"] = cands
    for sym in cands:
        df = eodhd_prices_daily(sym)
        if not df.empty:
            if start is not None:
                df = df.loc[df.index >= start]
            return df, sym, json.dumps(debug)
    return pd.DataFrame(), "", json.dumps(debug)

# ----------------------- Schedules & poids ------------------
def _month_schedule(d0: pd.Timestamp, d1: pd.Timestamp) -> List[pd.Timestamp]:
    """1er jour ouvré de chaque mois entre d0 et d1 inclus."""
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

def _weights_for(lines: List[Dict[str, Any]],
                 alloc_mode: str,
                 custom_weights: Dict[int, float],
                 single_target: Optional[int]) -> Dict[int, float]:
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
    # default
    w = 1.0 / len(keys)
    return {k: w for k in keys}

# ----------------------- Métriques ligne (frais dynamiques) -
def compute_line_metrics(line: Dict[str, Any], fee_pct: float, euro_rate: float) -> Tuple[float, float, float]:
    """
    (net_investi, prix_achat, quantite) en appliquant le % de frais fourni.
    EUROFUND => prix 1.0.
    """
    amt_brut = float(line.get("amount_gross", 0.0))
    net_amt  = amt_brut * (1.0 - fee_pct / 100.0)

    buy_ts = pd.Timestamp(line.get("buy_date"))
    px_manual = line.get("buy_px", None)

    if str(line.get("sym_used", "")).upper() == "EUROFUND" or str(line.get("isin", "")).upper() == "EUROFUND":
        px = 1.0
    else:
        dfp, _, _ = get_price_series(line.get("isin") or line.get("name"), buy_ts, euro_rate)
        if px_manual not in (None, "", 0, "0"):
            try:
                px = float(px_manual)
            except Exception:
                px = _get_close_on(dfp, buy_ts)
        else:
            px = _get_close_on(dfp, buy_ts)

    qty = (net_amt / px) if px and px > 0 else 0.0
    return float(net_amt), float(px), float(qty)

# ----------------------- Simulation vectorisée --------------
def simulate_portfolio(
    lines: List[Dict[str, Any]],
    monthly_amt_gross: float,
    one_amt_gross: float, one_date: date,
    alloc_mode: str,
    custom_weights: Dict[int, float],
    single_target: Optional[int],
    euro_rate: float,
    fee_pct: float
) -> Tuple[pd.DataFrame, float, float, float, Optional[float], pd.Timestamp, pd.Timestamp]:

    if not lines:
        return pd.DataFrame(), 0.0, 0.0, 0.0, None, TODAY, TODAY

    # 1) Prix de chaque ligne + dates d'achat effectives
    price_map: Dict[int, pd.Series] = {}
    eff_buy_date: Dict[int, pd.Timestamp] = {}
    buy_price_used: Dict[int, float] = {}

    for ln in lines:
        key_id = id(ln)
        df, sym, _ = get_price_series(ln.get("isin") or ln.get("name"), None, euro_rate)
        if df.empty:
            continue
        ln["sym_used"] = sym

        d_buy = pd.Timestamp(ln["buy_date"])
        if d_buy in df.index:
            px_buy = float(df.loc[d_buy, "Close"]); eff_dt = d_buy
        else:
            after = df.loc[df.index >= d_buy]
            if after.empty:
                px_buy = float(df.iloc[-1]["Close"]); eff_dt = df.index[-1]
            else:
                px_buy = float(after.iloc[0]["Close"]); eff_dt = after.index[0]

        px_manual = ln.get("buy_px", None)
        px_for_qty = float(px_manual) if (px_manual not in (None, "", 0, "0")) else px_buy

        price_map[key_id] = df["Close"].astype(float)
        eff_buy_date[key_id] = eff_dt
        buy_price_used[key_id] = px_for_qty

    if not price_map:
        return pd.DataFrame(), 0.0, 0.0, 0.0, None, TODAY, TODAY

    start_min = min(eff_buy_date.values())
    start_full = max(eff_buy_date.values())

    # 2) Calendrier business-day commun + ffill
    bidx = pd.bdate_range(start=start_min, end=TODAY, freq="B")
    prices = pd.DataFrame(index=bidx)
    for key_id, s in price_map.items():
        prices[key_id] = s.reindex(bidx).ffill()

    # 3) Achats initiaux (quantités) – frais dynamiques
    qty_events = pd.DataFrame(0.0, index=bidx, columns=prices.columns)
    total_brut = 0.0
    total_net = 0.0
    cash_flows: List[Tuple[pd.Timestamp, float]] = []

    for ln in lines:
        key_id = id(ln)
        if key_id not in prices.columns:
            continue
        brut = float(ln.get("amount_gross", 0.0))
        net  = brut * (1.0 - fee_pct/100.0)
        px   = float(buy_price_used[key_id])
        dt   = eff_buy_date[key_id]
        if brut > 0 and px > 0:
            q = net / px
            tgt = dt if dt in qty_events.index else qty_events.index[qty_events.index >= dt][0]
            qty_events.loc[tgt, key_id] += q
            total_brut += brut
            total_net  += net
            cash_flows.append((tgt, -brut))

    # 4) Poids d'allocation
    weights = _weights_for(lines, alloc_mode, custom_weights, single_target)

    # 5) Versement ponctuel
    if one_amt_gross > 0:
        dt = pd.Timestamp(one_date)
        if dt not in qty_events.index:
            after = qty_events.index[qty_events.index >= dt]
            if len(after) > 0:
                dt = after[0]
            else:
                dt = None
        if dt is not None:
            net_amt = one_amt_gross * (1.0 - fee_pct/100.0)
            for ln in lines:
                key_id = id(ln)
                w = weights.get(key_id, 0.0)
                if w <= 0 or key_id not in prices.columns:
                    continue
                px = float(prices.loc[dt, key_id])
                if px > 0:
                    qty_events.loc[dt, key_id] += (net_amt * w) / px
            total_brut += float(one_amt_gross)
            total_net  += float(net_amt)
            cash_flows.append((dt, -float(one_amt_gross)))

    # 6) Mensualités
    if monthly_amt_gross > 0:
        sched = _month_schedule(start_min, TODAY)
        for dt in sched:
            if dt not in qty_events.index:
                after = qty_events.index[qty_events.index >= dt]
                if len(after) == 0:
                    continue
                dt = after[0]
            net_m = monthly_amt_gross * (1.0 - fee_pct/100.0)
            for ln in lines:
                key_id = id(ln)
                w = weights.get(key_id, 0.0)
                if w <= 0 or key_id not in prices.columns:
                    continue
                px = float(prices.loc[dt, key_id])
                if px > 0:
                    qty_events.loc[dt, key_id] += (net_m * w) / px
            total_brut += float(monthly_amt_gross)
            total_net  += float(net_m)
            cash_flows.append((dt, -float(monthly_amt_gross)))

    # 7) Quantités cumulées + valeur
    qty_cum = qty_events.cumsum()
    values = (qty_cum * prices).sum(axis=1)
    df_val = pd.DataFrame({"Valeur": values})

    final_val = float(df_val["Valeur"].iloc[-1]) if not df_val.empty else 0.0
    cash_flows.append((TODAY, final_val))
    irr = xirr(cash_flows)

    return df_val, total_brut, total_net, final_val, (irr * 100.0 if irr is not None else None), start_min, start_full

# ----------------------- UI: Ligne (carte) ------------------
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
            st.caption(f"ISIN : `{line.get('isin','—')}` • Symbole :")
            st.code(line.get('sym_used','—'))
        with cols[1]:
            st.markdown(f"Investi (brut)\n\n**{to_eur(line.get('amount_gross', 0.0))}**")
            st.caption(f"Net après frais {fee_pct:.1f}% : **{to_eur(net_amt)}**")
            st.caption(f"le {fmt_date(line.get('buy_date'))}")
        with cols[2]:
            st.markdown(f"Prix achat\n\n**{to_eur(buy_px)}**")
            st.caption(f"Quantité : {qty_disp:.6f}")
            if line.get("note"):
                st.caption(line["note"])
        with cols[3]:
            try:
                dfl, _, _ = get_price_series(line.get("isin") or line.get("name"), None, euro_rate)
                last = float(dfl["Close"].iloc[-1]) if not dfl.empty else np.nan
                st.markdown(f"Dernier : **{to_eur(last)}**")
            except Exception:
                st.markdown("Dernier : —")
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
                    st.session_state[state_key] = False
                    st.success("Ligne mise à jour.")
                    st.experimental_rerun()

# ----------------------- UI: Table positions ----------------
def positions_table(title: str, port_key: str):
    fee_pct = st.session_state.get("FEE_A", 0.0) if port_key == "A_lines" else st.session_state.get("FEE_B", 0.0)
    euro_rate = st.session_state.get("EURO_RATE_PREVIEW", 2.0)
    lines = st.session_state.get(port_key, [])

    rows = []
    for ln in lines:
        net_amt, buy_px, qty = compute_line_metrics(ln, fee_pct, euro_rate)
        rows.append({
            "Nom": ln.get("name", ""),
            "ISIN": ln.get("isin", ""),
            "Symbole": ln.get("sym_used", ""),
            "Montant brut €": float(ln.get("amount_gross", 0.0)),
            "Net investi €": net_amt,
            "Quantité": qty,
            "Prix achat €": buy_px,
            "Date d'achat": fmt_date(ln.get("buy_date")),
        })

    df = pd.DataFrame(rows)
    st.markdown(f"### {title}")
    if df.empty:
        st.info("Aucune ligne.")
    else:
        st.dataframe(
            df.style.format({
                "Montant brut €": to_eur,
                "Net investi €": to_eur,
                "Prix achat €": to_eur,
                "Quantité": "{:,.6f}".format
            }),
            hide_index=True,
            use_container_width=True
        )

# ----------------------- UI: Sidebar Saisie -----------------
def _add_line_form(port_key: str, label: str):
    st.subheader(label)
    with st.form(key=f"form_add_{port_key}", clear_on_submit=False):
        c1, c2 = st.columns([3, 2])
        with c1:
            name = st.text_input("Nom du fonds (ou 'EUROFUND')", value="")
            isin = st.text_input("ISIN (peut être 'EUROFUND')", value="")
        with c2:
            amount = st.text_input("Montant investi (brut) €", value="")
            buy_date = st.date_input("Date d’achat", value=pd.Timestamp("2024-01-02").date())
        px = st.text_input("Prix d’achat (optionnel)", value="")
        note = st.text_input("Note (optionnel)", value="")
        add_btn = st.form_submit_button("➕ Ajouter cette ligne")

    if add_btn:
        try:
            amt = float(str(amount).replace(" ", "").replace(",", "."))
            assert amt > 0
        except Exception:
            st.warning("Montant invalide.")
            return
        ln = {
            "name": name.strip() or isin.strip() or "—",
            "isin": isin.strip() or name.strip(),
            "amount_gross": float(amt),
            "buy_date": pd.Timestamp(buy_date),
            "buy_px": float(str(px).replace(",", ".")) if px.strip() else "",
            "note": note.strip(),
            "sym_used": ""  # renseigné lors des téléchargements de prix
        }
        st.session_state[port_key].append(ln)
        st.success("Ligne ajoutée.")

def _paste_table_block(port_key: str, label: str):
    st.subheader(f"{label} — Coller un tableau")
    st.caption("Format attendu (une ligne par position) : **Nom;ISIN;Montant** (séparateur `;`).")
    txt = st.text_area("Coller ici :", value="", height=120, key=f"paste_{port_key}")
    if st.button("👀 Prévisualiser", key=f"prev_{port_key}"):
        rows = []
        for raw in txt.strip().splitlines():
            parts = [p.strip() for p in raw.split(";")]
            if len(parts) < 3:
                continue
            nm, isinn, amt = parts[0], parts[1], parts[2]
            try:
                a = float(str(amt).replace(" ", "").replace(",", "."))
            except Exception:
                a = 0.0
            rows.append({"name": nm, "isin": isinn, "amount_gross": a})
        st.session_state[f"preview_{port_key}"] = rows

    prev = st.session_state.get(f"preview_{port_key}", [])
    if prev:
        st.success(f"{len(prev)} ligne(s) détectée(s).")
        dfp = pd.DataFrame(prev)
        st.dataframe(dfp, hide_index=True, use_container_width=True)
        if st.button("➕ Ajouter ces lignes", key=f"addprev_{port_key}"):
            for r in prev:
                st.session_state[port_key].append({
                    "name": r["name"],
                    "isin": r["isin"],
                    "amount_gross": float(r["amount_gross"]),
                    "buy_date": pd.Timestamp("2024-01-02"),
                    "buy_px": "",
                    "note": "",
                    "sym_used": ""
                })
            st.session_state[f"preview_{port_key}"] = []
            st.success("Lignes ajoutées.")

# ----------------------- App --------------------------------
st.set_page_config(page_title=APP_TITLE, layout="wide")
st.title(APP_TITLE)

# Etat
for key in ["A_lines", "B_lines"]:
    if key not in st.session_state:
        st.session_state[key] = []

# --------- Sidebar: paramètres globaux ----------
with st.sidebar:
    st.header("🟦 Fonds en euros — Paramètre global")
    EURO_RATE = st.number_input("Taux annuel du fonds en euros (%)", 0.0, 10.0, 2.0, 0.10, key="EURO_RATE_PREVIEW")
    st.caption("Astuce : utilisez le symbole **EUROFUND** pour ajouter le fonds en euros.")

    st.header("💶 Frais d’entrée (%)")
    FEE_A = st.number_input("Frais d’entrée — Portefeuille 1 (Client)", 0.0, 10.0, 3.0, 0.10, key="FEE_A")
    FEE_B = st.number_input("Frais d’entrée — Portefeuille 2 (Vous)",   0.0, 10.0, 2.0, 0.10, key="FEE_B")
    st.caption("Ex.: 10 000 € investis avec 4 % de frais ⇒ 9 600 € réellement investis. Les frais s’appliquent aussi aux versements.")

    st.header("📅 Paramètres de versement")
    with st.expander("Portefeuille 1 — Client"):
        M_A = st.number_input("Mensuel brut (€)", 0.0, 1_000_000.0, 0.0, 100.0, key="M_A")
        ONE_A = st.number_input("Ponctuel brut (€)", 0.0, 1_000_000.0, 0.0, 100.0, key="ONE_A")
        ONE_A_DATE = st.date_input("Date du ponctuel", value=pd.Timestamp("2024-07-01").date(), key="ONE_A_DATE")
    with st.expander("Portefeuille 2 — Vous"):
        M_B = st.number_input("Mensuel brut (€)", 0.0, 1_000_000.0, 0.0, 100.0, key="M_B")
        ONE_B = st.number_input("Ponctuel brut (€)", 0.0, 1_000_000.0, 0.0, 100.0, key="ONE_B")
        ONE_B_DATE = st.date_input("Date du ponctuel", value=pd.Timestamp("2024-07-01").date(), key="ONE_B_DATE")

    st.header("🧮 Règle d’affectation des versements")
    ALLOC_MODE = st.selectbox("Mode", ["equal", "custom", "single"], index=0, help="Répartition des versements entre les lignes.")
    st.caption("• equal = parts égales • custom = poids personnalisés • single = tout sur une ligne")

    st.divider()
    _add_line_form("A_lines", "Portefeuille 1 — Client : ajouter une ligne")
    _paste_table_block("A_lines", "Portefeuille 1 — Client")

    st.divider()
    _add_line_form("B_lines", "Portefeuille 2 — Vous : ajouter une ligne")
    _paste_table_block("B_lines", "Portefeuille 2 — Vous")

# Poids perso / single target (par id)
custom_weights_A = {id(ln): 1.0 for ln in st.session_state["A_lines"]}
custom_weights_B = {id(ln): 1.0 for ln in st.session_state["B_lines"]}
single_target_A = id(st.session_state["A_lines"][0]) if st.session_state["A_lines"] else None
single_target_B = id(st.session_state["B_lines"][0]) if st.session_state["B_lines"] else None

# ---------------- Simulation -----------------
dfA, brutA, netA, valA, xirrA, startA, fullA = simulate_portfolio(
    st.session_state["A_lines"], st.session_state["M_A"], st.session_state["ONE_A"], st.session_state["ONE_A_DATE"],
    st.session_state["ALLOC_MODE"], custom_weights_A, single_target_A,
    st.session_state["EURO_RATE_PREVIEW"], st.session_state["FEE_A"]
)
dfB, brutB, netB, valB, xirrB, startB, fullB = simulate_portfolio(
    st.session_state["B_lines"], st.session_state["M_B"], st.session_state["ONE_B"], st.session_state["ONE_B_DATE"],
    st.session_state["ALLOC_MODE"], custom_weights_B, single_target_B,
    st.session_state["EURO_RATE_PREVIEW"], st.session_state["FEE_B"]
)

# Alignement graphique sur période commune
start_plot = min([x for x in [startA, startB] if pd.notnull(x)])
idx = pd.bdate_range(start=start_plot, end=TODAY, freq="B")
chart_df = pd.DataFrame(index=idx)
if not dfA.empty:
    chart_df["Client"] = dfA.reindex(idx)["Valeur"].ffill()
if not dfB.empty:
    chart_df["Vous"] = dfB.reindex(idx)["Valeur"].ffill()
chart_df = chart_df.reset_index().rename(columns={"index": "Date"})
chart_df = chart_df.melt("Date", var_name="variable", value_name="Valeur (€)")

st.subheader("📈 Évolution de la valeur des portefeuilles")
if chart_df.dropna().empty:
    st.info("Ajoutez des lignes et/ou vérifiez vos paramètres pour afficher le graphique.")
else:
    base = alt.Chart(chart_df).mark_line().encode(
        x=alt.X("Date:T", title="Date"),
        y=alt.Y("Valeur (€):Q", title="Valeur (€)"),
        color="variable:N"
    ).properties(height=360, use_container_width=True)
    st.altair_chart(base, use_container_width=True)

# ---------------- Synthèse -------------------
st.subheader("📊 Synthèse chiffrée")
c1, c2, c3, c4, c5, c6 = st.columns(6)
with c1: st.metric("Investi BRUT (Client)", to_eur(brutA))
with c2: st.metric("Net investi (Client)", to_eur(netA))
with c3: st.metric("XIRR (Client)", f"{xirrA:.2f}%" if xirrA is not None else "—")
with c4: st.metric("Investi BRUT (Vous)", to_eur(brutB))
with c5: st.metric("Net investi (Vous)", to_eur(netB))
with c6: st.metric("XIRR (Vous)", f"{xirrB:.2f}%" if xirrB is not None else "—")

st.markdown("### ✅ Et si c’était avec nous ?")
gain_vs_client = (valB - valA) if (valA and valB) else 0.0
delta_xirr = (xirrB - xirrA) if (xirrA is not None and xirrB is not None) else None
st.success(
    f"Vous auriez gagné **{to_eur(gain_vs_client)}** de plus."
    + (f" Soit **{delta_xirr:+.2f}%** de performance annualisée supplémentaire." if delta_xirr is not None else "")
)
st.markdown("- **Gain de valeur** vs portefeuille client : " + to_eur(gain_vs_client))
if delta_xirr is not None:
    st.markdown(f"- **Surperformance annualisée (Δ XIRR)** : {delta_xirr:+.2f}%")

# ---------------- Positions -------------------
positions_table("Portefeuille 1 — Client (positions)", "A_lines")
positions_table("Portefeuille 2 — Vous (positions)", "B_lines")

# ---------------- Aide rapide -----------------
with st.expander("🧩 Aide rapide"):
    st.markdown(
        """
- **Ajouter une ligne** côté gauche (ou collez un tableau `Nom;ISIN;Montant`).
- **Fonds en euros** : utilisez le symbole **EUROFUND** (taux paramétrable).
- **Frais d’entrée** : sliders indépendants pour Client et Vous.  
  Les frais s’appliquent **à chaque achat** (initial, mensuel, ponctuel).
- **Prix d’achat** : si non fourni, on prend la VL du jour d’achat (ou le 1er jour ouvré suivant).
- **Versements** : paramétrez des montants bruts et la règle d’affectation.
- **XIRR** : calculé à partir des flux bruts et de la valeur finale.
        """
    )
