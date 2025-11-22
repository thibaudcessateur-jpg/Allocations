from __future__ import annotations

import json
from datetime import date
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
import streamlit as st
import altair as alt

# ---------------------------------------------------------------------
# Constantes & config
# ---------------------------------------------------------------------
TODAY = pd.Timestamp.today().normalize()
APP_TITLE = "Comparateur de portefeuilles"

st.set_page_config(page_title=APP_TITLE, layout="wide")
st.title(APP_TITLE)

# ---------------------------------------------------------------------
# Univers de fonds recommandés (core + défensifs + fonds en euros)
# ---------------------------------------------------------------------
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

# ---------------------------------------------------------------------
# Utilitaires format / XIRR
# ---------------------------------------------------------------------
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

# ---------------------------------------------------------------------
# API EODHD
# ---------------------------------------------------------------------
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
def get_price_series(isin_or_name: str, start: Optional[pd.Timestamp], euro_rate: float) -> Tuple[pd.DataFrame, str, str]:
    """
    Retourne (df Close, symbol_utilisé, debug).
    EUROFUND => série qui capitalise au taux annuel euro_rate.
    """
    debug = {"cands": []}
    val = str(isin_or_name).strip()
    if not val:
        return pd.DataFrame(), "", json.dumps(debug)

    # Fonds euros synthétique
    if val.upper() == "EUROFUND":
        idx = pd.bdate_range(start=pd.Timestamp("2000-01-03"), end=TODAY, freq="B")
        # Taux annuel -> taux quotidien approx (252 jours de bourse)
        r_ann = float(euro_rate) / 100.0
        if r_ann <= 0:
            prices = np.ones(len(idx))
        else:
            daily = (1.0 + r_ann) ** (1.0 / 252.0) - 1.0
            n = np.arange(len(idx))
            prices = (1.0 + daily) ** n
        df = pd.DataFrame({"Close": prices}, index=idx)
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

# ---------------------------------------------------------------------
# Suggestion de fonds alternatifs si date < 1ère VL
# ---------------------------------------------------------------------
def suggest_alternative_funds(buy_date: pd.Timestamp, euro_rate: float) -> List[Tuple[str, str, pd.Timestamp]]:
    """
    Propose des fonds recommandés (core + défensifs) dont la première VL
    est antérieure ou égale à la date d'achat donnée.
    Retourne une liste de tuples (nom, isin, date_inception).
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

# ---------------------------------------------------------------------
# Calendrier de versements & poids
# ---------------------------------------------------------------------
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

# ---------------------------------------------------------------------
# Calcul par ligne (avec frais)
# ---------------------------------------------------------------------
def compute_line_metrics(line: Dict[str, Any], fee_pct: float, euro_rate: float) -> Tuple[float, float, float]:
    amt_brut = float(line.get("amount_gross", 0.0))
    net_amt = amt_brut * (1.0 - fee_pct / 100.0)
    buy_ts = pd.Timestamp(line.get("buy_date"))
    px_manual = line.get("buy_px", None)

    if str(line.get("sym_used", "")).upper() == "EUROFUND" or str(line.get("isin", "")).upper() == "EUROFUND":
        # Prix à la date d'achat = niveau du fonds en euros à cette date
        dfp, _, _ = get_price_series("EUROFUND", None, euro_rate)
        px = _get_close_on(dfp, buy_ts)
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

# ---------------------------------------------------------------------
# Simulation d'un portefeuille avec protection "date < 1ère VL"
# ---------------------------------------------------------------------
def simulate_portfolio(
    lines: List[Dict[str, Any]],
    monthly_amt_gross: float,
    one_amt_gross: float, one_date: date,
    alloc_mode: str,
    custom_weights: Dict[int, float],
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

    # Prix et dates effectives d'achat
    for ln in lines:
        key_id = id(ln)
        df, sym, _ = get_price_series(ln.get("isin") or ln.get("name"), None, euro_rate)
        if df.empty:
            continue

        inception = df.index.min()
        d_buy = pd.Timestamp(ln.get("buy_date"))

        # Cas problématique : date d'achat antérieure à la 1ère VL
        if d_buy < inception:
            invalid_found = True
            ln["invalid_date"] = True
            ln["inception_date"] = inception

            alts = suggest_alternative_funds(d_buy, euro_rate)
            if alts:
                alt_lines = []
                for alt_name, alt_isin, alt_incep in alts:
                    alt_lines.append(
                        f"- {alt_name} ({alt_isin}), historique depuis le {fmt_date(alt_incep)}"
                    )
                alt_msg = "\n".join(alt_lines)
            else:
                alt_msg = "Aucun fonds recommandé ne dispose d'un historique suffisant pour cette date."

            date_warnings.append(
                f"[{portfolio_label}] {ln.get('name','(sans nom)')} "
                f"({ln.get('isin','—')}) :\n"
                f"- Date d'achat saisie : {fmt_date(d_buy)}\n"
                f"- 1ère VL disponible : {fmt_date(inception)}\n\n"
                f"Impossible de simuler ce portefeuille avec ce fonds pour cette période.\n"
                f"Propositions d'alternatives (fonds recommandés) pour l'analyse historique :\n{alt_msg}"
            )
            continue  # on ne met pas ce fonds dans la simulation

        # Cas normal
        ln["sym_used"] = sym

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

    # Si au moins une ligne a un problème de date -> on annule la simulation du portefeuille
    if invalid_found or not price_map:
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
        if brut <= 0:
            continue
        net = brut * (1.0 - fee_pct / 100.0)
        px = float(buy_price_used[key_id])
        dt = eff_buy_date[key_id]
        if px > 0:
            q = net / px
            tgt = dt if dt in qty_events.index else qty_events.index[qty_events.index >= dt][0]
            qty_events.loc[tgt, key_id] += q
            total_brut += brut
            total_net += net
            cash_flows.append((tgt, -brut))

    # Versement ponctuel
    weights = _weights_for(lines, alloc_mode, custom_weights, single_target)
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
                w = weights.get(key_id, 0.0)
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
                w = weights.get(key_id, 0.0)
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

# ---------------------------------------------------------------------
# Affichage des lignes
# ---------------------------------------------------------------------
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
            if line.get("sym_used"):
                st.caption(f"Symbole EODHD : `{line.get('sym_used')}`")
        with cols[1]:
            st.markdown(f"Investi (brut)\n\n**{to_eur(line.get('amount_gross', 0.0))}**")
            st.caption(f"Net après frais {fee_pct:.1f}% : **{to_eur(net_amt)}**")
            st.caption(f"Date d'achat : {fmt_date(line.get('buy_date'))}")
        with cols[2]:
            st.markdown(f"Prix d'achat\n\n**{to_eur(buy_px)}**")
            st.caption(f"Quantité : {qty_disp:.6f}")
            if line.get("note"):
                st.caption(line["note"])
        with cols[3]:
            try:
                dfl, _, _ = get_price_series(line.get("isin") or line.get("name"), None, euro_rate)
                last = float(dfl["Close"].iloc[-1]) if not dfl.empty else np.nan
                st.markdown(f"Dernier prix\n\n**{to_eur(last)}**")
            except Exception:
                st.markdown("Dernier prix\n\n—")
        with cols[4]:
            if not st.session_state[state_key]:
                if st.button("✏️", key=f"edit_{port_key}_{idx}", help="Modifier cette ligne"):
                    st.session_state[state_key] = True
                    st.experimental_rerun()
            if st.button("🗑️", key=f"del_{port_key}_{idx}", help="Supprimer cette ligne"):
                st.session_state[port_key].pop(idx)
                st.experimental_rerun()

        if line.get("invalid_date"):
            st.warning(
                f"Date d'achat antérieure à la 1ère VL du fonds "
                f"({fmt_date(line.get('inception_date'))}). "
                f"Cette ligne empêche la simulation du portefeuille."
            )

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
                    # on réinitialise le flag d'erreur si l'utilisateur corrige la date
                    line.pop("invalid_date", None)
                    line.pop("inception_date", None)
                    st.session_state[state_key] = False
                    st.success("Ligne mise à jour.")
                    st.experimental_rerun()

# ---------------------------------------------------------------------
# Tableau positions (simple)
# ---------------------------------------------------------------------
def positions_table(title: str, port_key: str):
    fee_pct = st.session_state.get("FEE_A", 0.0) if port_key == "A_lines" else st.session_state.get("FEE_B", 0.0)
    euro_rate = st.session_state.get("EURO_RATE_PREVIEW", 2.0)
    lines = st.session_state.get(port_key, [])
    rows = []
    for ln in lines:
        net_amt, buy_px, qty = compute_line_metrics(ln, fee_pct, euro_rate)
        rows.append({
            "Nom": ln.get("name", ""),
            "ISIN / Code": ln.get("isin", ""),
            "Symbole EODHD": ln.get("sym_used", ""),
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

# ---------------------------------------------------------------------
# UI : composition des portefeuilles (à droite)
# ---------------------------------------------------------------------
def _portfolio_composition_ui(port_key: str, label: str):
    st.markdown(f"### {label}")

    mode = st.radio(
        "Mode de saisie",
        ["Fonds recommandés", "Saisie libre"],
        key=f"mode_{port_key}",
        horizontal=True,
    )

    if mode == "Fonds recommandés":
        cat = st.selectbox(
            "Catégorie",
            ["Core (référence)", "Défensif"],
            key=f"reco_cat_{port_key}"
        )
        fonds_list = RECO_FUNDS_CORE if "Core" in cat else RECO_FUNDS_DEF
        options = [f"{nm} ({isin})" for nm, isin in fonds_list]
        choice = st.selectbox("Fonds recommandé", options, key=f"reco_choice_{port_key}")
        idx_choice = options.index(choice) if choice in options else 0
        name, isin = fonds_list[idx_choice]

        with st.form(key=f"form_reco_{port_key}", clear_on_submit=False):
            c1, c2, c3 = st.columns([2, 2, 2])
            with c1:
                amount = st.text_input("Montant investi (brut) €", value="")
            with c2:
                buy_date = st.date_input("Date d’achat", value=pd.Timestamp("2024-01-02").date())
            with c3:
                px = st.text_input("Prix d’achat (optionnel)", value="")
            note = st.text_input("Note (optionnel)", value="")
            ok = st.form_submit_button("➕ Ajouter ce fonds")
        if ok:
            try:
                amt = float(str(amount).replace(" ", "").replace(",", "."))
                assert amt > 0
            except Exception:
                st.warning("Montant invalide.")
            else:
                ln = {
                    "name": name,
                    "isin": isin,
                    "amount_gross": float(amt),
                    "buy_date": pd.Timestamp(buy_date),
                    "buy_px": float(str(px).replace(",", ".")) if px.strip() else "",
                    "note": note.strip(),
                    "sym_used": ""
                }
                st.session_state[port_key].append(ln)
                st.success("Ligne ajoutée (fonds recommandé).")

    else:  # Saisie libre
        with st.form(key=f"form_free_{port_key}", clear_on_submit=False):
            c1, c2 = st.columns([3, 2])
            with c1:
                name = st.text_input("Nom du fonds (laisser vide pour auto-détection)", value="", key=f"free_name_{port_key}")
            with c2:
                isin = st.text_input("ISIN ou code (ex: ISIN, mnémo EASYN, EUROFUND)", value="", key=f"free_isin_{port_key}")
            c3, c4, c5 = st.columns([2, 2, 2])
            with c3:
                amount = st.text_input("Montant investi (brut) €", value="", key=f"free_amt_{port_key}")
            with c4:
                buy_date = st.date_input("Date d’achat", value=pd.Timestamp("2024-01-02").date(), key=f"free_date_{port_key}")
            with c5:
                px = st.text_input("Prix d’achat (optionnel)", value="", key=f"free_px_{port_key}")
            note = st.text_input("Note (optionnel)", value="", key=f"free_note_{port_key}")
            add_btn = st.form_submit_button("➕ Ajouter cette ligne")

        if add_btn:
            try:
                amt = float(str(amount).replace(" ", "").replace(",", "."))
                assert amt > 0
            except Exception:
                st.warning("Montant invalide.")
            else:
                final_name = name.strip()
                code = isin.strip()
                # Si nom vide : on tente de récupérer le nom via EODHD
                if not final_name and code:
                    res = eodhd_search(code)
                    match = None
                    for it in res:
                        if it.get("ISIN") == code or it.get("Code") == code:
                            match = it
                            break
                    if match is None and res:
                        match = res[0]
                    if match:
                        final_name = match.get("Name", code)
                if not final_name:
                    final_name = code or "—"

                ln = {
                    "name": final_name,
                    "isin": code or final_name,
                    "amount_gross": float(amt),
                    "buy_date": pd.Timestamp(buy_date),
                    "buy_px": float(str(px).replace(",", ".")) if px.strip() else "",
                    "note": note.strip(),
                    "sym_used": ""
                }
                st.session_state[port_key].append(ln)
                st.success("Ligne ajoutée (saisie libre).")

    # Lignes actuelles
    lines = st.session_state.get(port_key, [])
    st.markdown("#### Lignes actuelles")
    if not lines:
        st.info("Aucune ligne pour le moment.")
    else:
        for idx, ln in enumerate(lines):
            _line_card(ln, idx, port_key)

# ---------------------------------------------------------------------
# Init session_state
# ---------------------------------------------------------------------
st.session_state.setdefault("A_lines", [])
st.session_state.setdefault("B_lines", [])
st.session_state.setdefault("EURO_RATE_PREVIEW", 2.0)
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

# ---------------------------------------------------------------------
# Sidebar : paramètres globaux
# ---------------------------------------------------------------------
with st.sidebar:
    st.header("🟦 Fonds en euros — Paramètre global")
    EURO_RATE = st.number_input(
        "Taux annuel du fonds en euros (%)",
        0.0, 10.0, st.session_state.get("EURO_RATE_PREVIEW", 2.0),
        0.10, key="EURO_RATE_PREVIEW"
    )
    st.caption("Utilisez le symbole **EUROFUND** comme support fonds en euros.")

    st.header("💶 Frais d’entrée (%)")
    FEE_A = st.number_input(
        "Frais d’entrée — Portefeuille 1 (Client)",
        0.0, 10.0, st.session_state.get("FEE_A", 3.0),
        0.10, key="FEE_A"
    )
    FEE_B = st.number_input(
        "Frais d’entrée — Portefeuille 2 (Valority)",
        0.0, 10.0, st.session_state.get("FEE_B", 2.0),
        0.10, key="FEE_B"
    )
    st.caption("Les frais s’appliquent sur chaque investissement (initial, mensuel, ponctuel).")

    st.header("📅 Paramètres de versement")
    with st.expander("Portefeuille 1 — Client"):
        M_A = st.number_input("Mensuel brut (€)", 0.0, 1_000_000.0, st.session_state.get("M_A", 0.0), 100.0, key="M_A")
        ONE_A = st.number_input("Ponctuel brut (€)", 0.0, 1_000_000.0, st.session_state.get("ONE_A", 0.0), 100.0, key="ONE_A")
        ONE_A_DATE = st.date_input("Date du ponctuel", value=st.session_state.get("ONE_A_DATE", pd.Timestamp("2024-07-01").date()), key="ONE_A_DATE")
    with st.expander("Portefeuille 2 — Valority"):
        M_B = st.number_input("Mensuel brut (€)", 0.0, 1_000_000.0, st.session_state.get("M_B", 0.0), 100.0, key="M_B")
        ONE_B = st.number_input("Ponctuel brut (€)", 0.0, 1_000_000.0, st.session_state.get("ONE_B", 0.0), 100.0, key="ONE_B")
        ONE_B_DATE = st.date_input("Date du ponctuel", value=st.session_state.get("ONE_B_DATE", pd.Timestamp("2024-07-01").date()), key="ONE_B_DATE")

    st.header("🧮 Règle d’affectation des versements")
    st.selectbox(
        "Mode",
        ["equal", "custom", "single"],
        index=["equal", "custom", "single"].index(st.session_state.get("ALLOC_MODE", "equal")),
        key="ALLOC_MODE",
        help="Répartition des versements entre les lignes."
    )

# ---------------------------------------------------------------------
# Layout principal : 2 colonnes (gauche résultat, droite composition)
# ---------------------------------------------------------------------
left_col, right_col = st.columns([2.0, 1.4])

# ------------------------- Colonne droite : composition ----------------
with right_col:
    st.subheader("🧩 Composition des portefeuilles")
    tabs = st.tabs(["Portefeuille Client", "Portefeuille Valority"])
    with tabs[0]:
        _portfolio_composition_ui("A_lines", "Portefeuille 1 — Client")
    with tabs[1]:
        _portfolio_composition_ui("B_lines", "Portefeuille 2 — Valority")

# ------------------------- Colonne gauche : résultat -------------------
with left_col:
    # Reset warnings avant simulation
    st.session_state["DATE_WARNINGS"] = []

    # Poids (pour l'instant = égal / 1.0)
    custom_weights_A = {id(ln): 1.0 for ln in st.session_state.get("A_lines", [])}
    custom_weights_B = {id(ln): 1.0 for ln in st.session_state.get("B_lines", [])}
    single_target_A = id(st.session_state["A_lines"][0]) if st.session_state["A_lines"] else None
    single_target_B = id(st.session_state["B_lines"][0]) if st.session_state["B_lines"] else None

    # Simulation
    dfA, brutA, netA, valA, xirrA, startA, fullA = simulate_portfolio(
        st.session_state.get("A_lines", []),
        st.session_state.get("M_A", 0.0),
        st.session_state.get("ONE_A", 0.0),
        st.session_state.get("ONE_A_DATE", pd.Timestamp("2024-07-01").date()),
        st.session_state.get("ALLOC_MODE", "equal"),
        custom_weights_A,
        single_target_A,
        st.session_state.get("EURO_RATE_PREVIEW", 2.0),
        st.session_state.get("FEE_A", 0.0),
        portfolio_label="Client"
    )

    dfB, brutB, netB, valB, xirrB, startB, fullB = simulate_portfolio(
        st.session_state.get("B_lines", []),
        st.session_state.get("M_B", 0.0),
        st.session_state.get("ONE_B", 0.0),
        st.session_state.get("ONE_B_DATE", pd.Timestamp("2024-07-01").date()),
        st.session_state.get("ALLOC_MODE", "equal"),
        custom_weights_B,
        single_target_B,
        st.session_state.get("EURO_RATE_PREVIEW", 2.0),
        st.session_state.get("FEE_B", 0.0),
        portfolio_label="Valority"
    )

    # Avertissements sur les dates / 1ère VL
    if st.session_state.get("DATE_WARNINGS"):
        with st.expander("⚠️ Problèmes d'historique / dates de 1ère VL"):
            for msg in st.session_state["DATE_WARNINGS"]:
                st.warning(msg)

    # Graphique
    st.subheader("📈 Évolution de la valeur des portefeuilles")
    dates_for_min = [d for d in [startA, startB] if isinstance(d, pd.Timestamp)]
    start_plot = min(dates_for_min) if dates_for_min else TODAY
    idx = pd.bdate_range(start=start_plot, end=TODAY, freq="B")
    chart_df = pd.DataFrame(index=idx)
    if not dfA.empty:
        chart_df["Client"] = dfA.reindex(idx)["Valeur"].ffill()
    if not dfB.empty:
        chart_df["Valority"] = dfB.reindex(idx)["Valeur"].ffill()
    chart_df = chart_df.reset_index().rename(columns={"index": "Date"})
    chart_df = chart_df.melt("Date", var_name="Portefeuille", value_name="Valeur (€)")

    if chart_df.dropna().empty:
        st.info("Ajoutez des lignes et/ou vérifiez vos paramètres pour afficher le graphique.")
    else:
        base = alt.Chart(chart_df).mark_line().encode(
            x=alt.X("Date:T", title="Date"),
            y=alt.Y("Valeur (€):Q", title="Valeur du portefeuille (€)"),
            color="Portefeuille:N"
        ).properties(height=360, width="container")
        st.altair_chart(base, use_container_width=True)

    # Synthèse chiffrée sous forme de "cartes"
    st.subheader("📊 Synthèse chiffrée")

    def _perf_metrics(brut: float, net: float, val: float, irr_pct: Optional[float]) -> Tuple[float, float]:
        if net <= 0:
            total = 0.0
        else:
            total = (val / net - 1.0) * 100.0
        return total, irr_pct if irr_pct is not None else None

    totalA, annA = _perf_metrics(brutA, netA, valA, xirrA)
    totalB, annB = _perf_metrics(brutB, netB, valB, xirrB)

    colA, colB = st.columns(2)
    with colA:
        st.markdown("#### Portefeuille Client")
        with st.container(border=True):
            st.metric("Investi BRUT", to_eur(brutA))
            st.metric("Net investi", to_eur(netA))
            st.metric("Valeur actuelle", to_eur(valA))
            st.metric("Rendement total", f"{totalA:.2f} %" if netA > 0 else "—")
            st.metric("Rendement annualisé (XIRR)", f"{annA:.2f} %" if annA is not None else "—")
    with colB:
        st.markdown("#### Portefeuille Valority")
        with st.container(border=True):
            st.metric("Investi BRUT", to_eur(brutB))
            st.metric("Net investi", to_eur(netB))
            st.metric("Valeur actuelle", to_eur(valB))
            st.metric("Rendement total", f"{totalB:.2f} %" if netB > 0 else "—")
            st.metric("Rendement annualisé (XIRR)", f"{annB:.2f} %" if annB is not None else "—")

    # Bloc comparaison
    st.markdown("### ✅ Et si c’était avec nous ?")
    gain_vs_client = (valB - valA) if (valA and valB) else 0.0
    delta_total = (totalB - totalA) if (netA > 0 and netB > 0) else None
    delta_xirr = (annB - annA) if (annA is not None and annB is not None) else None

    msg = f"Vous auriez **{to_eur(gain_vs_client)}** de plus."
    if delta_total is not None:
        msg += f" Soit **{delta_total:+.2f}%** de rendement total supplémentaire."
    if delta_xirr is not None:
        msg += f" Et **{delta_xirr:+.2f}%** en annualisé (XIRR)."

    st.success(msg)
    st.markdown("- Gain de valeur vs portefeuille client : " + to_eur(gain_vs_client))
    if delta_total is not None:
        st.markdown(f"- Surperformance en rendement total : {delta_total:+.2f}%")
    if delta_xirr is not None:
        st.markdown(f"- Surperformance annualisée (Δ XIRR) : {delta_xirr:+.2f}%")

    # Tables de positions
    positions_table("Portefeuille 1 — Client (positions)", "A_lines")
    positions_table("Portefeuille 2 — Valority (positions)", "B_lines")

    with st.expander("🧩 Aide rapide"):
        st.markdown(
            """
- **Composer les portefeuilles** dans l’onglet de droite (fonds recommandés ou saisie libre).
- Utilisez **EUROFUND** pour le fonds en euros (taux paramétrable dans la barre de gauche).
- Les **frais d’entrée** s’appliquent à chaque investissement.
- Les **versements mensuels & ponctuels** suivent la règle d’affectation choisie.
- Si un fonds n’existait pas à la date indiquée, la simulation est bloquée et des **alternatives** sont proposées.
- Le **rendement total** est calculé sur le net investi, le **XIRR** donne le rendement annualisé.
            """
        )
