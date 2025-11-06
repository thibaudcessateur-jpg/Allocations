# =========================================
# app.py — Comparateur Portefeuilles CGP
# - Saisie par MONTANT (€), quantité auto
# - Résolution EODHD "ISIN-first" avec fallbacks
# - Noms de fonds auto (même si ISIN seul)
# - Courbes d'évolution (valeur quotidienne)
# - Versements mensuels + complémentaire (répartis au prorata)
# - "Et si c'était avec nous ?" (delta valeur & delta XIRR)
# =========================================
import os, re, math, requests, calendar
from datetime import date, datetime
from typing import Optional, List, Dict, Any, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

st.set_page_config(page_title="Comparateur Portefeuilles CGP", page_icon="🦉", layout="wide")
TODAY = pd.Timestamp.today().normalize()

# ---------- Utils: secrets & formats ----------
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

# ---------- EODHD client ----------
EODHD_BASE = "https://eodhd.com/api"

def eodhd_get(path: str, params: Optional[Dict[str, Any]] = None) -> Any:
    url = f"{EODHD_BASE.rstrip('/')}{path}"
    p = {"api_token": Secret_Token("EODHD_API_KEY"), "fmt": "json"}
    if params: p.update(params)
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
    q = (q or "").strip()
    if not q: return None
    if _looks_like_isin(q):
        base = q.upper()
        for suf in [".EUFUND", ".FUND", ".USFUND"]:
            cand = f"{base}{suf}"
            if _eod_ok(cand): return cand
        res = eod_search(base)
        if res:
            preferred = ["EUFUND", "FUND", "USFUND"]
            exact = [it for it in res if str(it.get("ISIN","")).upper()==base]
            for exch in preferred:
                for it in exact:
                    ex = str(it.get("Exchange","")).upper()
                    if ex==exch:
                        cand = f"{base}.{ex}"
                        if _eod_ok(cand): return cand
                        code = str(it.get("Code","")).strip()
                        if code and _eod_ok(code): return code
            for it in exact:
                code = str(it.get("Code","")).strip()
                if code and _eod_ok(code): return code
        for suf in [".PA",".AS",".MI",".DE",".LSE"]:
            cand = f"{base}{suf}"
            if _eod_ok(cand): return cand
        return None
    # nom
    res = eod_search(q)
    if res:
        preferred = ["EUFUND","FUND","USFUND"]
        for exch in preferred:
            for it in res:
                ex = str(it.get("Exchange","")).upper()
                if ex==exch:
                    isin = str(it.get("ISIN","")).upper()
                    if _looks_like_isin(isin):
                        cand = f"{isin}.{ex}"
                        if _eod_ok(cand): return cand
        for it in res:
            code = str(it.get("Code","")).strip()
            if code and _eod_ok(code): return code
    return None

@st.cache_data(ttl=3*3600, show_spinner=False)
def eod_prices_any(symbol_or_isin: str, start_dt: Optional[pd.Timestamp] = None) -> Tuple[pd.DataFrame, str, str]:
    """df, symbol_used, note"""
    q = (symbol_or_isin or "").strip().upper()
    note = ""
    def _fetch(sym: str, from_dt: Optional[str]) -> pd.DataFrame:
        params={"period":"d"}; 
        if from_dt: params["from"]=from_dt
        js = eodhd_get(f"/eod/{sym}", params=params)
        df = pd.DataFrame(js)
        if df.empty: return pd.DataFrame()
        df["date"]=pd.to_datetime(df["date"]); df=df.set_index("date").sort_index()
        df["close"]=pd.to_numeric(df["close"], errors="coerce")
        return df[["close"]].rename(columns={"close":"Close"})
    def _try(sym: str, from_ts: Optional[pd.Timestamp]) -> Optional[pd.DataFrame]:
        df=_fetch(sym, None)
        if not df.empty:
            if from_ts is not None: df=df.loc[df.index>=from_ts]
            if not df.empty: return df
        f=from_ts.strftime("%Y-%m-%d") if from_ts is not None else None
        df=_fetch(sym, f)
        if not df.empty: return df
        return None
    if "." in q and not _looks_like_isin(q.split(".")[0]):
        df=_try(q,start_dt); 
        if df is not None: return df,q,note
    base=q.split(".")[0]
    if _looks_like_isin(base):
        sym=resolve_symbol(base)
        if sym:
            df=_try(sym,start_dt)
            if df is not None: return df,sym,note
        for suf in [".EUFUND",".FUND",".USFUND"]:
            df=_try(f"{base}{suf}", start_dt)
            if df is not None: return df,f"{base}{suf}",note
        for suf in [".PA",".AS",".MI",".DE",".LSE"]:
            df=_try(f'{base}{suf}', start_dt)
            if df is not None: return df,f"{base}{suf}",note
    # fallback: dernier cours via search
    srch=eod_search(base if _looks_like_isin(base) else q)
    last_px,last_dt=None,None
    if srch:
        choices=srch
        if _looks_like_isin(base):
            choices=[it for it in srch if str(it.get("ISIN","")).upper()==base] or srch
        it0=choices[0]; last_px=it0.get("previousClose"); last_dt=it0.get("previousCloseDate")
    if last_px is not None:
        last_px=float(last_px); last_dt=pd.to_datetime(last_dt).normalize() if last_dt else TODAY
        df=pd.DataFrame({"Close":[last_px]}, index=[last_dt])
        note="⚠️ Historique VL indisponible via /eod — utilisation du dernier cours (/search)."
        used=resolve_symbol(base) or base
        return df, used, note
    return pd.DataFrame(), q, "⚠️ Aucune VL récupérée."

@st.cache_data(ttl=3600, show_spinner=False)
def load_price_series_any(symbol_or_isin: str, from_dt: Optional[pd.Timestamp]) -> Tuple[pd.DataFrame, str, str]:
    return eod_prices_any(symbol_or_isin, from_dt)

# ---------- XIRR ----------
def xnpv(rate: float, cash_flows: List[Tuple[pd.Timestamp, float]]) -> float:
    t0=cash_flows[0][0]
    return sum(cf/((1+rate)**((t-t0).days/365.2425)) for t,cf in cash_flows)

def xirr(cash_flows: List[Tuple[pd.Timestamp, float]]) -> Optional[float]:
    if not cash_flows or len(cash_flows)<2: return None
    lo,hi=-0.9999,10.0
    for _ in range(100):
        mid=(lo+hi)/2; val=xnpv(mid,cash_flows)
        if abs(val)<1e-6: return mid
        if xnpv(lo,cash_flows)*val<0: hi=mid
        else: lo=mid
    return None

# ---------- Univers pré-rempli ----------
UNIVERSE_GENERALI = [
    {"name":"R-co Valor C EUR","isin":"FR0011253624","type":"Actions Monde"},
    {"name":"Vivalor International","isin":"FR0014001LS1","type":"Actions Monde"},
    {"name":"CARMIGNAC Investissement A EUR","isin":"FR0010148981","type":"Actions Monde"},
    {"name":"FIDELITY Funds - World Fund","isin":"LU0069449576","type":"Actions Monde"},
    {"name":"CLARTAN Valeurs C","isin":"LU1100076550","type":"Actions Europe"},
    {"name":"CARMIGNAC Patrimoine","isin":"FR0010135103","type":"Diversifié patrimonial"},
    {"name":"SYCOYIELD 2030 RC","isin":"FR001400MCQ6","type":"Obligataire échéance"},
    {"name":"R-Co Target 2029 HY","isin":"FR001400AWH8","type":"Obligataire HY"},
]
UNI_OPTIONS = ["— Saisie libre —"] + [f"{r['name']} — {r['isin']}" for r in UNIVERSE_GENERALI]

# ---------- Etat ----------
for key in ["A_lines","B_lines"]:
    if key not in st.session_state: st.session_state[key]=[]

# ---------- UI : paramètres de versements ----------
with st.sidebar:
    st.header("⚙️ Paramètres")
    st.subheader("Portefeuille 1 — Client")
    mA = st.number_input("Versement mensuel (A)", min_value=0.0, value=0.0, step=100.0, help="0€ = désactivé")
    oneA_amt = st.number_input("Versement complémentaire (A)", min_value=0.0, value=0.0, step=100.0)
    oneA_date = st.date_input("Date versement complémentaire (A)", value=date.today())
    st.divider()
    st.subheader("Portefeuille 2 — Vous")
    mB = st.number_input("Versement mensuel (B)", min_value=0.0, value=0.0, step=100.0, help="0€ = désactivé")
    oneB_amt = st.number_input("Versement complémentaire (B)", min_value=0.0, value=0.0, step=100.0)
    oneB_date = st.date_input("Date versement complémentaire (B)", value=date.today())

# ---------- helpers ----------
def _auto_name_from_isin(isin: str) -> str:
    if not isin: return ""
    res = eod_search(isin)
    for it in res:
        if str(it.get("ISIN","")).upper()==isin.upper():
            nm = str(it.get("Name","")).strip()
            if nm: return nm
    return ""

def _get_close_on(df: pd.DataFrame, dt: pd.Timestamp) -> Optional[float]:
    if df.empty: return None
    if dt in df.index:
        v=df.loc[dt,"Close"]; return float(v) if pd.notna(v) else None
    after=df.loc[df.index>=dt]
    if not after.empty:
        return float(after["Close"].iloc[0])
    return float(df["Close"].iloc[-1]) if not df.empty else None

def _add_line_ui(port_key: str, title: str):
    st.subheader(title)
    with st.form(key=f"{port_key}_form", clear_on_submit=False):
        c1,c2 = st.columns([2,1])
        with c1:
            sel = st.selectbox("Choisir un fonds (ou saisie libre) :", UNI_OPTIONS, key=f"{port_key}_select")
        with c2:
            amount = st.text_input("Montant investi (€)", value="", key=f"{port_key}_amount")

        c3,c4 = st.columns(2)
        with c3:
            dt = st.date_input("Date d’achat", value=date(2024,1,2), key=f"{port_key}_date")
        with c4:
            px_opt = st.text_input("Prix d’achat (optionnel, sinon VL)", value="", key=f"{port_key}_px")

        if sel!="— Saisie libre —":
            name, isin = sel.split(" — ")
            name=name.strip(); isin=isin.strip().upper()
        else:
            name = st.text_input("Nom du fonds (facultatif si ISIN saisi)", key=f"{port_key}_name").strip()
            isin = st.text_input("ISIN (recommandé)", key=f"{port_key}_isin").strip().upper()

        submitted = st.form_submit_button("➕ Ajouter", type="primary")

    if submitted:
        # montant
        try:
            amt = float(str(amount).replace(" ","").replace(",",".")); assert amt>0
        except Exception:
            st.warning("Entre un **montant investi (€)** valide (>0)."); st.stop()

        if not isin and not name:
            st.warning("Indique au minimum l’**ISIN** ou le **nom** du fonds."); st.stop()

        # série de VL, symbole, note
        dfp, sym_used, note = load_price_series_any(isin or name, pd.Timestamp(dt))
        if dfp.empty:
            st.error("Impossible de récupérer des VL pour ce fonds."); st.stop()

        # prix d'achat
        px = None
        if px_opt:
            try: px = float(str(px_opt).replace(",", "."))
            except: px = None
        if px is None:
            px = _get_close_on(dfp, pd.Timestamp(dt))
        if not px or px<=0:
            st.error("Prix d’achat non déterminable."); st.stop()

        qty = amt/px

        # auto-nom si besoin
        if not name and isin:
            name = _auto_name_from_isin(isin) or "—"

        st.session_state[port_key].append({
            "name": name or "—",
            "isin": isin or "",
            "amount": float(amt),
            "qty_calc": float(qty),
            "buy_date": pd.Timestamp(dt),
            "buy_px": float(px),
            "sym_used": sym_used,
            "note": note,
        })
        st.success(f"Ligne ajoutée ({name or isin})")

def _line_card(line: Dict[str,Any], idx:int, port_key:str):
    with st.container(border=True):
        c0,c1,c2,c3 = st.columns([3,2,2,2])
        with c0:
            st.markdown(f"**{line.get('name','—')}**")
            st.caption(f"ISIN : `{line.get('isin','—')}` • Symbole : ")
            st.code(line.get("sym_used","—"))
        with c1:
            st.markdown(f"Investi\n\n**{to_eur(line.get('amount',0.0))}**")
            st.caption(f"Quantité : {line.get('qty_calc'):.6f}")
        with c2:
            st.markdown(f"Prix achat\n\n**{to_eur(line.get('buy_px'))}**")
            if line.get("note"): st.caption(line["note"])
        with c3:
            if st.button("🗑️ Supprimer", key=f"del_{port_key}_{idx}"):
                st.session_state[port_key].pop(idx)
                st.experimental_rerun()

# ---------- UI deux portefeuilles ----------
st.title("🟣 Comparer deux portefeuilles (Client vs Vous)")

tabA, tabB = st.tabs(["📁 Portefeuille 1 — Client","🟣 Portefeuille 2 — Vous"])
with tabA:
    _add_line_ui("A_lines","Portefeuille 1 — Client")
    for i,ln in enumerate(st.session_state["A_lines"]): _line_card(ln,i,"A_lines")
with tabB:
    _add_line_ui("B_lines","Portefeuille 2 — Vous")
    for i,ln in enumerate(st.session_state["B_lines"]): _line_card(ln,i,"B_lines")

# ---------- Construction séries portefeuille + versements ----------
def _month_end(d: pd.Timestamp) -> pd.Timestamp:
    last_day = calendar.monthrange(d.year, d.month)[1]
    return pd.Timestamp(year=d.year, month=d.month, day=last_day)

def _month_schedule(start_dt: pd.Timestamp, end_dt: pd.Timestamp) -> List[pd.Timestamp]:
    dates=[]; cur=pd.Timestamp(year=start_dt.year, month=start_dt.month, day=start_dt.day)
    cur = min(cur, _month_end(cur))
    while cur<=end_dt:
        dates.append(cur)
        # next month same day (clamp)
        y = cur.year + (cur.month//12)
        m = 1 if cur.month==12 else cur.month+1
        day = min(start_dt.day, calendar.monthrange(y,m)[1])
        cur = pd.Timestamp(year=y, month=m, day=day)
    return dates

@st.cache_data(ttl=1800, show_spinner=False)
def build_portfolio_series(lines: List[Dict[str,Any]],
                           monthly_amt: float,
                           one_amt: float, one_date: date) -> Tuple[pd.DataFrame, float, float, Optional[float]]:
    """Retourne (df_valeur, total_investi, valeur_finale, xirr_pct)
       df_valeur : index dates, colonne 'Valeur'
       Achats mensuels répartis au prorata des montants initiaux.
    """
    if not lines:
        return pd.DataFrame(), 0.0, 0.0, None

    # collect price series and initial positions
    min_dt = min(pd.Timestamp(l["buy_date"]) for l in lines)
    max_dt = TODAY
    # load all series
    series = {}
    for ln in lines:
        df,_,_ = load_price_series_any(ln.get("isin") or ln.get("name"), None)
        if df.empty: continue
        series[id(ln)] = df["Close"]

    # common date index (business days union of all)
    idx = pd.Index(sorted(set().union(*[s.index for s in series.values()])))
    idx = idx[(idx>=min_dt) & (idx<=max_dt)]
    if len(idx)==0:
        return pd.DataFrame(), 0.0, 0.0, None

    # current quantities per line (start with initial)
    qty = {id(ln): float(ln["qty_calc"]) for ln in lines}
    invested_initial = sum(float(ln["amount"]) for ln in lines)

    # weights for contributions (pro-rata initial amounts; if zero -> égalitaire)
    weights = {id(ln): (float(ln["amount"])/invested_initial if invested_initial>0 else 1/len(lines)) for ln in lines}

    # cash flows for XIRR
    cash_flows = []
    for ln in lines:
        cash_flows.append((pd.Timestamp(ln["buy_date"]), -float(ln["amount"])))

    # schedule contributions
    if monthly_amt>0:
        sched = _month_schedule(min_dt, TODAY)
    else:
        sched = []
    one_dt = pd.Timestamp(one_date) if one_amt>0 else None

    values=[]
    for d in idx:
        # process contributions on this date
        if one_dt is not None and d==one_dt:
            for ln in lines:
                sid=id(ln); s=series[sid]
                # prix du jour (ou prochain dispo)
                if d in s.index: px=float(s.loc[d])
                else:
                    after=s.loc[s.index>=d]
                    if after.empty: continue
                    px=float(after.iloc[0])
                alloc = one_amt * weights[sid]
                qty[sid] += alloc/px
            cash_flows.append((d, -float(one_amt)))

        if d in sched:
            for ln in lines:
                sid=id(ln); s=series[sid]
                if d in s.index: px=float(s.loc[d])
                else:
                    after=s.loc[s.index>=d]
                    if after.empty: continue
                    px=float(after.iloc[0])
                alloc = monthly_amt * weights[sid]
                qty[sid] += alloc/px
            cash_flows.append((d, -float(monthly_amt)))

        # compute value today
        v=0.0
        for ln in lines:
            sid=id(ln)
            s=series[sid]
            if d in s.index:
                px=float(s.loc[d]); v += qty[sid]*px
            else:
                before=s.loc[s.index<=d]
                if before.empty: continue
                px=float(before.iloc[-1]); v += qty[sid]*px
        values.append((d, v))

    df_val = pd.DataFrame(values, columns=["date","Valeur"]).set_index("date")
    total_invested = invested_initial + (len(sched)*monthly_amt if monthly_amt>0 else 0.0) + (one_amt if one_amt>0 else 0.0)
    final_val = float(df_val["Valeur"].iloc[-1])
    cash_flows.append((TODAY, final_val))
    irr = xirr(cash_flows)
    return df_val, total_invested, final_val, (irr*100.0 if irr is not None else None)

# ---------- Action : Calcul & affichages ----------
st.divider()
run = st.button("🚀 Lancer la comparaison", type="primary")

if run:
    # Séries + KPIs
    dfA, investA, valA, xirrA = build_portfolio_series(st.session_state["A_lines"], mA, oneA_amt, oneA_date)
    dfB, investB, valB, xirrB = build_portfolio_series(st.session_state["B_lines"], mB, oneB_amt, oneB_date)

    st.subheader("📈 Évolution de la valeur des portefeuilles")
    if not dfA.empty or not dfB.empty:
        df_plot = pd.DataFrame(index=sorted(set(dfA.index).union(dfB.index)))
        if not dfA.empty: df_plot["Client"] = dfA["Valeur"]
        if not dfB.empty: df_plot["Vous"] = dfB["Valeur"]
        fig = px.line(df_plot, x=df_plot.index, y=df_plot.columns, labels={"value":"Valeur (€)","index":"Date"}, title="Valeur quotidienne")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Ajoute des lignes pour au moins un portefeuille.")

    # Synthèse
    st.subheader("📊 Synthèse chiffrée")
    c1,c2,c3,c4,c5,c6 = st.columns(6)
    c1.metric("Investi (Client)", to_eur(investA))
    c2.metric("Valeur (Client)", to_eur(valA))
    c3.metric("XIRR (Client)", f"{xirrA:.2f}%" if xirrA is not None else "—")
    c4.metric("Investi (Vous)", to_eur(investB))
    c5.metric("Valeur (Vous)", to_eur(valB))
    c6.metric("XIRR (Vous)", f"{xirrB:.2f}%" if xirrB is not None else "—")

    st.subheader("✅ Et si c’était avec nous ?")
    if valA and valB:
        delta_val = valB - valA
        delta_inv = investB - investA
        # on compare la valeur finale à investissement possiblement différent
        st.markdown(f"- **Gain de valeur** vs portefeuille client : **{to_eur(delta_val)}**")
        if xirrA is not None and xirrB is not None:
            st.markdown(f"- **Surperformance annualisée (Δ XIRR)** : **{(xirrB - xirrA):+.2f}%**")
        if abs(delta_inv) > 1e-6:
            st.caption("Note : les investissements totaux diffèrent (versements). Le Δ valeur compare les valorisations finales.")
    else:
        st.info("Ajoute des lignes et relance pour voir le delta.")

    # Détail des positions (état actuel simple)
    def _detail_table(lines: List[Dict[str,Any]], title:str):
        st.markdown(f"#### {title}")
        if not lines:
            st.info("Aucune ligne.")
            return
        rows=[]
        for ln in lines:
            df,_,_ = load_price_series_any(ln.get("isin") or ln.get("name"), None)
            last = float(df["Close"].iloc[-1]) if not df.empty else np.nan
            rows.append({
                "Nom": ln.get("name","—"),
                "ISIN": ln.get("isin","—"),
                "Symbole": ln.get("sym_used","—"),
                "Montant investi €": float(ln.get("amount",0.0)),
                "Quantité": float(ln.get("qty_calc",0.0)),
                "Prix achat": float(ln.get("buy_px",np.nan)),
                "Dernier cours": last,
            })
        dfv=pd.DataFrame(rows)
        st.dataframe(dfv.style.format({"Montant investi €":to_eur,"Quantité":"{:.6f}","Prix achat":to_eur,"Dernier cours":to_eur}),
                     use_container_width=True, hide_index=True)

    d1,d2 = st.columns(2)
    with d1: _detail_table(st.session_state["A_lines"], "Portefeuille 1 — Client (positions)")
    with d2: _detail_table(st.session_state["B_lines"], "Portefeuille 2 — Vous (positions)")
else:
    st.info("Renseigne tes lignes, tes versements (optionnels) puis clique **Lancer la comparaison**.")

# ---------- Debug (optionnel) ----------
with st.expander("🔧 Debug EODHD (optionnel)"):
    test_q = st.text_input("Tester une recherche (ISIN ou nom)")
    if test_q:
        st.write("Résultat /search :", eod_search(test_q))
        df_dbg, sym_dbg, note_dbg = load_price_series_any(test_q, None)
        st.write("Symbole testé :", sym_dbg)
        if note_dbg: st.caption(note_dbg)
        if not df_dbg.empty: st.dataframe(df_dbg.tail(5))
        else: st.warning("Aucune VL trouvée.")
