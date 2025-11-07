# =========================================
# app.py — Comparateur Portefeuilles CGP
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

def fmt_date(d: pd.Timestamp | date | None) -> str:
    if d is None: return ""
    if isinstance(d, date) and not isinstance(d, pd.Timestamp):
        d = pd.Timestamp(d)
    return d.strftime("%d/%m/%Y")

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

# ---------- Fonds en euros (2% base, crédit annuel fin d’année) ----------
BASE_EURO_RATE = 0.02  # 2%/an

def is_euro_fund(isin: str) -> bool:
    return (isin or "").upper().startswith("FONDS_EURO_")

def euro_series_annual(rate_annual: float) -> pd.DataFrame:
    """
    Série quotidienne (D) base 100 avec crédit UNIQUEMENT au 31/12 de chaque année.
    Entre deux 31/12, la VL est plate (pas de capitalisation intra-annuelle).
    """
    idx = pd.date_range(start="2010-01-01", end=TODAY, freq="D")
    vals = []
    v = 100.0
    for d in idx:
        # crédit le dernier jour de l'année
        if d.is_year_end:
            v *= (1.0 + rate_annual)
        vals.append(v)
    return pd.DataFrame({"Close": vals}, index=idx)

# ---------- Univers pré-rempli ----------
UNIVERSE_GENERALI = [
    # Core
    {"name":"R-co Valor C EUR","isin":"FR0011253624","type":"Actions Monde"},
    {"name":"Vivalor International","isin":"FR0014001LS1","type":"Actions Monde"},
    {"name":"CARMIGNAC Investissement A EUR","isin":"FR0010148981","type":"Actions Monde"},
    {"name":"FIDELITY Funds - World Fund","isin":"LU0069449576","type":"Actions Monde"},
    {"name":"CLARTAN Valeurs C","isin":"LU1100076550","type":"Actions Europe"},
    {"name":"CARMIGNAC Patrimoine","isin":"FR0010135103","type":"Diversifié patrimonial"},
    {"name":"SYCOYIELD 2030 RC","isin":"FR001400MCQ6","type":"Obligataire échéance"},
    {"name":"R-Co Target 2029 HY","isin":"FR001400AWH8","type":"Obligataire HY"},
    # Fonds en euros simulés (nom générique; le taux effectif dépend des options choisies dans chaque portefeuille)
    {"name":"Generali Eurossima","isin":"FONDS_EURO_EUROSSIMA","type":"Fonds en euros"},
    {"name":"Generali Netissima","isin":"FONDS_EURO_NETISSIMA","type":"Fonds en euros"},
    {"name":"Apicil Euro Garanti","isin":"FONDS_EURO_APICIL","type":"Fonds en euros"},
]
UNI_OPTIONS = ["— Saisie libre —"] + [f"{r['name']} — {r['isin']}" for r in UNIVERSE_GENERALI]

# ---------- Etat ----------
for key in ["A_lines","B_lines"]:
    if key not in st.session_state: st.session_state[key]=[]

# ---------- helpers ----------
def _auto_name_from_isin(isin: str) -> str:
    if not isin: return ""
    if is_euro_fund(isin):
        for r in UNIVERSE_GENERALI:
            if r["isin"].upper()==isin.upper(): return r["name"]
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

def _month_end(d: pd.Timestamp) -> pd.Timestamp:
    last_day = calendar.monthrange(d.year, d.month)[1]
    return pd.Timestamp(year=d.year, month=d.month, day=last_day)

def _month_schedule(start_dt: pd.Timestamp, end_dt: pd.Timestamp) -> List[pd.Timestamp]:
    dates=[]; cur=pd.Timestamp(year=start_dt.year, month=start_dt.month, day=start_dt.day)
    cur = min(cur, _month_end(cur))
    while cur<=end_dt:
        dates.append(cur)
        y = cur.year + (cur.month//12)
        m = 1 if cur.month==12 else cur.month+1
        day = min(start_dt.day, calendar.monthrange(y,m)[1])
        cur = pd.Timestamp(year=y, month=m, day=day)
    return dates

# ---------- UI : construction lignes ----------
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
        try:
            amt = float(str(amount).replace(" ","").replace(",",".")); assert amt>0
        except Exception:
            st.warning("Entre un **montant investi (€)** valide (>0)."); st.stop()

        if not isin and not name:
            st.warning("Indique au minimum l’**ISIN** ou le **nom** du fonds."); st.stop()

        # --- Récup VL (euro-fund avec options appliquées plus tard dans le moteur) ou EODHD ici pour l'affichage achat ---
        if is_euro_fund(isin):
            dfp = euro_series_annual(BASE_EURO_RATE)  # base 2% pour déterminer un prix d’achat de référence
            sym_used, note = isin, "Fonds en euros (VL synthétique)."
        else:
            dfp, sym_used, note = load_price_series_any(isin or name, pd.Timestamp(dt))
        if dfp.empty:
            st.error("Impossible de récupérer des VL pour ce fonds."); st.stop()

        px = None
        if px_opt:
            try: px = float(str(px_opt).replace(",", "."))
            except: px = None
        if px is None:
            px = _get_close_on(dfp, pd.Timestamp(dt))
        if not px or px<=0:
            st.error("Prix d’achat non déterminable."); st.stop()

        qty = amt/px
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
            st.caption(f"le {fmt_date(line.get('buy_date'))}")
            st.caption(f"Quantité : {line.get('qty_calc'):.6f}")
        with c2:
            st.markdown(f"Prix achat\n\n**{to_eur(line.get('buy_px'))}**")
            st.caption(f"le {fmt_date(line.get('buy_date'))}")
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

# ---------- Paramètres de versements & Options fonds en euros ----------
def _alloc_sidebar(side_label: str, lines_key: str, prefix: str):
    st.subheader(side_label)
    m = st.number_input(f"Versement mensuel ({prefix})", min_value=0.0, value=0.0, step=100.0)
    one_amt = st.number_input(f"Versement complémentaire ({prefix})", min_value=0.0, value=0.0, step=100.0)
    one_date = st.date_input(f"Date versement complémentaire ({prefix})", value=date.today())

    # Options fonds en euros (appliquées à toutes les lignes fonds € de ce portefeuille)
    st.caption("⚙️ Options fonds en euros du portefeuille")
    euro_mode = st.selectbox(
        f"Option fonds en euros ({prefix})",
        ["Aucune", "Bonus annuel (+X%)", "Fonds euros performant (taux défini)"],
        index=0,
        key=f"{prefix}_euro_mode"
    )
    euro_bonus = 0.0
    euro_perf = BASE_EURO_RATE
    if euro_mode == "Bonus annuel (+X%)":
        euro_bonus = st.number_input(f"Bonus annuel (+X%) — {prefix}", min_value=0.0, max_value=5.0, value=0.0, step=0.1) / 100.0
    elif euro_mode == "Fonds euros performant (taux défini)":
        euro_perf = st.number_input(f"Taux annuel du fonds € performant — {prefix}", min_value=0.0, max_value=10.0, value=3.0, step=0.1) / 100.0

    mode = st.selectbox(f"Affectation des versements ({prefix})",
                        ["Pro-rata montants initiaux", "Répartition personnalisée", "Tout sur un seul fonds"])
    custom = {}
    single = None
    lines = st.session_state[lines_key]
    if mode == "Répartition personnalisée":
        if lines:
            st.caption("Répartir sur les lignes ci-dessous (total ≈ 100 %).")
            default = round(100.0/len(lines), 2)
            tot = 0.0
            for i, ln in enumerate(lines):
                w = st.slider(f"{ln['name']} ({ln['isin']})", 0.0, 100.0, default, 1.0, key=f"{prefix}_w{i}")
                custom[id(ln)] = w/100.0
                tot += w
            if abs(tot-100.0) > 1.0:
                st.warning("La somme des poids s’éloigne de 100 % — elle sera renormalisée automatiquement.")
        else:
            st.info("Ajoute au moins une ligne pour définir des poids personnalisés.")
    elif mode == "Tout sur un seul fonds":
        if lines:
            options = [f"{ln['name']} — {ln['isin']}" for ln in lines]
            pick = st.selectbox("Choisir la ligne cible", options, key=f"{prefix}_single_pick")
            idx = options.index(pick)
            single = id(lines[idx])
        else:
            st.info("Ajoute au moins une ligne pour choisir une cible unique.")

    euro_opts = {"mode": euro_mode, "bonus": euro_bonus, "perf": euro_perf}
    return m, one_amt, one_date, mode, custom, single, euro_opts

with st.sidebar:
    st.header("⚙️ Paramètres")
    mA, oneA_amt, oneA_date, modeA, customA, singleA, euroA = _alloc_sidebar("Portefeuille 1 — Client", "A_lines", "A")
    st.divider()
    mB, oneB_amt, oneB_date, modeB, customB, singleB, euroB = _alloc_sidebar("Portefeuille 2 — Vous", "B_lines", "B")

def _weights_for(lines: List[Dict[str,Any]], mode: str, custom: Dict[int,float], single_id: Optional[int]) -> Dict[int,float]:
    if not lines:
        return {}
    if mode == "Tout sur un seul fonds" and single_id is not None:
        return {id(ln): (1.0 if id(ln)==single_id else 0.0) for ln in lines}
    if mode == "Répartition personnalisée" and custom:
        s = sum(max(0.0, v) for v in custom.values())
        if s <= 0:
            return {id(ln): 1.0/len(lines) for ln in lines}
        return {k: max(0.0, v)/s for k,v in custom.items()}
    total = sum(float(ln["amount"]) for ln in lines)
    if total > 0:
        return {id(ln): float(ln["amount"])/total for ln in lines}
    return {id(ln): 1.0/len(lines) for ln in lines}

# ---------- CONSTRUCTION SÉRIES (avec options fonds € & versement ponctuel robuste) ----------
@st.cache_data(ttl=1800, show_spinner=False)
def build_portfolio_series(lines: List[Dict[str,Any]],
                           monthly_amt: float,
                           one_amt: float, one_date: date,
                           alloc_mode: str,
                           custom_weights: Dict[int,float],
                           single_target: Optional[int],
                           euro_opts: Dict[str, float | str]) -> Tuple[pd.DataFrame, float, float, Optional[float], pd.Timestamp, pd.Timestamp]:
    """
    Retourne (df_valeur, total_investi, valeur_finale, xirr_pct, start_min, start_full)
    - Quantités initiales ajoutées à la 1ère VL >= date d’achat
    - Versement ponctuel investi à la 1ère VL >= sa date (par ligne, CF XIRR alignés)
    - Fonds en euros à 2% base, fin d'année; options Bonus (+X%) ou Perf (taux défini)
    """
    if not lines:
        return pd.DataFrame(), 0.0, 0.0, None, TODAY, TODAY

    # Paramètres fonds € pour CE portefeuille
    euro_mode = euro_opts.get("mode", "Aucune")
    euro_bonus = float(euro_opts.get("bonus", 0.0) or 0.0)
    euro_perf  = float(euro_opts.get("perf", BASE_EURO_RATE) or BASE_EURO_RATE)

    def _effective_euro_rate():
        if euro_mode == "Bonus annuel (+X%)":
            return BASE_EURO_RATE + euro_bonus
        if euro_mode == "Fonds euros performant (taux défini)":
            return euro_perf
        return BASE_EURO_RATE

    # Séries de prix par ligne
    series: Dict[int, pd.Series] = {}
    for ln in lines:
        isin = ln.get("isin","")
        if is_euro_fund(isin):
            rate = _effective_euro_rate()
            df = euro_series_annual(rate)
            series[id(ln)] = df["Close"]
        else:
            df,_,_ = load_price_series_any(isin or ln.get("name"), None)
            if not df.empty:
                series[id(ln)] = df["Close"]
    if not series:
        return pd.DataFrame(), 0.0, 0.0, None, TODAY, TODAY

    # Dates effectives & quantités initiales
    eff_buy_date: Dict[int, pd.Timestamp] = {}
    qty_init: Dict[int, float] = {}
    for ln in lines:
        sid=id(ln); s=series.get(sid)
        if s is None or s.empty: continue
        d_buy = pd.Timestamp(ln["buy_date"])
        if d_buy in s.index:
            px_buy=float(s.loc[d_buy]); eff_dt=d_buy
        else:
            after=s.loc[s.index>=d_buy]
            if after.empty:
                px_buy=float(s.iloc[-1]); eff_dt=s.index[-1]
            else:
                px_buy=float(after.iloc[0]); eff_dt=after.index[0]
        px_manual = ln.get("buy_px", None)
        px_for_qty = float(px_manual) if (px_manual and px_manual>0) else px_buy
        qty_init[sid] = float(ln["amount"]) / float(px_for_qty)
        eff_buy_date[sid] = eff_dt

    start_min = min(eff_buy_date.values())
    start_full = max(eff_buy_date.values())

    idx = pd.Index(sorted(set().union(*[s.index for s in series.values()])))
    idx = idx[(idx>=start_min) & (idx<=TODAY)]
    if len(idx)==0:
        return pd.DataFrame(), 0.0, 0.0, None, start_min, start_full

    # Quantités courantes
    qty_curr: Dict[int, float] = {id(ln): 0.0 for ln in lines}
    weights = _weights_for(lines, alloc_mode, custom_weights, single_target)

    # Cash-flows XIRR
    cash_flows: List[Tuple[pd.Timestamp, float]] = []
    for ln in lines:
        sid=id(ln)
        if sid in eff_buy_date:
            cash_flows.append((eff_buy_date[sid], -float(ln["amount"])))

    # Plan du versement ponctuel: date effective par ligne
    oneshot_plan: Dict[int, Tuple[pd.Timestamp, float]] = {}
    if one_amt > 0:
        for ln in lines:
            sid=id(ln); s=series.get(sid)
            if s is None or s.empty: continue
            after = s.loc[s.index >= pd.Timestamp(one_date)]
            if after.empty:
                eff_dt = s.index[-1]; px = float(s.iloc[-1])
            else:
                eff_dt = after.index[0]; px = float(after.iloc[0])
            oneshot_plan[sid] = (eff_dt, px)
    oneshot_done: Dict[int, bool] = {id(ln): False for ln in lines}

    # Calendrier versements récurrents
    sched = _month_schedule(start_min, TODAY) if monthly_amt>0 else []

    values=[]
    for d in idx:
        # Ajout quantités initiales au jour effectif
        for ln in lines:
            sid=id(ln)
            if sid in eff_buy_date and d == eff_buy_date[sid]:
                qty_curr[sid] += qty_init[sid]

        # Versement ponctuel : investir le jour effectif propre à chaque ligne
        if one_amt > 0:
            for ln in lines:
                sid=id(ln); s=series.get(sid)
                if s is None or s.empty: continue
                if oneshot_done.get(sid, False): continue
                w = weights.get(sid, 0.0)
                if w <= 0: 
                    oneshot_done[sid] = True
                    continue
                eff_dt, px_line = oneshot_plan[sid]
                if d == eff_dt:
                    qty_curr[sid] += (one_amt*w)/px_line
                    cash_flows.append((eff_dt, -float(one_amt*w)))
                    oneshot_done[sid] = True

        # Versements mensuels
        if d in sched and monthly_amt>0:
            for ln in lines:
                sid=id(ln); s=series.get(sid)
                if s is None or s.empty: continue
                w=weights.get(sid,0.0)
                if w<=0: continue
                px = float(s.loc[d]) if d in s.index else float(s.loc[s.index>=d].iloc[0])
                qty_curr[sid] += (monthly_amt*w)/px
            cash_flows.append((d, -float(monthly_amt)))

        # Valorisation
        v=0.0
        for ln in lines:
            sid=id(ln); s=series.get(sid)
            if s is None or s.empty: continue
            if d in s.index: px=float(s.loc[d])
            else:
                before=s.loc[s.index<=d]
                if before.empty: continue
                px=float(before.iloc[-1])
            v += qty_curr[sid]*px
        values.append((d, v))

    df_val = pd.DataFrame(values, columns=["date","Valeur"]).set_index("date")
    total_invested = sum(float(ln["amount"]) for ln in lines) \
                     + len(sched)*monthly_amt \
                     + (one_amt if one_amt>0 else 0.0)
    final_val = float(df_val["Valeur"].iloc[-1])
    cash_flows.append((TODAY, final_val))
    irr = xirr(cash_flows)
    return df_val, total_invested, final_val, (irr*100.0 if irr is not None else None), start_min, start_full

# ---------- Contrôles d’affichage des courbes ----------
start_mode = st.radio(
    "Départ du graphique",
    ["Premier euro investi", "Quand le portefeuille est entièrement en place"],
    horizontal=True
)
rebase_100 = st.checkbox("Normaliser les courbes à 100 au départ", value=False)

# ---------- Action : Calcul & affichages ----------
st.divider()
run = st.button("🚀 Lancer la comparaison", type="primary")

if run:
    dfA, investA, valA, xirrA, A_first, A_full = build_portfolio_series(
        st.session_state["A_lines"], mA, oneA_amt, oneA_date, modeA, customA, singleA, euroA
    )
    dfB, investB, valB, xirrB, B_first, B_full = build_portfolio_series(
        st.session_state["B_lines"], mB, oneB_amt, oneB_date, modeB, customB, singleB, euroB
    )

    # Choix du point de départ par portefeuille
    A_start = A_first if start_mode == "Premier euro investi" else A_full
    B_start = B_first if start_mode == "Premier euro investi" else B_full

    dfA_plot = dfA.loc[dfA.index >= A_start].copy() if not dfA.empty else dfA
    dfB_plot = dfB.loc[dfB.index >= B_start].copy() if not dfB.empty else dfB

    # Option base 100
    if rebase_100:
        if not dfA_plot.empty:
            dfA_plot["Valeur"] = 100 * dfA_plot["Valeur"] / dfA_plot["Valeur"].iloc[0]
        if not dfB_plot.empty:
            dfB_plot["Valeur"] = 100 * dfB_plot["Valeur"] / dfB_plot["Valeur"].iloc[0]

    st.subheader("📈 Évolution de la valeur des portefeuilles")
    if not dfA_plot.empty or not dfB_plot.empty:
        df_plot = pd.DataFrame(index=sorted(set(dfA_plot.index).union(dfB_plot.index)))
        if not dfA_plot.empty: df_plot["Client"] = dfA_plot["Valeur"]
        if not dfB_plot.empty: df_plot["Vous"] = dfB_plot["Valeur"]
        y_label = "Indice (base 100)" if rebase_100 else "Valeur (€)"
        fig = px.line(df_plot, x=df_plot.index, y=df_plot.columns,
                      labels={"value": y_label, "index": "Date"},
                      title="Valeur quotidienne (avec versements selon l’affectation choisie)")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Ajoute des lignes pour au moins un portefeuille.")

    st.subheader("📊 Synthèse chiffrée")
    c1,c2,c3,c4,c5,c6 = st.columns(6)
    c1.metric("Investi (Client)", to_eur(investA))
    c2.metric("Valeur (Client)", to_eur(valA))
    c3.metric("XIRR (Client)", f"{xirrA:.2f}%" if xirrA is not None else "—")
    c4.metric("Investi (Vous)", to_eur(investB))
    c5.metric("Valeur (Vous)", to_eur(valB))
    c6.metric("XIRR (Vous)", f"{xirrB:.2f}%" if xirrB is not None else "—")

    # ===== Bloc clair "Et si c’était avec nous ?" =====
    st.subheader("✅ Et si c’était avec nous ?")
    if (valA is not None) and (valB is not None):
        delta_val = (valB or 0.0) - (valA or 0.0)
        msg = f"**Vous auriez gagné {to_eur(delta_val)} de plus.**"
        if (xirrA is not None) and (xirrB is not None):
            delta_xirr = (xirrB - xirrA)
            msg += f" **Soit +{delta_xirr:.2f}% de performance annualisée.**"
        st.success(msg)
        st.markdown(f"- **Gain de valeur** vs portefeuille client : **{to_eur(delta_val)}**")
        if xirrA is not None and xirrB is not None:
            st.markdown(f"- **Surperformance annualisée (Δ XIRR)** : **{(xirrB - xirrA):+.2f}%**")
        if abs((investB or 0.0) - (investA or 0.0)) > 1e-6:
            st.caption("Note : les investissements totaux peuvent différer (versements). Le Δ valeur compare les valorisations finales.")
    else:
        st.info("Ajoute des lignes et relance pour voir le delta.")

    # Détail positions actuelles
    def _detail_table(lines: List[Dict[str,Any]], title:str, euro_opts: Dict[str, float | str]):
        st.markdown(f"#### {title}")
        if not lines:
            st.info("Aucune ligne.")
            return
        # taux effectif pour affichage des fonds €
        mode = euro_opts.get("mode","Aucune")
        bonus = float(euro_opts.get("bonus",0.0) or 0.0)
        perf  = float(euro_opts.get("perf",BASE_EURO_RATE) or BASE_EURO_RATE)
        def _rate():
            if mode == "Bonus annuel (+X%)": return BASE_EURO_RATE + bonus
            if mode == "Fonds euros performant (taux défini)": return perf
            return BASE_EURO_RATE

        rows=[]
        for ln in lines:
            isin = ln.get("isin") or ""
            if is_euro_fund(isin):
                df = euro_series_annual(_rate())
            else:
                df,_,_ = load_price_series_any(isin or ln.get("name"), None)
            last = float(df["Close"].iloc[-1]) if not df.empty else np.nan
            rows.append({
                "Nom": ln.get("name","—"),
                "ISIN": ln.get("isin","—"),
                "Symbole": ln.get("sym_used","—"),
                "Montant investi €": float(ln.get("amount",0.0)),
                "Quantité": float(ln.get("qty_calc",0.0)),
                "Prix achat": float(ln.get("buy_px",np.nan)),
                "Dernier cours": last,
                "Date d’achat": fmt_date(ln.get("buy_date")),
            })
        dfv=pd.DataFrame(rows)
        st.dataframe(dfv.style.format({"Montant investi €":to_eur,"Quantité":"{:.6f}","Prix achat":to_eur,"Dernier cours":to_eur}),
                     use_container_width=True, hide_index=True)

    d1,d2 = st.columns(2)
    with d1: _detail_table(st.session_state["A_lines"], "Portefeuille 1 — Client (positions)", euroA)
    with d2: _detail_table(st.session_state["B_lines"], "Portefeuille 2 — Vous (positions)", euroB)
else:
    st.info("Renseigne tes lignes, paramètre les **options fonds en euros** (bonus/taux), les versements, puis clique **Lancer la comparaison**.")

# ---------- Debug (optionnel) ----------
with st.expander("🔧 Debug EODHD (optionnel)"):
    test_q = st.text_input("Tester une recherche (ISIN ou nom)")
    if test_q:
        st.write("Résultat /search :", eod_search(test_q))
        if is_euro_fund(test_q):
            st.caption("Fonds en euros — la VL affichée dépendra des options choisies dans le portefeuille.")
            df_dbg = euro_series_annual(BASE_EURO_RATE)
            sym_dbg = test_q; note_dbg = "Fonds en euros (2%/an, crédit fin d’année)."
        else:
            df_dbg, sym_dbg, note_dbg = load_price_series_any(test_q, None)
        st.write("Symbole testé :", sym_dbg)
        if note_dbg: st.caption(note_dbg)
        if not df_dbg.empty: st.dataframe(df_dbg.tail(5))
        else: st.warning("Aucune VL trouvée.")
