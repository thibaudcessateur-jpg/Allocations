# =========================================
# app.py — Comparateur Portefeuilles CGP
# - Fonds en euros (simulé) comme un support standard (EUROFUND)
# - Taux annuel paramétrable (sidebar), intérêts capitalisés le 31/12 (rebasé à 1 au départ)
# - Départ du graphe = "premier euro investi"
# - Axe Y ajusté autour du niveau d'investissement initial
# - Import portefeuille (CSV/Excel*), "Coller un tableau" pour saisie massive
# - Edition inline des lignes (montant/date/prix)
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

# ---------- Fonds en euros : série simulée ----------
def _eurofund_series(euro_rate: float,
                     start: pd.Timestamp = pd.Timestamp("1990-01-01"),
                     end: pd.Timestamp = TODAY) -> pd.Series:
    """NAV quotidienne constante, crédit d'intérêt le 31/12."""
    idx = pd.date_range(start=start, end=end, freq="D")
    vals = [1.0]
    for i in range(1, len(idx)):
        d = idx[i]
        v = vals[-1]
        if d.month == 12 and d.day == 31:
            v *= (1.0 + euro_rate/100.0)
        vals.append(v)
    return pd.Series(vals, index=idx, name="Close")

# ---------- Prix (fonds classiques via EODHD, fonds euros simulé) ----------
@st.cache_data(ttl=3*3600, show_spinner=False)
def eod_prices_any(symbol_or_isin: str,
                   start_dt: Optional[pd.Timestamp],
                   euro_rate: float) -> Tuple[pd.DataFrame, str, str]:
    """df, symbol_used, note. EUROFUND simulé (rebasé à 1 au départ réel)."""
    q = (symbol_or_isin or "").strip()
    if not q:
        return pd.DataFrame(), q, "⚠️ identifiant vide."

    # --- Cas Fonds en euros (simulé) ---
    if q.upper() in {"EUROFUND", "FONDS EN EUROS", "FONDS EN EUROS (SIMULÉ)"}:
        ser = _eurofund_series(euro_rate=euro_rate,
                               start=pd.Timestamp("1990-01-01"),
                               end=TODAY)
        if start_dt is not None:
            ser = ser.loc[ser.index >= start_dt]
        if not ser.empty:
            ser = ser / ser.iloc[0]  # rebase à 1 au jour d'achat (affichage intuitif)
        df = ser.to_frame()
        note = f"Fonds en euros simulé — intérêts le 31/12 au taux {euro_rate:.2f}%/an (rebasé à 1 au départ)."
        return df, "EUROFUND", note

    # --- Sinon : flux EODHD (fonds/ETF réels) ---
    qU = q.upper()
    note = ""
    def _fetch(sym: str, from_dt: Optional[str]) -> pd.DataFrame:
        params={"period":"d"}
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

    if "." in qU and not _looks_like_isin(qU.split(".")[0]):
        df=_try(qU,start_dt)
        if df is not None: return df,qU,note

    base=qU.split(".")[0]
    if _looks_like_isin(base):
        sym=resolve_symbol(base)
        if sym:
            df=_try(sym,start_dt)
            if df is not None: return df,sym,note
        for suf in [".EUFUND",".FUND",".USFUND"]:
            df=_try(f"{base}{suf}", start_dt)
            if df is not None: return df,f"{base}{suf}",note
        for suf in [".PA",".AS",".MI",".DE",".LSE"]:
            df=_try(f"{base}{suf}", start_dt)
            if df is not None: return df,f"{base}{suf}",note

    # fallback /search dernier cours
    srch=eod_search(base if _looks_like_isin(base) else qU)
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
    return pd.DataFrame(), qU, "⚠️ Aucune VL récupérée."

@st.cache_data(ttl=3600, show_spinner=False)
def load_price_series_any(symbol_or_isin: str, from_dt: Optional[pd.Timestamp], euro_rate: float) -> Tuple[pd.DataFrame, str, str]:
    return eod_prices_any(symbol_or_isin, from_dt, euro_rate)

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
    {"name":"Fonds en euros (simulé)","isin":"EUROFUND","type":"Fonds en euros"},
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

# ---------- Helpers dates & quantités ----------
def _auto_name_from_isin(isin: str) -> str:
    if not isin: return ""
    if isin.upper()=="EUROFUND": return "Fonds en euros (simulé)"
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

# ---------- Import/Export Portefeuille (Excel/CSV) ----------
def _normalize_col(s: str) -> str:
    s = (s or "").strip().lower()
    s = s.replace("€", "").replace("(", "").replace(")", "").replace(".", "").replace(",", "").replace("  ", " ")
    s = s.replace("montant investi", "amount").replace("montant", "amount").replace("investi", "amount")
    s = s.replace("nom du fonds", "name").replace("fonds", "name").replace("nom", "name")
    s = s.replace("isin code", "isin").replace("code isin", "isin")
    s = s.replace("prix dachat", "buy_price").replace("prix achat", "buy_price").replace("prix", "buy_price")
    s = s.replace("date dachat", "buy_date").replace("date achat", "buy_date").replace("date", "buy_date")
    return s

def _standardize_df(df_raw: pd.DataFrame) -> pd.DataFrame:
    colmap = {c: _normalize_col(str(c)) for c in df_raw.columns}
    df = df_raw.rename(columns=colmap)
    keep = [c for c in ["name","isin","amount","buy_price","buy_date"] if c in df.columns]
    df = df[keep].copy()
    if "amount" in df:
        df["amount"] = (
            df["amount"].astype(str).str.replace(" ", "").str.replace("€", "").str.replace(",", ".")
            .apply(lambda x: pd.to_numeric(x, errors="coerce"))
        )
    if "buy_price" in df:
        df["buy_price"] = (
            df["buy_price"].astype(str).str.replace(" ", "").str.replace("€", "").str.replace(",", ".")
            .apply(lambda x: pd.to_numeric(x, errors="coerce"))
        )
    if "buy_date" in df:
        df["buy_date"] = pd.to_datetime(df["buy_date"], errors="coerce", dayfirst=True).dt.date
    if "isin" in df:
        df["isin"] = df["isin"].astype(str).str.upper().str.strip()
    if "name" in df:
        df["name"] = df["name"].astype(str).str.strip()
    df = df.dropna(how="all")
    return df

def build_import_template_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"name":"R-co Valor C EUR","isin":"FR0011253624","amount":10000,"buy_price":"","buy_date":"2024-01-02"},
            {"name":"Fonds en euros (simulé)","isin":"EUROFUND","amount":10000,"buy_price":"","buy_date":"2024-01-02"},
        ],
        columns=["name","isin","amount","buy_price","buy_date"]
    )

def export_template_csv_bytes() -> bytes:
    return build_import_template_df().to_csv(index=False).encode("utf-8")

def lines_from_dataframe(df_std: pd.DataFrame, euro_rate: float, default_dt: Optional[pd.Timestamp]=None) -> List[Dict[str, Any]]:
    lines: List[Dict[str, Any]] = []
    for _, row in df_std.iterrows():
        isin = str(row.get("isin","") or "").strip().upper()
        name = str(row.get("name","") or "").strip()
        amount = row.get("amount", None)
        buy_price = row.get("buy_price", None)
        buy_date = row.get("buy_date", None)

        if (not isin and not name) or (amount is None or pd.isna(amount) or float(amount) <= 0):
            continue

        # date: si absente -> default_dt requis
        if not buy_date or pd.isna(buy_date):
            if default_dt is None: 
                continue
            buy_ts = pd.Timestamp(default_dt)
        else:
            buy_ts = pd.Timestamp(buy_date)

        dfp, sym_used, note = load_price_series_any(isin or name, buy_ts, euro_rate)
        if dfp.empty:
            continue

        px = None
        if (buy_price is not None) and not pd.isna(buy_price) and float(buy_price) > 0:
            px = float(buy_price)
        else:
            px = _get_close_on(dfp, buy_ts)
        if not px or px <= 0:
            continue

        if not name and isin:
            name = _auto_name_from_isin(isin) or (isin if isin else "—")

        qty = float(amount) / float(px)

        lines.append({
            "name": name or "—",
            "isin": isin or "",
            "amount": float(amount),
            "qty_calc": float(qty),
            "buy_date": buy_ts,
            "buy_px": float(px),
            "sym_used": sym_used,
            "note": note,
        })
    return lines

# ---------- Coller un tableau utilitaires ----------
def _detect_delimiter(s: str) -> Optional[str]:
    if "\t" in s: return "\t"
    counts = {";": s.count(";"), "|": s.count("|"), ",": s.count(",")}
    delim = max(counts, key=counts.get)
    if counts[delim] > 0: return delim
    return None  # fallback: split on 2+ spaces

def _normalize_header(h: str) -> str:
    h = h.strip().lower()
    h = h.replace("€","").replace("(", "").replace(")", "")
    h = h.replace("  "," ")
    mapping = {
        "nom": "name", "nom du fonds":"name", "fonds":"name", "name":"name",
        "isin":"isin", "code isin":"isin", "isin code":"isin",
        "montant":"amount", "montant investi":"amount", "amount":"amount", "investi":"amount",
        "date":"buy_date", "date d'achat":"buy_date", "date achat":"buy_date", "buy date":"buy_date",
        "prix":"buy_price", "prix d'achat":"buy_price", "buy price":"buy_price"
    }
    return mapping.get(h, h)

def parse_pasted_table(text: str) -> pd.DataFrame:
    lines = [ln for ln in text.strip().splitlines() if ln.strip()]
    if not lines: return pd.DataFrame()
    delim = _detect_delimiter(text)
    rows = []
    for ln in lines:
        if delim:
            parts = [p.strip() for p in ln.split(delim)]
        else:
            parts = re.split(r"\s{2,}", ln.strip())
        rows.append(parts)
    header = rows[0]
    normalized = [_normalize_header(h) for h in header]
    known = set(normalized) & {"name","isin","amount","buy_date","buy_price"}
    if not known:
        cols = ["name","isin","amount"]
        if len(header)>=4: cols.append("buy_date")
        if len(header)>=5: cols.append("buy_price")
        data = rows
        df = pd.DataFrame(data, columns=cols[:len(data[0])])
    else:
        data = rows[1:]
        df = pd.DataFrame(data, columns=normalized[:len(rows[0])])

    # nettoyage basique
    for col in df.columns:
        df[col] = df[col].astype(str).str.strip()
        if col == "amount":
            df[col] = df[col].str.replace(" ", "").str.replace("€","").str.replace(",", ".")
        if col == "buy_price":
            df[col] = df[col].str.replace(" ", "").str.replace("€","").str.replace(",", ".")
    return df

# ---------- Saisie d'une ligne + édition ----------
def _add_line_ui(port_key: str, title: str, euro_rate: float):
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

        dfp, sym_used, note = load_price_series_any(isin or name, pd.Timestamp(dt), euro_rate)
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

        qty = float(amt)/float(px)
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
    state_key = f"edit_mode_{port_key}_{idx}"
    if state_key not in st.session_state:
        st.session_state[state_key] = False

    with st.container(border=True):
        header = st.columns([3,2,2,2,1])
        with header[0]:
            st.markdown(f"**{line.get('name','—')}**")
            st.caption(f"ISIN : `{line.get('isin','—')}` • Symbole : ")
            st.code(line.get('sym_used','—'))
        with header[1]:
            st.markdown(f"Investi\n\n**{to_eur(line.get('amount',0.0))}**")
            st.caption(f"le {fmt_date(line.get('buy_date'))}")
            st.caption(f"Quantité : {line.get('qty_calc'):.6f}")
        with header[2]:
            st.markdown(f"Prix achat\n\n**{to_eur(line.get('buy_px'))}**")
            st.caption(f"le {fmt_date(line.get('buy_date'))}")
            if line.get("note"): st.caption(line["note"])
        with header[3]:
            try:
                df_last,_,_ = load_price_series_any(line.get("isin") or line.get("name"), None, st.session_state.get("EURO_RATE_PREVIEW", 2.0))
                last = float(df_last["Close"].iloc[-1]) if not df_last.empty else np.nan
                st.markdown(f"Dernier : **{to_eur(last)}**")
            except Exception:
                pass
        with header[4]:
            if not st.session_state[state_key]:
                if st.button("✏️", key=f"edit_{port_key}_{idx}", help="Modifier"):
                    st.session_state[state_key] = True
                    st.experimental_rerun()
            if st.button("🗑️", key=f"del_{port_key}_{idx}", help="Supprimer"):
                st.session_state[port_key].pop(idx)
                st.experimental_rerun()

        # Edition inline
        if st.session_state[state_key]:
            with st.form(key=f"form_edit_{port_key}_{idx}", clear_on_submit=False):
                c1,c2,c3,c4 = st.columns([2,2,2,1])
                with c1:
                    new_amount = st.text_input("Montant investi (€)", value=str(line.get("amount","")))
                with c2:
                    new_date = st.date_input("Date d’achat", value=pd.Timestamp(line.get("buy_date")).date())
                with c3:
                    new_px = st.text_input("Prix d’achat (optionnel)", value=str(line.get("buy_px","")))
                with c4:
                    st.caption(" ")
                    submitted = st.form_submit_button("💾 Enregistrer")

                if submitted:
                    try:
                        amt = float(str(new_amount).replace(" ","").replace(",","."))
                        assert amt>0
                    except Exception:
                        st.warning("Montant invalide."); st.stop()

                    buy_ts = pd.Timestamp(new_date)
                    dfp,_,_ = load_price_series_any(line.get("isin") or line.get("name"), buy_ts, st.session_state.get("EURO_RATE_PREVIEW", 2.0))
                    if dfp.empty:
                        st.error("Impossible de recalculer la VL au jour choisi."); st.stop()

                    if new_px.strip():
                        try:
                            px = float(str(new_px).replace(",","."))
                        except Exception:
                            px = _get_close_on(dfp, buy_ts)
                    else:
                        px = _get_close_on(dfp, buy_ts)

                    if not px or px<=0:
                        st.error("Prix d’achat non déterminable."); st.stop()

                    qty = float(amt)/float(px)

                    line["amount"] = float(amt)
                    line["buy_date"] = buy_ts
                    line["buy_px"] = float(px)
                    line["qty_calc"] = float(qty)

                    st.session_state[state_key] = False
                    st.success("Ligne mise à jour.")
                    st.experimental_rerun()

# ---------- UI deux portefeuilles ----------
st.title("🟣 Comparer deux portefeuilles (Client vs Vous)")

with st.sidebar:
    st.header("💶 Fonds en euros — Paramètre global")
    EURO_RATE = st.number_input("Taux annuel du fonds en euros (%)",
                                min_value=0.0, max_value=10.0, value=2.0, step=0.1,
                                help="Intérêts capitalisés le 31/12 (série rebasée à 1 au départ).")
    st.session_state["EURO_RATE_PREVIEW"] = EURO_RATE

tabA, tabB = st.tabs(["📁 Portefeuille 1 — Client","🟣 Portefeuille 2 — Vous"])

# ---------------- Portefeuille A ----------------
with tabA:
    _add_line_ui("A_lines","Portefeuille 1 — Client", EURO_RATE)

    # ---- Coller un tableau ----
    with st.expander("📋 Coller un tableau (Nom | ISIN | Montant | [Date] | [Prix])"):
        default_dt_A = st.date_input("Date d’achat par défaut (si absente dans le tableau)", value=date(2024,1,2), key="default_dt_A")
        pasteA = st.text_area("Colle ici depuis Excel/Sheets", height=180, key="pasteA")
        if st.button("🔎 Prévisualiser (Client)"):
            if not pasteA.strip():
                st.warning("Rien à parser.")
            else:
                dfp = parse_pasted_table(pasteA)
                if dfp.empty:
                    st.warning("Impossible de détecter un tableau.")
                else:
                    st.write("Aperçu :", dfp)
                    st.session_state["pasteA_preview"] = dfp
                    st.session_state["pasteA_default_dt"] = pd.Timestamp(default_dt_A)

        if st.button("➕ Ajouter ces lignes (Client)"):
            dfp = st.session_state.get("pasteA_preview", pd.DataFrame())
            default_dt_saved = st.session_state.get("pasteA_default_dt", None)
            if dfp.empty:
                st.warning("Fais d’abord la prévisualisation.")
            else:
                # transformer types pour robustesse
                dfp = dfp.replace("", np.nan)
                # essayer de caster amount/prix
                for col in ["amount","buy_price"]:
                    if col in dfp.columns:
                        dfp[col] = pd.to_numeric(dfp[col], errors="coerce")
                # dates : laisser lines_from_dataframe gérer via default_dt
                new_lines = lines_from_dataframe(dfp, EURO_RATE, default_dt=default_dt_saved)
                if not new_lines:
                    st.warning("Aucune ligne valide à ajouter (vérifie ISIN/nom, montant, date par défaut).")
                else:
                    st.session_state["A_lines"].extend(new_lines)
                    st.success(f"{len(new_lines)} ligne(s) ajoutée(s).")
                    st.experimental_rerun()

    # ---- Import portefeuille (Excel/CSV) ----
    with st.expander("📥 Importer un portefeuille client (Excel/CSV)"):
        cta1, cta2 = st.columns(2)
        with cta1:
            st.download_button(
                "⬇️ Télécharger le template CSV",
                data=export_template_csv_bytes(),
                file_name="template_portefeuille_client.csv",
                mime="text/csv"
            )
        with cta2:
            st.caption("Colonnes : **name**, **isin**, **amount**, **buy_price** (opt.), **buy_date** (YYYY-MM-DD).")
        default_dt_file_A = st.date_input("Date d’achat par défaut pour l’import (si absente)", value=date(2024,1,2), key="default_dt_file_A")

        up = st.file_uploader("Choisir un fichier .xlsx/.xls/.csv", type=["xlsx","xls","csv"], key="uploader_A")
        replace_mode = st.radio("Mode d’import", ["Remplacer le portefeuille client", "Ajouter aux lignes existantes"], horizontal=True, key="import_mode_A")
        if up is not None:
            try:
                if up.name.lower().endswith(".csv"):
                    df_raw = pd.read_csv(up)
                else:
                    try:
                        import openpyxl  # noqa
                        df_raw = pd.read_excel(up)
                    except Exception:
                        st.error("Lecture Excel indisponible (openpyxl manquant). Utilise plutôt un **CSV** ou **Coller un tableau**.")
                        df_raw = None
                if df_raw is not None:
                    df_std = _standardize_df(df_raw)
                    st.write("Aperçu détecté :", df_std.head())
                    if st.button("Importer ces lignes", type="primary", key="btn_import_A"):
                        new_lines = lines_from_dataframe(df_std, EURO_RATE, default_dt=pd.Timestamp(default_dt_file_A))
                        if not new_lines:
                            st.warning("Aucune ligne valide détectée dans le fichier.")
                        else:
                            if replace_mode.startswith("Remplacer"):
                                st.session_state["A_lines"] = new_lines
                            else:
                                st.session_state["A_lines"].extend(new_lines)
                            st.success(f"{len(new_lines)} ligne(s) importée(s).")
                            st.experimental_rerun()
            except Exception as e:
                st.error(f"Impossible de lire le fichier : {e}. Astuce : utilise CSV ou 'Coller un tableau'.")

    for i,ln in enumerate(st.session_state["A_lines"]): _line_card(ln,i,"A_lines")

# ---------------- Portefeuille B ----------------
with tabB:
    _add_line_ui("B_lines","Portefeuille 2 — Vous", EURO_RATE)

    # ---- Coller un tableau ----
    with st.expander("📋 Coller un tableau (Nom | ISIN | Montant | [Date] | [Prix])"):
        default_dt_B = st.date_input("Date d’achat par défaut (si absente dans le tableau)", value=date(2024,1,2), key="default_dt_B")
        pasteB = st.text_area("Colle ici depuis Excel/Sheets", height=180, key="pasteB")
        if st.button("🔎 Prévisualiser (Vous)"):
            if not pasteB.strip():
                st.warning("Rien à parser.")
            else:
                dfp = parse_pasted_table(pasteB)
                if dfp.empty:
                    st.warning("Impossible de détecter un tableau.")
                else:
                    st.write("Aperçu :", dfp)
                    st.session_state["pasteB_preview"] = dfp
                    st.session_state["pasteB_default_dt"] = pd.Timestamp(default_dt_B)

        if st.button("➕ Ajouter ces lignes (Vous)"):
            dfp = st.session_state.get("pasteB_preview", pd.DataFrame())
            default_dt_saved = st.session_state.get("pasteB_default_dt", None)
            if dfp.empty:
                st.warning("Fais d’abord la prévisualisation.")
            else:
                dfp = dfp.replace("", np.nan)
                for col in ["amount","buy_price"]:
                    if col in dfp.columns:
                        dfp[col] = pd.to_numeric(dfp[col], errors="coerce")
                new_lines = lines_from_dataframe(dfp, EURO_RATE, default_dt=default_dt_saved)
                if not new_lines:
                    st.warning("Aucune ligne valide à ajouter (vérifie ISIN/nom, montant, date par défaut).")
                else:
                    st.session_state["B_lines"].extend(new_lines)
                    st.success(f"{len(new_lines)} ligne(s) ajoutée(s).")
                    st.experimental_rerun()

    for i,ln in enumerate(st.session_state["B_lines"]): _line_card(ln,i,"B_lines")

# ---------- Paramètres de versements & AFFECTATION ----------
def _alloc_sidebar(side_label: str, lines_key: str, prefix: str):
    st.subheader(side_label)
    m = st.number_input(f"Versement mensuel ({prefix})", min_value=0.0, value=0.0, step=100.0)
    one_amt = st.number_input(f"Versement complémentaire ({prefix})", min_value=0.0, value=0.0, step=100.0)
    one_date = st.date_input(f"Date versement complémentaire ({prefix})", value=date.today())
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
    return m, one_amt, one_date, mode, custom, single

with st.sidebar:
    st.header("⚙️ Paramètres de versement")
    mA, oneA_amt, oneA_date, modeA, customA, singleA = _alloc_sidebar("Portefeuille 1 — Client", "A_lines", "A")
    st.divider()
    mB, oneB_amt, oneB_date, modeB, customB, singleB = _alloc_sidebar("Portefeuille 2 — Vous", "B_lines", "B")

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

# ---------- CONSTRUCTION SÉRIES ----------
@st.cache_data(ttl=1800, show_spinner=False)
def build_portfolio_series(lines: List[Dict[str,Any]],
                           monthly_amt: float,
                           one_amt: float, one_date: date,
                           alloc_mode: str,
                           custom_weights: Dict[int,float],
                           single_target: Optional[int],
                           euro_rate: float
                           ) -> Tuple[pd.DataFrame, float, float, Optional[float], pd.Timestamp, pd.Timestamp]:
    """
    Retourne (df_valeur, total_investi, valeur_finale, xirr_pct, start_min, start_full)
    Quantités initiales ajoutées le JOUR EFFECTIF (1ère VL >= date d’achat).
    start_min = date du premier euro investi
    start_full = date à laquelle toutes les lignes initiales sont en place
    """
    if not lines:
        return pd.DataFrame(), 0.0, 0.0, None, TODAY, TODAY

    # Séries de prix pour chaque ligne (EUROFUND inclus)
    series: Dict[int, pd.Series] = {}
    for ln in lines:
        df,_,_ = load_price_series_any(ln.get("isin") or ln.get("name"), None, euro_rate)
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

    qty_curr: Dict[int, float] = {id(ln): 0.0 for ln in lines}
    weights = _weights_for(lines, alloc_mode, custom_weights, single_target)

    # Cash-flows XIRR aux dates EFFECTIVES
    cash_flows: List[Tuple[pd.Timestamp, float]] = []
    for ln in lines:
        sid=id(ln)
        if sid in eff_buy_date:
            cash_flows.append((eff_buy_date[sid], -float(ln["amount"])))

    # Calendrier versements
    sched = _month_schedule(start_min, TODAY) if monthly_amt>0 else []
    one_dt = pd.Timestamp(one_date) if one_amt>0 else None

    values=[]
    for d in idx:
        # Ajout des quantités initiales le jour effectif
        for ln in lines:
            sid=id(ln)
            if sid in eff_buy_date and d == eff_buy_date[sid]:
                qty_curr[sid] += qty_init[sid]

        # Versement ponctuel
        if one_dt is not None and d==one_dt and one_amt>0:
            for ln in lines:
                sid=id(ln); s=series.get(sid)
                if s is None or s.empty: continue
                w=weights.get(sid,0.0)
                if w<=0: continue
                px = float(s.loc[d]) if d in s.index else float(s.loc[s.index>=d].iloc[0])
                qty_curr[sid] += (one_amt*w)/px
            cash_flows.append((d, -float(one_amt)))

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
    total_invested = sum(float(ln["amount"]) for ln in lines) + len(sched)*monthly_amt + (one_amt if one_amt>0 else 0.0)
    final_val = float(df_val["Valeur"].iloc[-1])
    cash_flows.append((TODAY, final_val))
    irr = xirr(cash_flows)
    return df_val, total_invested, final_val, (irr*100.0 if irr is not None else None), start_min, start_full

# ---------- Contrôles d’affichage ----------
rebase_100 = st.checkbox("Normaliser les courbes à 100 au départ", value=False)

# ---------- Action : Calcul & affichages ----------
st.divider()
run = st.button("🚀 Lancer la comparaison", type="primary")

if run:
    dfA, investA, valA, xirrA, A_first, A_full = build_portfolio_series(
        st.session_state["A_lines"], 
        st.session_state.get("mA", 0.0), 
        st.session_state.get("oneA_amt", 0.0), 
        st.session_state.get("oneA_date", date.today()), 
        st.session_state.get("modeA", "Pro-rata montants initiaux"), 
        st.session_state.get("customA", {}), 
        st.session_state.get("singleA", None), 
        st.session_state["EURO_RATE_PREVIEW"]
    )
    dfB, investB, valB, xirrB, B_first, B_full = build_portfolio_series(
        st.session_state["B_lines"], 
        st.session_state.get("mB", 0.0), 
        st.session_state.get("oneB_amt", 0.0), 
        st.session_state.get("oneB_date", date.today()), 
        st.session_state.get("modeB", "Pro-rata montants initiaux"), 
        st.session_state.get("customB", {}), 
        st.session_state.get("singleB", None), 
        st.session_state["EURO_RATE_PREVIEW"]
    )

    # Départ = premier euro investi
    A_start, B_start = A_first, B_first

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

        # Axe Y calé autour du niveau d'investissement initial (±5%)
        starts = []
        if not dfA_plot.empty: starts.append(float(dfA_plot["Valeur"].iloc[0]))
        if not dfB_plot.empty: starts.append(float(dfB_plot["Valeur"].iloc[0]))
        base_min = min(starts) if starts else (min(df_plot.min()) if not df_plot.empty else 0.0)
        y_max = max(df_plot.max()) if not df_plot.empty else 1.0
        y_margin = (y_max - base_min) * 0.05
        y_min_adj = max(0.0, base_min - y_margin)

        fig = px.line(
            df_plot, x=df_plot.index, y=df_plot.columns,
            labels={"value": y_label, "index": "Date"},
            title="Valeur quotidienne (avec versements selon l’affectation choisie)",
        )
        fig.update_yaxes(range=[y_min_adj, y_max + y_margin])
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Ajoute des lignes pour au moins un portefeuille.")

    st.subheader("📊 Synthèse chifrée")
    c1,c2,c3,c4,c5,c6 = st.columns(6)
    c1.metric("Investi (Client)", to_eur(investA))
    c2.metric("Valeur (Client)", to_eur(valA))
    c3.metric("XIRR (Client)", f"{xirrA:.2f}%" if xirrA is not None else "—")
    c4.metric("Investi (Vous)", to_eur(investB))
    c5.metric("Valeur (Vous)", to_eur(valB))
    c6.metric("XIRR (Vous)", f"{xirrB:.2f}%" if xirrB is not None else "—")

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
    else:
        st.info("Ajoute des lignes et relance pour voir le delta.")

    def _detail_table(lines: List[Dict[str,Any]], title:str):
        st.markdown(f"#### {title}")
        if not lines:
            st.info("Aucune ligne.")
            return
        rows=[]
        for ln in lines:
            df,_,_ = load_price_series_any(ln.get("isin") or ln.get("name"), None, st.session_state["EURO_RATE_PREVIEW"])
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
        st.dataframe(
            dfv.style.format({
                "Montant investi €":to_eur,
                "Quantité":"{:.6f}",
                "Prix achat":to_eur,
                "Dernier cours":to_eur
            }),
            use_container_width=True, hide_index=True)

    d1,d2 = st.columns(2)
    with d1: _detail_table(st.session_state["A_lines"], "Portefeuille 1 — Client (positions)")
    with d2: _detail_table(st.session_state["B_lines"], "Portefeuille 2 — Vous (positions)")
else:
    st.info("Ajoute des lignes (formulaire ou **Coller un tableau**), règle les versements dans la barre latérale, puis clique **Lancer la comparaison**.")

# ---------- Versements & affectation (sidebar persistant) ----------
def _alloc_sidebar(side_label: str, lines_key: str, prefix: str):
    st.subheader(side_label)
    m = st.number_input(f"Versement mensuel ({prefix})", min_value=0.0, value=0.0, step=100.0, key=f"m{prefix}")
    one_amt = st.number_input(f"Versement complémentaire ({prefix})", min_value=0.0, value=0.0, step=100.0, key=f"one{prefix}_amt")
    one_date = st.date_input(f"Date versement complémentaire ({prefix})", value=date.today(), key=f"one{prefix}_date")
    mode = st.selectbox(f"Affectation des versements ({prefix})",
                        ["Pro-rata montants initiaux", "Répartition personnalisée", "Tout sur un seul fonds"],
                        key=f"mode{prefix}")
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
    # stocker pour le run
    st.session_state[f"m{prefix}"] = m
    st.session_state[f"one{prefix}_amt"] = one_amt
    st.session_state[f"one{prefix}_date"] = one_date
    st.session_state[f"mode{prefix}"] = mode
    st.session_state[f"custom{prefix}"] = custom
    st.session_state[f"single{prefix}"] = single

with st.sidebar:
    st.header("⚙️ Paramètres de versement")
    _alloc_sidebar("Portefeuille 1 — Client", "A_lines", "A")
    st.divider()
    _alloc_sidebar("Portefeuille 2 — Vous", "B_lines", "B")

# ---------- Debug (optionnel) ----------
with st.expander("🔧 Debug EODHD (optionnel)"):
    test_q = st.text_input("Tester une recherche (ISIN, nom ou EUROFUND)")
    if test_q:
        st.write("Résultat /search :", eod_search(test_q))
        df_dbg, sym_dbg, note_dbg = load_price_series_any(test_q, None, st.session_state.get("EURO_RATE_PREVIEW", 2.0))
        st.write("Symbole testé :", sym_dbg)
        if note_dbg: st.caption(note_dbg)
        if not df_dbg.empty: st.dataframe(df_dbg.tail(5))
        else: st.warning("Aucune VL trouvée.")
