# =========================================
# app.py — Comparateur Portefeuilles CGP (simplifié)
# - Ordre des sources VL: EODHD -> VL synthétique (base_date + VL départ + VL actuelle)
# - Fonds en euros simulé (taux paramétrable, intérêts le 31/12)
# - Frais d’entrée (%) côté barre latérale (appliqués à tous les apports)
# - Saisie individuelle, "Coller un tableau", édition/suppression
# - Versements mensuels & ponctuels (modes d’affectation)
# - Courbes (option base 100), XIRR et delta de gains
# =========================================

# -------- 1) Imports & Config --------
import os, re, math, requests, calendar
from datetime import date
from typing import Optional, List, Dict, Any, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

st.set_page_config(page_title="Comparateur Portefeuilles CGP", page_icon="🦉", layout="wide")
TODAY = pd.Timestamp.today().normalize()

# -------- 2) Session state init --------
if "SYNTH_PARAMS" not in st.session_state:
    # keyU -> {"base_date": ts, "start_px": float, "last_px": float}
    st.session_state["SYNTH_PARAMS"] = {}

for key_ in ["A_lines", "B_lines"]:
    if key_ not in st.session_state:
        st.session_state[key_] = []

# -------- 3) Utils --------
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

def _looks_like_isin(s: str) -> bool:
    return bool(re.fullmatch(r"[A-Z]{2}[A-Z0-9]{9}[0-9]", (s or "").strip().upper()))

# -------- 4) EODHD client + resolution --------
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

def _eod_ok(sym: str) -> bool:
    try:
        js = eodhd_get(f"/eod/{sym}", params={"period": "d"})
        return isinstance(js, list) and len(js) > 0
    except Exception:
        return False

@st.cache_data(ttl=24*3600, show_spinner=False)
def resolve_symbol(q: str) -> Optional[str]:
    q = (q or "").strip()
    if not q:
        return None
    if _looks_like_isin(q):
        base = q.upper()
        for suf in [".EUFUND", ".FUND"]:
            cand = f"{base}{suf}"
            if _eod_ok(cand):
                return cand
        return None
    res = eod_search(q)
    if res:
        for it in res:
            code = str(it.get("Code", "")).strip()
            if code and _eod_ok(code):
                return code
    return None

# -------- 5) Fonds en euros (simulé) --------
def _eurofund_series(euro_rate: float,
                     start: pd.Timestamp = pd.Timestamp("1990-01-01"),
                     end: pd.Timestamp = TODAY) -> pd.Series:
    idx = pd.date_range(start=start, end=end, freq="D")
    vals = [1.0]
    for i in range(1, len(idx)):
        d = idx[i]
        v = vals[-1]
        if d.month == 12 and d.day == 31:
            v *= (1.0 + euro_rate/100.0)
        vals.append(v)
    return pd.Series(vals, index=idx, name="Close")

# -------- 6) EODHD price series (fund/etf/dol) --------
@st.cache_data(ttl=3*3600, show_spinner=False)
def eod_prices_any(symbol_or_isin: str,
                   start_dt: Optional[pd.Timestamp],
                   euro_rate: float) -> Tuple[pd.DataFrame, str, str]:
    q = (symbol_or_isin or "").strip()
    if not q:
        return pd.DataFrame(), q, "⚠️ identifiant vide."

    # Fonds en euros simulé
    if q.upper() in {"EUROFUND", "FONDS EN EUROS", "FONDS EN EUROS (SIMULÉ)"}:
        ser = _eurofund_series(euro_rate=euro_rate, start=pd.Timestamp("1990-01-01"), end=TODAY)
        if start_dt is not None:
            ser = ser.loc[ser.index >= start_dt]
        if not ser.empty:
            ser = ser / ser.iloc[0]
        return ser.to_frame(), "EUROFUND", f"Fonds en euros simulé — intérêts le 31/12 au taux {euro_rate:.2f}%/an (rebasé à 1 au départ)."

    qU = q.upper()
    note = ""

    def _fetch(sym: str, from_dt: Optional[str]) -> pd.DataFrame:
        try:
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
        except Exception:
            return pd.DataFrame()

    sym = resolve_symbol(qU) or qU
    df = _fetch(sym, None)
    if not df.empty:
        if start_dt is not None:
            df = df.loc[df.index >= start_dt]
        return df, sym, note

    return pd.DataFrame(), qU, "⚠️ Aucune VL récupérée."

@st.cache_data(ttl=3600, show_spinner=False)
def load_price_series_any(symbol_or_isin: str, from_dt: Optional[pd.Timestamp], euro_rate: float):
    return eod_prices_any(symbol_or_isin, from_dt, euro_rate)

# -------- 7) VL synthétique (départ + actuel -> interpolation) --------
def set_synth_params(key: str, base_date: pd.Timestamp,
                     start_px: float, last_px: float) -> None:
    st.session_state["SYNTH_PARAMS"][key.upper()] = {
        "base_date": pd.Timestamp(base_date),
        "start_px": float(start_px),
        "last_px": float(last_px),
    }

def get_synth_series(key: str, from_dt: Optional[pd.Timestamp]) -> pd.DataFrame:
    keyU = (key or "").strip().upper()
    params = st.session_state["SYNTH_PARAMS"].get(keyU)
    if not params:
        return pd.DataFrame()

    start0: pd.Timestamp = params["base_date"]
    s0: float = params["start_px"]
    sT: float = params["last_px"]

    if s0 <= 0 or sT <= 0:
        return pd.DataFrame()

    start = max(start0, pd.Timestamp(from_dt)) if from_dt is not None else start0
    end = TODAY
    if end <= start:
        return pd.DataFrame({"Close": [sT]}, index=[end])

    idx = pd.bdate_range(start=start, end=end)
    total_days = (end - start).days
    if total_days <= 0:
        total_days = 1

    # interpolation géométrique déterministe: passe par (start=s0) et (end=sT)
    growth = (sT / s0) ** (1.0 / total_days)
    vals = [s0]
    for i in range(1, len(idx)):
        vals.append(vals[-1] * growth)
    df = pd.DataFrame({"Close": vals}, index=idx)
    return df

# -------- 8) get_price_series: EODHD -> Synth --------
def get_price_series(symbol_or_isin: str, from_dt: Optional[pd.Timestamp], euro_rate: float) -> Tuple[pd.DataFrame, str, str]:
    key = (symbol_or_isin or "").strip()
    if not key:
        return pd.DataFrame(), "", "⚠️ identifiant vide."

    # 1) EODHD (ou fonds € simulé)
    df, sym, note = load_price_series_any(key, from_dt, euro_rate)
    if not df.empty:
        return df, sym, note

    # 2) Synthétique
    sdf = get_synth_series(key, from_dt)
    if not sdf.empty:
        return sdf, f"{key.upper()}.SYNTH", "VL synthétique (départ + actuelle)."

    base = key.upper().split(".")[0]
    if base != key.upper():
        sdf2 = get_synth_series(base, from_dt)
        if not sdf2.empty:
            return sdf2, f"{base}.SYNTH", "VL synthétique (départ + actuelle)."

    return pd.DataFrame(), sym, note

# -------- 9) Univers (exemples) --------
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

# -------- 10) Helpers pour UI & parsing --------
def _auto_name_from_isin(isin: str) -> str:
    if not isin:
        return ""
    if isin.upper() == "EUROFUND":
        return "Fonds en euros (simulé)"
    res = eod_search(isin)
    for it in res:
        if str(it.get("ISIN", "")).upper() == isin.upper():
            nm = str(it.get("Name", "")).strip()
            if nm:
                return nm
    return ""

def _get_close_on(df: pd.DataFrame, dt: pd.Timestamp) -> Optional[float]:
    if df.empty:
        return None
    if dt in df.index:
        v = df.loc[dt, "Close"]
        return float(v) if pd.notna(v) else None
    after = df.loc[df.index >= dt]
    if not after.empty:
        return float(after["Close"].iloc[0])
    return float(df["Close"].iloc[-1]) if not df.empty else None

def _month_end(d: pd.Timestamp) -> pd.Timestamp:
    last_day = calendar.monthrange(d.year, d.month)[1]
    return pd.Timestamp(year=d.year, month=d.month, day=last_day)

def _month_schedule(start_dt: pd.Timestamp, end_dt: pd.Timestamp) -> List[pd.Timestamp]:
    dates = []
    cur = pd.Timestamp(year=start_dt.year, month=start_dt.month, day=start_dt.day)
    cur = min(cur, _month_end(cur))
    while cur <= end_dt:
        dates.append(cur)
        y = cur.year + (cur.month // 12)
        m = 1 if cur.month == 12 else cur.month + 1
        day = min(start_dt.day, calendar.monthrange(y, m)[1])
        cur = pd.Timestamp(year=y, month=m, day=day)
    return dates

def _normalize_header(h: str) -> str:
    h = h.strip().lower()
    h = h.replace("€", "").replace("(", "").replace(")", "")
    h = h.replace("  ", " ")
    mapping = {
        "nom": "name", "nom du fonds": "name", "fonds": "name", "name": "name",
        "isin": "isin", "code isin": "isin", "isin code": "isin",
        "montant": "amount", "montant investi": "amount", "amount": "amount", "investi": "amount",
        "date": "buy_date", "date d'achat": "buy_date", "date achat": "buy_date", "buy date": "buy_date",
        "prix": "buy_price", "prix d'achat": "buy_price", "buy price": "buy_price"
    }
    return mapping.get(h, h)

def _detect_delimiter(s: str) -> Optional[str]:
    if "\t" in s: return "\t"
    counts = {";": s.count(";"), "|": s.count("|"), ",": s.count(",")}
    delim = max(counts, key=counts.get)
    if counts[delim] > 0: return delim
    return None

def parse_pasted_table(text: str) -> pd.DataFrame:
    lines = [ln for ln in text.strip().splitlines() if ln.strip()]
    if not lines: return pd.DataFrame()
    delim = _detect_delimiter(text)
    rows = []
    for ln in lines:
        parts = [p.strip() for p in (ln.split(delim) if delim else re.split(r"\s{2,}", ln.strip()))]
        rows.append(parts)
    header = rows[0]
    normalized = [_normalize_header(h) for h in header]
    known = set(normalized) & {"name", "isin", "amount", "buy_date", "buy_price"}
    if not known:
        cols = ["name", "isin", "amount"]
        if len(header) >= 4: cols.append("buy_date")
        if len(header) >= 5: cols.append("buy_price")
        data = rows
        df = pd.DataFrame(data, columns=cols[:len(data[0])])
    else:
        data = rows[1:]
        df = pd.DataFrame(data, columns=normalized[:len(rows[0])])

    for col in df.columns:
        df[col] = df[col].astype(str).str.strip()
        if col == "amount":
            df[col] = df[col].str.replace(" ", "").str.replace("€", "").str.replace(",", ".")
        if col == "buy_price":
            df[col] = df[col].str.replace(" ", "").str.replace("€", "").str.replace(",", ".")
    return df

def _standardize_df(df_raw: pd.DataFrame) -> pd.DataFrame:
    colmap = {c: _normalize_header(str(c)) for c in df_raw.columns}
    df = df_raw.rename(columns=colmap)
    keep = [c for c in ["name", "isin", "amount", "buy_price", "buy_date"] if c in df.columns]
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
            {"name": "R-co Valor C EUR", "isin": "FR0011253624", "amount": 10000, "buy_price": "", "buy_date": "2024-01-02"},
            {"name": "Fonds en euros (simulé)", "isin": "EUROFUND", "amount": 10000, "buy_price": "", "buy_date": "2024-01-02"},
        ],
        columns=["name", "isin", "amount", "buy_price", "buy_date"]
    )

def export_template_csv_bytes() -> bytes:
    return build_import_template_df().to_csv(index=False).encode("utf-8")

# -------- 11) VL synth UI (simplifiée) --------
with st.sidebar:
    st.header("🧮 VL synthétique (si VL absente)")
    synth_key = st.text_input("ISIN ou Nom (clé)")
    c1, c2 = st.columns(2)
    with c1:
        base_date = st.date_input("Date de départ", value=date(2018, 1, 1))
        start_px = st.number_input("VL de départ", min_value=0.0, value=100.0, step=0.01, format="%.2f")
    with c2:
        last_px = st.number_input("VL actuelle", min_value=0.0, value=120.0, step=0.01, format="%.2f")

    if st.button("✅ Enregistrer la VL synthétique"):
        key_in = (synth_key or "").strip()
        if not key_in:
            st.warning("Saisis une clé (ISIN ou Nom).")
        elif start_px <= 0 or last_px <= 0:
            st.warning("VL de départ et VL actuelle doivent être > 0.")
        else:
            set_synth_params(
                key=key_in,
                base_date=pd.Timestamp(base_date),
                start_px=float(start_px),
                last_px=float(last_px),
            )
            st.success(f"Paramètres enregistrés pour « {key_in.upper()} ».")

# -------- 12) Fonds en euros (taux) --------
with st.sidebar:
    st.header("💶 Fonds en euros — Paramètre global")
    EURO_RATE = st.number_input(
        "Taux annuel du fonds en euros (%)",
        min_value=0.0, max_value=10.0, value=2.0, step=0.1,
        help="Intérêts capitalisés le 31/12 (série rebasée à 1 au départ)."
    )
    st.session_state["EURO_RATE_PREVIEW"] = EURO_RATE

# -------- 13) Frais d’entrée (%) --------
with st.sidebar:
    st.header("💸 Frais d’entrée (%)")
    FEE_A = st.number_input("Frais d’entrée — Portefeuille 1 (Client)", min_value=0.0, max_value=10.0, value=0.0, step=0.1)
    FEE_B = st.number_input("Frais d’entrée — Portefeuille 2 (Vous)", min_value=0.0, max_value=10.0, value=0.0, step=0.1)
    st.caption("Ex.: 10 000 € investis avec 4 % de frais => 9 600 € réellement investis. Les frais s’appliquent aussi aux versements.")

# -------- 14) Saisie/édition d’une ligne --------
def _add_line_ui(port_key: str, title: str, euro_rate: float):
    st.subheader(title)
    with st.form(key=f"{port_key}_form", clear_on_submit=False):
        c1, c2 = st.columns([2, 1])
        with c1:
            sel = st.selectbox("Choisir un fonds (ou saisie libre) :", ["— Saisie libre —"] + [f"{r['name']} — {r['isin']}" for r in UNIVERSE_GENERALI], key=f"{port_key}_select")
        with c2:
            amount = st.text_input("Montant investi (brut) €", value="", key=f"{port_key}_amount")

        c3, c4 = st.columns(2)
        with c3:
            dt = st.date_input("Date d’achat", value=date(2024, 1, 2), key=f"{port_key}_date")
        with c4:
            px_opt = st.text_input("Prix d’achat (optionnel, sinon VL)", value="", key=f"{port_key}_px")

        if sel != "— Saisie libre —":
            name, isin = sel.split(" — ")
            name = name.strip()
            isin = isin.strip().upper()
        else:
            name = st.text_input("Nom du fonds (facultatif si ISIN saisi)", key=f"{port_key}_name").strip()
            isin = st.text_input("ISIN (recommandé ou EUROFUND)", key=f"{port_key}_isin").strip().upper()

        submitted = st.form_submit_button("➕ Ajouter", type="primary")

    if submitted:
        try:
            amt_gross = float(str(amount).replace(" ", "").replace(",", "."))
            assert amt_gross > 0
        except Exception:
            st.warning("Entre un **montant brut (€)** valide (>0).")
            st.stop()

        if not isin and not name:
            st.warning("Indique au minimum l’**ISIN** ou le **nom** du fonds.")
            st.stop()

        dfp, sym_used, note = get_price_series(isin or name, pd.Timestamp(dt), euro_rate)
        if dfp.empty:
            st.error("Impossible de récupérer des VL (EODHD ou synth). Si pas de VL, saisis une VL synthétique dans la barre latérale.")
            st.stop()

        px = None
        if px_opt:
            try:
                px = float(str(px_opt).replace(",", "."))
            except:
                px = None
        if px is None:
            px = _get_close_on(dfp, pd.Timestamp(dt))
        if not px or px <= 0:
            st.error("Prix d’achat non déterminable.")
            st.stop()

        if not name and isin:
            name = _auto_name_from_isin(isin) or "—"

        fee_pct = FEE_A if port_key == "A_lines" else FEE_B
        net_amt = amt_gross * (1.0 - fee_pct/100.0)
        qty = float(net_amt) / float(px)

        st.session_state[port_key].append({
            "name": name or "—",
            "isin": isin or "",
            "amount_gross": float(amt_gross),
            "amount_net": float(net_amt),
            "qty_calc": float(qty),
            "buy_date": pd.Timestamp(dt),
            "buy_px": float(px),
            "sym_used": sym_used,
            "note": note,
        })
        st.success(f"Ligne ajoutée ({name or isin})")

def _line_card(line: Dict[str, Any], idx: int, port_key: str):
    state_key = f"edit_mode_{port_key}_{idx}"
    if state_key not in st.session_state:
        st.session_state[state_key] = False

    fee_pct = FEE_A if port_key == "A_lines" else FEE_B

    with st.container(border=True):
        header = st.columns([3, 2, 2, 2, 1])
        with header[0]:
            st.markdown(f"**{line.get('name','—')}**")
            st.caption(f"ISIN : `{line.get('isin','—')}` • Symbole : ")
            st.code(line.get('sym_used','—'))
        with header[1]:
            st.markdown(f"Investi (brut)\n\n**{to_eur(line.get('amount_gross',0.0))}**")
            st.caption(f"Net après frais {fee_pct:.1f}% : **{to_eur(line.get('amount_net',0.0))}**")
            st.caption(f"le {fmt_date(line.get('buy_date'))}")
        with header[2]:
            st.markdown(f"Prix achat\n\n**{to_eur(line.get('buy_px'))}**")
            st.caption(f"Quantité : {line.get('qty_calc'):.6f}")
            if line.get("note"):
                st.caption(line["note"])
        with header[3]:
            try:
                df_last, _, _ = get_price_series(line.get("isin") or line.get("name"),
                                                 None, st.session_state.get("EURO_RATE_PREVIEW", 2.0))
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
                    dfp, _, _ = get_price_series(line.get("isin") or line.get("name"),
                                                 buy_ts, st.session_state.get("EURO_RATE_PREVIEW", 2.0))
                    if dfp.empty:
                        st.error("Impossible de recalculer la VL au jour choisi.")
                        st.stop()

                    if new_px.strip():
                        try:
                            px = float(str(new_px).replace(",", "."))
                        except Exception:
                            px = _get_close_on(dfp, buy_ts)
                    else:
                        px = _get_close_on(dfp, buy_ts)

                    if not px or px <= 0:
                        st.error("Prix d’achat non déterminable.")
                        st.stop()

                    net_amt = amt_gross * (1.0 - fee_pct/100.0)
                    qty = float(net_amt) / float(px)

                    line["amount_gross"] = float(amt_gross)
                    line["amount_net"] = float(net_amt)
                    line["buy_date"] = buy_ts
                    line["buy_px"] = float(px)
                    line["qty_calc"] = float(qty)

                    st.session_state[state_key] = False
                    st.success("Ligne mise à jour.")
                    st.experimental_rerun()

# -------- 15) Coller un tableau --------
def lines_from_dataframe(df_std: pd.DataFrame, euro_rate: float, default_dt: Optional[pd.Timestamp], fee_pct: float) -> List[Dict[str, Any]]:
    lines: List[Dict[str, Any]] = []
    for _, row in df_std.iterrows():
        isin = str(row.get("isin", "") or "").strip().upper()
        name = str(row.get("name", "") or "").strip()
        amount = row.get("amount", None)
        buy_price = row.get("buy_price", None)
        buy_date = row.get("buy_date", None)

        if (not isin and not name) or (amount is None or pd.isna(amount) or float(amount) <= 0):
            continue

        if not buy_date or pd.isna(buy_date):
            buy_ts = pd.Timestamp(default_dt) if default_dt is not None else TODAY
        else:
            buy_ts = pd.Timestamp(buy_date)

        dfp, sym_used, note = get_price_series(isin or name, buy_ts, euro_rate)
        if dfp.empty:
            continue

        if (buy_price is not None) and not pd.isna(buy_price) and float(buy_price) > 0:
            px = float(buy_price)
        else:
            px = _get_close_on(dfp, buy_ts)
        if not px or px <= 0:
            continue

        if not name and isin:
            name = _auto_name_from_isin(isin) or (isin if isin else "—")

        amt_gross = float(amount)
        net_amt = amt_gross * (1.0 - fee_pct/100.0)
        qty = float(net_amt) / float(px)

        lines.append({
            "name": name or "—",
            "isin": isin or "",
            "amount_gross": float(amt_gross),
            "amount_net": float(net_amt),
            "qty_calc": float(qty),
            "buy_date": buy_ts,
            "buy_px": float(px),
            "sym_used": sym_used,
            "note": note,
        })
    return lines

def _expander_paste_block(who_label: str, paste_key_prefix: str, lines_key: str, fee_pct: float):
    with st.expander(f"📋 Coller un tableau (Nom | ISIN | Montant | [Date] | [Prix]) — {who_label}"):
        default_dt = st.date_input("Date d’achat par défaut (si absente)", value=date(2024, 1, 2), key=f"default_dt_{paste_key_prefix}")
        text = st.text_area("Colle ici depuis Excel/Sheets", height=180, key=f"paste_{paste_key_prefix}")
        if st.button(f"🔎 Prévisualiser ({who_label})"):
            if not text.strip():
                st.warning("Rien à parser.")
            else:
                dfp = parse_pasted_table(text)
                if dfp.empty:
                    st.warning("Impossible de détecter un tableau.")
                else:
                    st.write("Aperçu :", dfp)
                    st.session_state[f"paste_preview_{paste_key_prefix}"] = dfp
                    st.session_state[f"paste_default_dt_{paste_key_prefix}"] = pd.Timestamp(default_dt)

        if st.button(f"➕ Ajouter ces lignes ({who_label})"):
            dfp = st.session_state.get(f"paste_preview_{paste_key_prefix}", pd.DataFrame())
            default_dt_saved = st.session_state.get(f"paste_default_dt_{paste_key_prefix}", None)
            if default_dt_saved is None:
                default_dt_saved = pd.Timestamp(default_dt)
            if dfp.empty:
                st.warning("Fais d’abord la prévisualisation.")
            else:
                dfp = dfp.replace("", np.nan)
                for col in ["amount", "buy_price"]:
                    if col in dfp.columns:
                        dfp[col] = pd.to_numeric(dfp[col], errors="coerce")
                new_lines = lines_from_dataframe(dfp, st.session_state["EURO_RATE_PREVIEW"], default_dt=default_dt_saved, fee_pct=fee_pct)
                if not new_lines:
                    st.warning("Aucune ligne valide à ajouter. Si un fonds n’a pas de VL, crée une **VL synthétique** (barre latérale).")
                else:
                    st.session_state[lines_key].extend(new_lines)
                    st.success(f"{len(new_lines)} ligne(s) ajoutée(s).")
                    st.experimental_rerun()

# -------- 16) Sidebar paramètres de versement --------
def _weights_for(lines: List[Dict[str, Any]], mode: str, custom: Dict[int, float], single_id: Optional[int]) -> Dict[int, float]:
    if not lines:
        return {}
    if mode == "Tout sur un seul fonds" and single_id is not None:
        return {id(ln): (1.0 if id(ln) == single_id else 0.0) for ln in lines}
    if mode == "Répartition personnalisée" and custom:
        s = sum(max(0.0, v) for v in custom.values())
        if s <= 0:
            return {id(ln): 1.0 / len(lines) for ln in lines}
        return {k: max(0.0, v) / s for k, v in custom.items()}
    total = sum(float(ln.get("amount_net", 0.0)) for ln in lines)  # pro-rata sur le net investi initial
    if total > 0:
        return {id(ln): float(ln.get("amount_net", 0.0)) / total for ln in lines}
    return {id(ln): 1.0 / len(lines) for ln in lines}

def _alloc_sidebar(side_label: str, lines_key: str, prefix: str):
    st.subheader(side_label)
    m = st.number_input(f"Versement mensuel (brut) — {prefix}", min_value=0.0, value=0.0, step=100.0, key=f"m_{prefix}")
    one_amt = st.number_input(f"Versement ponctuel (brut) — {prefix}", min_value=0.0, value=0.0, step=100.0, key=f"one_{prefix}_amt")
    one_date = st.date_input(f"Date versement ponctuel — {prefix}", value=date.today(), key=f"one_{prefix}_date")
    mode = st.selectbox(f"Affectation des versements — {prefix}",
                        ["Pro-rata (net init.)", "Répartition personnalisée", "Tout sur un seul fonds"],
                        key=f"mode_{prefix}")
    custom = {}
    single = None
    lines = st.session_state[lines_key]
    if mode == "Répartition personnalisée":
        if lines:
            st.caption("Répartir sur les lignes ci-dessous (total ≈ 100 %).")
            default = round(100.0 / len(lines), 2)
            tot = 0.0
            for i, ln in enumerate(lines):
                w = st.slider(f"{ln['name']} ({ln['isin']})", 0.0, 100.0, default, 1.0, key=f"{prefix}_w{i}")
                custom[id(ln)] = w / 100.0
                tot += w
            if abs(tot - 100.0) > 1.0:
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

# -------- 17) Onglets Portefeuilles --------
st.title("🟣 Comparer deux portefeuilles (Client vs Vous)")
tabA, tabB = st.tabs(["📁 Portefeuille 1 — Client", "🟣 Portefeuille 2 — Vous"])

with tabA:
    _add_line_ui("A_lines", "Portefeuille 1 — Client", st.session_state.get("EURO_RATE_PREVIEW", 2.0))
    _expander_paste_block("Client", "A", "A_lines", FEE_A)
    for i, ln in enumerate(st.session_state["A_lines"]):
        _line_card(ln, i, "A_lines")

with tabB:
    _add_line_ui("B_lines", "Portefeuille 2 — Vous", st.session_state.get("EURO_RATE_PREVIEW", 2.0))
    _expander_paste_block("Vous", "B", "B_lines", FEE_B)
    for i, ln in enumerate(st.session_state["B_lines"]):
        _line_card(ln, i, "B_lines")

# -------- 18) Simulation & XIRR --------
def xnpv(rate: float, cash_flows: List[Tuple[pd.Timestamp, float]]) -> float:
    t0 = cash_flows[0][0]
    return sum(cf / ((1 + rate) ** ((t - t0).days / 365.2425)) for t, cf in cash_flows)

def xirr(cash_flows: List[Tuple[pd.Timestamp, float]]) -> Optional[float]:
    if not cash_flows or len(cash_flows) < 2:
        return None
    lo, hi = -0.9999, 10.0
    for _ in range(100):
        mid = (lo + hi) / 2
        val = xnpv(mid, cash_flows)
        if abs(val) < 1e-6:
            return mid
        if xnpv(lo, cash_flows) * val < 0:
            hi = mid
        else:
            lo = mid
    return None

def simulate_portfolio(lines: List[Dict[str, Any]],
                       monthly_amt_gross: float,
                       one_amt_gross: float, one_date: date,
                       alloc_mode: str,
                       custom_weights: Dict[int, float],
                       single_target: Optional[int],
                       euro_rate: float,
                       fee_pct: float) -> Tuple[pd.DataFrame, float, float, float, Optional[float], pd.Timestamp, pd.Timestamp]:
    """
    Retourne: (df_valeur, total_investi_brut, total_investi_net, valeur_finale, xirr%, start_min, start_full)
    """
    if not lines:
        return pd.DataFrame(), 0.0, 0.0, 0.0, None, TODAY, TODAY

    series: Dict[int, pd.Series] = {}
    for ln in lines:
        df, _, _ = get_price_series(ln.get("isin") or ln.get("name"), None, euro_rate)
        if not df.empty:
            series[id(ln)] = df["Close"]
    if not series:
        return pd.DataFrame(), 0.0, 0.0, 0.0, None, TODAY, TODAY

    eff_buy_date: Dict[int, pd.Timestamp] = {}
    qty_init: Dict[int, float] = {}
    for ln in lines:
        sid = id(ln)
        s = series.get(sid)
        if s is None or s.empty:
            continue
        d_buy = pd.Timestamp(ln["buy_date"])
        if d_buy in s.index:
            px_buy = float(s.loc[d_buy]); eff_dt = d_buy
        else:
            after = s.loc[s.index >= d_buy]
            if after.empty:
                px_buy = float(s.iloc[-1]); eff_dt = s.index[-1]
            else:
                px_buy = float(after.iloc[0]); eff_dt = after.index[0]
        px_manual = ln.get("buy_px", None)
        px_for_qty = float(px_manual) if (px_manual and px_manual > 0) else px_buy
        # qty déjà calculée lors de l’ajout (sur amount_net). On recalcule par sécurité.
        net_amt = float(ln.get("amount_net", ln.get("amount_gross", 0.0) * (1.0 - fee_pct/100.0)))
        qty_init[sid] = float(net_amt) / float(px_for_qty)
        eff_buy_date[sid] = eff_dt

    start_min = min(eff_buy_date.values())
    start_full = max(eff_buy_date.values())

    idx = pd.Index(sorted(set().union(*[s.index for s in series.values()])))
    idx = idx[(idx >= start_min) & (idx <= TODAY)]
    if len(idx) == 0:
        return pd.DataFrame(), 0.0, 0.0, 0.0, None, start_min, start_full

    qty_curr: Dict[int, float] = {id(ln): 0.0 for ln in lines}
    weights = _weights_for(lines, alloc_mode, custom_weights, single_target)

    cash_flows: List[Tuple[pd.Timestamp, float]] = []
    total_brut = 0.0
    total_net = 0.0

    # apports initiaux
    for ln in lines:
        sid = id(ln)
        if sid in eff_buy_date:
            brut = float(ln.get("amount_gross", 0.0))
            net = float(ln.get("amount_net", 0.0))
            total_brut += brut
            total_net += net
            cash_flows.append((eff_buy_date[sid], -brut))

    # plannings
    sched = _month_schedule(start_min, TODAY) if monthly_amt_gross > 0 else []
    one_dt = pd.Timestamp(one_date) if one_amt_gross > 0 else None

    values = []
    for d in idx:
        # init
        for ln in lines:
            sid = id(ln)
            if sid in eff_buy_date and d == eff_buy_date[sid]:
                qty_curr[sid] += qty_init[sid]

        # one-off brut -> net après frais -> acheter
        if one_dt is not None and d == one_dt and one_amt_gross > 0:
            net_amt = one_amt_gross * (1.0 - fee_pct/100.0)
            for ln in lines:
                sid = id(ln); s = series.get(sid)
                if s is None or s.empty: continue
                w = weights.get(sid, 0.0)
                if w <= 0: continue
                px = float(s.loc[d]) if d in s.index else float(s.loc[s.index >= d].iloc[0])
                qty_curr[sid] += (net_amt * w) / px
            cash_flows.append((d, -float(one_amt_gross)))
            total_brut += float(one_amt_gross)
            total_net += float(net_amt)

        # mensualités
        if d in sched and monthly_amt_gross > 0:
            net_amt_m = monthly_amt_gross * (1.0 - fee_pct/100.0)
            for ln in lines:
                sid = id(ln); s = series.get(sid)
                if s is None or s.empty: continue
                w = weights.get(sid, 0.0)
                if w <= 0: continue
                px = float(s.loc[d]) if d in s.index else float(s.loc[s.index >= d].iloc[0])
                qty_curr[sid] += (net_amt_m * w) / px
            cash_flows.append((d, -float(monthly_amt_gross)))
            total_brut += float(monthly_amt_gross)
            total_net += float(net_amt_m)

        # valorisation
        v = 0.0
        for ln in lines:
            sid = id(ln); s = series.get(sid)
            if d in s.index:
                px = float(s.loc[d])
            else:
                before = s.loc[s.index <= d]
                if before.empty:
                    continue
                px = float(before.iloc[-1])
            v += qty_curr[sid] * px
        values.append((d, v))

    df_val = pd.DataFrame(values, columns=["date", "Valeur"]).set_index("date")
    final_val = float(df_val["Valeur"].iloc[-1]) if not df_val.empty else 0.0
    cash_flows.append((TODAY, final_val))
    irr = xirr(cash_flows)
    return df_val, total_brut, total_net, final_val, (irr * 100.0 if irr is not None else None), start_min, start_full

# -------- 19) Run & affichage --------
rebase_100 = st.checkbox("Normaliser les courbes à 100 au départ", value=False)
st.divider()
run = st.button("🚀 Lancer la comparaison", type="primary")

if run:
    dfA, investA_brut, investA_net, valA, xirrA, A_first, A_full = simulate_portfolio(
        st.session_state["A_lines"],
        st.session_state.get("m_A", 0.0),
        st.session_state.get("one_A_amt", 0.0),
        st.session_state.get("one_A_date", date.today()),
        st.session_state.get("mode_A", "Pro-rata (net init.)"),
        {k: v for k, v in st.session_state.items() if str(k).startswith("A_") and isinstance(v, float)},
        st.session_state.get("A_single_pick", None),
        st.session_state.get("EURO_RATE_PREVIEW", 2.0),
        FEE_A
    )
    dfB, investB_brut, investB_net, valB, xirrB, B_first, B_full = simulate_portfolio(
        st.session_state["B_lines"],
        st.session_state.get("m_B", 0.0),
        st.session_state.get("one_B_amt", 0.0),
        st.session_state.get("one_B_date", date.today()),
        st.session_state.get("mode_B", "Pro-rata (net init.)"),
        {k: v for k, v in st.session_state.items() if str(k).startswith("B_") and isinstance(v, float)},
        st.session_state.get("B_single_pick", None),
        st.session_state.get("EURO_RATE_PREVIEW", 2.0),
        FEE_B
    )

    A_start, B_start = A_first, B_first
    dfA_plot = dfA.loc[dfA.index >= A_start].copy() if not dfA.empty else dfA
    dfB_plot = dfB.loc[dfB.index >= B_start].copy() if not dfB.empty else dfB

    if rebase_100:
        if not dfA_plot.empty: dfA_plot["Valeur"] = 100 * dfA_plot["Valeur"] / dfA_plot["Valeur"].iloc[0]
        if not dfB_plot.empty: dfB_plot["Valeur"] = 100 * dfB_plot["Valeur"] / dfB_plot["Valeur"].iloc[0]

    st.subheader("📈 Évolution de la valeur des portefeuilles")
    if not dfA_plot.empty or not dfB_plot.empty:
        df_plot = pd.DataFrame(index=sorted(set(dfA_plot.index).union(dfB_plot.index)))
        if not dfA_plot.empty: df_plot["Client"] = dfA_plot["Valeur"]
        if not dfB_plot.empty: df_plot["Vous"] = dfB_plot["Valeur"]

        y_label = "Indice (base 100)" if rebase_100 else "Valeur (€)"
        starts = []
        if not dfA_plot.empty: starts.append(float(dfA_plot["Valeur"].iloc[0]))
        if not dfB_plot.empty: starts.append(float(dfB_plot["Valeur"].iloc[0]))
        base_min = min(starts) if starts else (min(df_plot.min()) if not df_plot.empty else 0.0)
        y_max = max(df_plot.max()) if not df_plot.empty else 1.0
        y_margin = (y_max - base_min) * 0.05
        y_min_adj = max(0.0, base_min - y_margin)

        fig = px.line(df_plot, x=df_plot.index, y=df_plot.columns,
                      labels={"value": y_label, "index": "Date"},
                      title="Valeur quotidienne (versements pris en compte, frais d’entrée déduits)")
        fig.update_yaxes(range=[y_min_adj, y_max + y_margin])
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Ajoute des lignes pour au moins un portefeuille.")

    st.subheader("📊 Synthèse chiffrée")
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Investi BRUT (Client)", to_eur(investA_brut))
    c2.metric("Net investi (Client)", to_eur(investA_net))
    c3.metric("XIRR (Client)", f"{xirrA:.2f}%" if xirrA is not None else "—")
    c4.metric("Investi BRUT (Vous)", to_eur(investB_brut))
    c5.metric("Net investi (Vous)", to_eur(investB_net))
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

    # Détail positions
    def _detail_table(lines: List[Dict[str, Any]], title: str):
        st.markdown(f"#### {title}")
        if not lines:
            st.info("Aucune ligne.")
            return
        rows = []
        for ln in lines:
            df_, _, _ = get_price_series(ln.get("isin") or ln.get("name"),
                                         None, st.session_state.get("EURO_RATE_PREVIEW", 2.0))
            last = float(df_["Close"].iloc[-1]) if not df_.empty else np.nan
            rows.append({
                "Nom": ln.get("name", "—"),
                "ISIN": ln.get("isin", "—"),
                "Symbole": ln.get("sym_used", "—"),
                "Montant brut €": float(ln.get("amount_gross", 0.0)),
                "Net investi €": float(ln.get("amount_net", 0.0)),
                "Quantité": float(ln.get("qty_calc", 0.0)),
                "Prix achat": float(ln.get("buy_px", np.nan)),
                "Dernier cours": last,
                "Date d’achat": fmt_date(ln.get("buy_date")),
            })
        dfv = pd.DataFrame(rows)
        st.dataframe(
            dfv.style.format({
                "Montant brut €": to_eur,
                "Net investi €": to_eur,
                "Quantité": "{:.6f}",
                "Prix achat": to_eur,
                "Dernier cours": to_eur
            }),
            use_container_width=True, hide_index=True
        )

    d1, d2 = st.columns(2)
    with d1: _detail_table(st.session_state["A_lines"], "Portefeuille 1 — Client (positions)")
    with d2: _detail_table(st.session_state["B_lines"], "Portefeuille 2 — Vous (positions)")
else:
    st.info("Ajoute des lignes (formulaire ou **Coller un tableau**), règle les frais & versements, puis clique **Lancer la comparaison**.")

# -------- 20) Debug --------
with st.expander("🔧 Debug EODHD / Synth (optionnel)"):
    test_q = st.text_input("Tester une recherche (ISIN, nom ou EUROFUND)")
    if test_q:
        st.write("Résultat /search :", eod_search(test_q))
        df_dbg, sym_dbg, note_dbg = get_price_series(test_q, None, st.session_state.get("EURO_RATE_PREVIEW", 2.0))
        st.write("Symbole testé :", sym_dbg)
        if note_dbg: st.caption(note_dbg)
        if not df_dbg.empty: st.dataframe(df_dbg.tail(5))
        else: st.warning("Aucune VL trouvée (EODHD ou synth).")
