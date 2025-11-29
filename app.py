from __future__ import annotations

import json
from datetime import date
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
import streamlit as st
import altair as alt

# ------------------------------------------------------------
# Constantes & univers de fonds recommandés
# ------------------------------------------------------------
TODAY = pd.Timestamp.today().normalize()

APP_TITLE = "Comparateur de portefeuilles — Version V2 (EODHD All-in-One)"

# Fonds "core" / "défensifs" / "fonds en euros"
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
    ("R-Co Target 2029 HY", "FR0014002XJ3"),   # à corriger si besoin
    ("Euro Bond 1-3 Years", "LU0321462953"),   # exemple
]


# ------------------------------------------------------------
# Utilitaires formatage & XIRR
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
# API EODHD — wrapper générique + search + prix
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


# ------------------------------------------------------------
# EODHD Fundamentals : récupération + normalisation des breakdowns
# ------------------------------------------------------------
@st.cache_data(show_spinner=False, ttl=3600)
def eodhd_fundamentals(symbol: str) -> Dict[str, Any]:
    """
    Récupère le JSON complet des fondamentaux EODHD pour un symbole
    (fonds/ETF/action).
    """
    try:
        js = eodhd_get(f"/fundamentals/{symbol}", params=None)
        if isinstance(js, dict):
            return js
        return {}
    except Exception:
        return {}


def _deep_find_first(node: Any, key_candidates: List[str]) -> Any:
    """
    Parcours récursif du JSON pour trouver la première valeur dont la clé
    est dans key_candidates. Retourne la valeur trouvée ou None.
    """
    if isinstance(node, dict):
        for k, v in node.items():
            if k in key_candidates:
                return v
            res = _deep_find_first(v, key_candidates)
            if res is not None:
                return res
    elif isinstance(node, list):
        for it in node:
            res = _deep_find_first(it, key_candidates)
            if res is not None:
                return res
    return None


def _normalize_breakdown(raw: Any) -> Dict[str, float]:
    """
    Transforme un breakdown quelle que soit sa forme en dict {label: float%}.
    Gère à la fois les dicts et les listes d'objets.
    """
    out: Dict[str, float] = {}

    if isinstance(raw, dict):
        for k, v in raw.items():
            label = str(k)
            val = None
            if isinstance(v, dict):
                for key_pct in ("Fund_%", "Assets_%", "Net_Assets_%", "Equity_%", "Weight", "Percent"):
                    if key_pct in v:
                        val = v[key_pct]
                        break
            else:
                val = v
            try:
                if val is not None:
                    out[label] = float(val)
            except Exception:
                continue

    if isinstance(raw, list):
        for item in raw:
            if not isinstance(item, dict):
                continue
            label = item.get("Name") or item.get("Region") or item.get("Sector") or item.get("Code")
            val = None
            for key_pct in ("Fund_%", "Assets_%", "Net_Assets_%", "Equity_%", "Weight", "Percent"):
                if key_pct in item:
                    val = item[key_pct]
                    break
            if not label or val is None:
                continue
            try:
                out[str(label)] = float(val)
            except Exception:
                continue

    out = {k: v for k, v in out.items() if v is not None and v > 0}
    return out


@st.cache_data(show_spinner=False, ttl=3600)
def get_fund_breakdowns(symbol: str) -> Dict[str, Dict[str, float] | List[Dict[str, Any]]]:
    """
    Retourne un dict avec :
      - 'asset_allocation': {classe: %}
      - 'regions': {région: %}
      - 'sectors': {secteur: %}
      - 'top10': liste de dicts {name, weight}

    Logique renforcée :
    - on essaie plusieurs symboles pour les fundamentals :
        1) le symbole utilisé pour les prix (ex. FR0011253624.EUFUND)
        2) l'ISIN nu (ex. FR0011253624)
        3) les codes retournés par eodhd_search()
    - si EODHD renvoie un champ 'error' / 'Error', on le signale dans l'UI.
    """

    def _try_symbol(sym: str) -> Optional[Dict[str, Any]]:
        if not sym:
            return None
        js = eodhd_fundamentals(sym)
        if not js:
            return None

        err_msg = js.get("error") or js.get("Error")
        if err_msg:
            st.caption(f"⚠️ EODHD Fundamentals indisponibles pour **{sym}** : {err_msg}")
            return None

        has_any_breakdown = _deep_find_first(
            js,
            [
                "Asset_Allocation",
                "AssetAllocation",
                "World_Regions",
                "WorldRegions",
                "Sector_Weights",
                "SectorWeights",
                "Sector_Weightings",
                "Top_10_Holdings",
                "Top10Holdings",
                "Top_Holdings",
            ],
        )
        if not has_any_breakdown:
            return js
        return js

    candidates: List[str] = []
    base = symbol.strip()
    if base:
        candidates.append(base)
        if "." in base:
            root = base.split(".", 1)[0]
            candidates.append(root)

        try:
            search_res = eodhd_search(base)
            for it in search_res:
                code = it.get("Code")
                exch = it.get("Exchange")
                if code and exch:
                    candidates.append(f"{code}.{exch}")
                elif code:
                    candidates.append(code)
        except Exception:
            pass

    seen = set()
    uniq_candidates: List[str] = []
    for c in candidates:
        if c and c not in seen:
            seen.add(c)
            uniq_candidates.append(c)

    js_used: Optional[Dict[str, Any]] = None
    for sym in uniq_candidates:
        js_try = _try_symbol(sym)
        if js_try is not None:
            js_used = js_try
            break

    if not js_used:
        return {
            "asset_allocation": {},
            "regions": {},
            "sectors": {},
            "top10": [],
        }

    raw_alloc = _deep_find_first(js_used, ["Asset_Allocation", "AssetAllocation"])
    raw_regions = _deep_find_first(js_used, ["World_Regions", "WorldRegions"])
    raw_sectors = _deep_find_first(js_used, ["Sector_Weights", "SectorWeights", "Sector_Weightings"])
    raw_top10 = _deep_find_first(js_used, ["Top_10_Holdings", "Top10Holdings", "Top_Holdings"])

    alloc = _normalize_breakdown(raw_alloc) if raw_alloc is not None else {}
    regions = _normalize_breakdown(raw_regions) if raw_regions is not None else {}
    sectors = _normalize_breakdown(raw_sectors) if raw_sectors is not None else {}

    top10_list: List[Dict[str, Any]] = []
    if isinstance(raw_top10, list):
        for item in raw_top10:
            if not isinstance(item, dict):
                continue
            label = (
                item.get("Name")
                or item.get("Code")
                or item.get("Ticker")
                or item.get("Symbol")
            )
            val = None
            for key_pct in ("Assets_%", "Fund_%", "Weight", "Percent"):
                if key_pct in item:
                    val = item[key_pct]
                    break
            if not label or val is None:
                continue
            try:
                w = float(val)
            except Exception:
                continue
            top10_list.append({"name": str(label), "weight": w})

    def _renormalize(d: Dict[str, float]) -> Dict[str, float]:
        s = sum(d.values())
        if s <= 0:
            return {}
        return {k: 100.0 * v / s for k, v in d.items()}

    alloc = _renormalize(alloc)
    regions = _renormalize(regions)
    sectors = _renormalize(sectors)

    return {
        "asset_allocation": alloc,
        "regions": regions,
        "sectors": sectors,
        "top10": top10_list,
    }


def _accumulate_weighted(target: Dict[str, float], src: Dict[str, float], w_line: float) -> None:
    """
    Ajoute à 'target' les pourcentages d'un breakdown de fonds
    pondérés par le poids du fonds dans le portefeuille.
    src = {label: %du fonds}, w_line = poids du fonds dans le portefeuille (0-1).
    """
    for k, v in src.items():
        target[k] = target.get(k, 0.0) + w_line * (v / 100.0)


def aggregate_portfolio_breakdowns(
    lines: List[Dict[str, Any]],
    euro_rate: float,
    fee_pct: float,
) -> Dict[str, Any]:
    """
    Calcule la répartition globale du portefeuille à partir des breakdowns EODHD
    et des montants investis nets (investi net des frais d'entrée).
    """
    if not lines:
        return {
            "asset_allocation": {},
            "regions": {},
            "sectors": {},
            "top10": [],
        }

    total_net = 0.0
    by_symbol_net: Dict[str, float] = {}

    for ln in lines:
        isin_or_name = ln.get("isin") or ln.get("name")
        if not isin_or_name:
            continue

        if str(isin_or_name).upper() == "EUROFUND":
            symbol = "EUROFUND"
        else:
            sym_used = ln.get("sym_used", "")
            if not sym_used:
                df_tmp, sym, _dbg = get_price_series(isin_or_name, None, euro_rate)
                symbol = sym
                ln["sym_used"] = sym
            else:
                symbol = sym_used

        if not symbol:
            continue

        try:
            brut = float(ln.get("amount_gross", 0.0))
        except Exception:
            brut = 0.0
        if brut <= 0:
            continue

        net = brut * (1.0 - fee_pct / 100.0)
        if net <= 0:
            continue

        total_net += net
        by_symbol_net[symbol] = by_symbol_net.get(symbol, 0.0) + net

    if total_net <= 0:
        return {
            "asset_allocation": {},
            "regions": {},
            "sectors": {},
            "top10": [],
        }

    alloc_tot: Dict[str, float] = {}
    regions_tot: Dict[str, float] = {}
    sectors_tot: Dict[str, float] = {}
    top10_tot: Dict[str, float] = {}

    for symbol, net_amt in by_symbol_net.items():
        w_line = net_amt / total_net

        if symbol == "EUROFUND":
            _accumulate_weighted(alloc_tot, {"Fonds en euros": 100.0}, w_line)
            continue

        bks = get_fund_breakdowns(symbol)
        alloc = bks.get("asset_allocation", {}) or {}
        regs = bks.get("regions", {}) or {}
        sect = bks.get("sectors", {}) or {}
        top10 = bks.get("top10", []) or []

        _accumulate_weighted(alloc_tot, alloc, w_line)
        _accumulate_weighted(regions_tot, regs, w_line)
        _accumulate_weighted(sectors_tot, sect, w_line)

        for h in top10:
            nm = h.get("name")
            wt = h.get("weight")
            if nm is None or wt is None:
                continue
            try:
                wt = float(wt)
            except Exception:
                continue
            top10_tot[nm] = top10_tot.get(nm, 0.0) + w_line * (wt / 100.0)

    def _as_percent(d: Dict[str, float]) -> Dict[str, float]:
        s = sum(d.values())
        if s <= 0:
            return {}
        return {k: 100.0 * v / s for k, v in d.items()}

    alloc_tot = _as_percent(alloc_tot)
    regions_tot = _as_percent(regions_tot)
    sectors_tot = _as_percent(sectors_tot)

    sorted_holdings = sorted(top10_tot.items(), key=lambda kv: kv[1], reverse=True)
    top10_list = [
        {"name": nm, "weight": 100.0 * w}
        for nm, w in sorted_holdings[:10]
    ]

    return {
        "asset_allocation": alloc_tot,
        "regions": regions_tot,
        "sectors": sectors_tot,
        "top10": top10_list,
    }


def _pie_chart_from_dict(title: str, d: Dict[str, float]):
    if not d:
        st.caption(f"{title} : données indisponibles.")
        return
    df = (
        pd.DataFrame({"Label": list(d.keys()), "Poids": list(d.values())})
        .sort_values("Poids", ascending=False)
    )
    chart = (
        alt.Chart(df)
        .mark_arc()
        .encode(
            theta=alt.Theta("Poids:Q", title="Poids (%)"),
            color=alt.Color("Label:N", legend=alt.Legend(title=title)),
            tooltip=["Label", alt.Tooltip("Poids:Q", format=".1f")],
        )
        .properties(height=260, width=260)
    )
    st.altair_chart(chart, use_container_width=False)


def _bar_chart_top_holdings(title: str, holdings: List[Dict[str, Any]]):
    if not holdings:
        st.caption(f"{title} : données indisponibles.")
        return
    df = (
        pd.DataFrame(holdings)
        .sort_values("weight", ascending=False)
        .head(10)
    )
    chart = (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x=alt.X("weight:Q", title="Poids (%)"),
            y=alt.Y("name:N", sort="-x", title=""),
            tooltip=["name", alt.Tooltip("weight:Q", format=".1f")],
        )
        .properties(height=260)
    )
    st.altair_chart(chart, use_container_width=True)


# ------------------------------------------------------------
# Symboles / séries de prix / helpers existants
# ------------------------------------------------------------
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


# ------------------------------------------------------------
# Calendrier de versements & poids
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
def compute_line_metrics(line: Dict[str, Any], fee_pct: float, euro_rate: float) -> Tuple[float, float, float]:
    amt_brut = float(line.get("amount_gross", 0.0))
    net_amt = amt_brut * (1.0 - fee_pct / 100.0)
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


# ------------------------------------------------------------
# Simulation d'un portefeuille
# ------------------------------------------------------------
def simulate_portfolio(
    lines: List[Dict[str, Any]],
    monthly_amt_gross: float,
    one_amt_gross: float, one_date: date,
    alloc_mode: str,
    custom_weights: Dict[int, float],
    single_target: Optional[int],
    euro_rate: float,
    fee_pct: float,
) -> Tuple[pd.DataFrame, float, float, float, Optional[float], pd.Timestamp, pd.Timestamp]:
    if not lines:
        return pd.DataFrame(), 0.0, 0.0, 0.0, None, TODAY, TODAY

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


# ------------------------------------------------------------
# Affichage des lignes & tables
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


# ------------------------------------------------------------
# Saisie / construction portefeuilles (page principale)
# ------------------------------------------------------------
def _add_line_form(port_key: str, label: str):
    st.markdown(f"**{label}**")
    with st.form(key=f"form_add_{port_key}", clear_on_submit=False):
        c1, c2 = st.columns([3, 2])
        with c1:
            name = st.text_input("Nom du fonds (ou EUROFUND)", value="", key=f"name_{port_key}")
            isin = st.text_input("ISIN (ou EUROFUND)", value="", key=f"isin_{port_key}")
        with c2:
            amount = st.text_input("Montant investi (brut) €", value="", key=f"amt_{port_key}")
            buy_date = st.date_input("Date d’achat", value=pd.Timestamp("2024-01-02").date(), key=f"date_{port_key}")
        px = st.text_input("Prix d’achat (optionnel)", value="", key=f"px_{port_key}")
        note = st.text_input("Note (optionnel)", value="", key=f"note_{port_key}")
        add_btn = st.form_submit_button("➕ Ajouter")
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
            "sym_used": ""
        }
        st.session_state[port_key].append(ln)
        st.success("Ligne ajoutée.")


def _paste_table_block(port_key: str):
    st.markdown("**Coller un tableau (Nom;ISIN;Montant)**")
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


def _add_from_reco_block(port_key: str):
    st.markdown("**Ajouter un fonds recommandé**")
    cat = st.selectbox(
        "Catégorie",
        ["Core (référence)", "Défensif"],
        key=f"reco_cat_{port_key}"
    )
    fonds_list = RECO_FUNDS_CORE if "Core" in cat else RECO_FUNDS_DEF
    options = [f"{nm} ({isin})" for nm, isin in fonds_list]
    choice = st.selectbox("Fonds", options, key=f"reco_choice_{port_key}")
    idx = options.index(choice) if choice in options else 0
    name, isin = fonds_list[idx]

    c1, c2 = st.columns([2, 2])
    with c1:
        amount = st.text_input("Montant investi (brut) €", value="", key=f"reco_amt_{port_key}")
    with c2:
        buy_date = st.date_input("Date d’achat", value=pd.Timestamp("2024-01-02").date(), key=f"reco_date_{port_key}")
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
            "sym_used": ""
        }
        st.session_state[port_key].append(ln)
        st.success("Fonds recommandé ajouté.")


def _add_external_isin_block(port_key: str):
    st.markdown("**Ajouter un fonds externe par ISIN**")
    isin = st.text_input("ISIN du fonds externe", key=f"ext_isin_{port_key}")
    if st.button("🔎 Chercher", key=f"ext_search_{port_key}"):
        res = eodhd_search(isin)
        match = None
        for it in res:
            if it.get("ISIN") == isin:
                match = it
                break
        if match is None and res:
            match = res[0]
        if match:
            st.session_state[f"ext_found_{port_key}"] = {
                "name": match.get("Name", isin),
                "isin": isin
            }
        else:
            st.warning("Aucun fonds trouvé pour cet ISIN.")
            st.session_state[f"ext_found_{port_key}"] = None

    data = st.session_state.get(f"ext_found_{port_key}")
    if data:
        st.caption(f"Fonds trouvé : **{data['name']}** (`{data['isin']}`)")
        with st.form(key=f"form_ext_{port_key}", clear_on_submit=False):
            c1, c2 = st.columns([2, 2])
            with c1:
                amount = st.text_input("Montant investi (brut) €", value="")
            with c2:
                buy_date = st.date_input("Date d’achat", value=pd.Timestamp("2024-01-02").date())
            px = st.text_input("Prix d’achat (optionnel)", value="")
            ok = st.form_submit_button("➕ Ajouter ce fonds externe")
        if ok:
            try:
                amt = float(str(amount).replace(" ", "").replace(",", "."))
                assert amt > 0
            except Exception:
                st.warning("Montant invalide.")
                return
            ln = {
                "name": data["name"],
                "isin": data["isin"],
                "amount_gross": float(amt),
                "buy_date": pd.Timestamp(buy_date),
                "buy_px": float(str(px).replace(",", ".")) if px.strip() else "",
                "note": "",
                "sym_used": ""
            }
            st.session_state[port_key].append(ln)
            st.success("Fonds externe ajouté.")


# ------------------------------------------------------------
# Layout principal
# ------------------------------------------------------------
st.set_page_config(page_title=APP_TITLE, layout="wide")
st.title(APP_TITLE)

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

with st.sidebar:
    st.header("Fonds en euros")
    EURO_RATE = st.number_input(
        "Taux annuel du fonds euros (%)",
        0.0, 10.0, st.session_state.get("EURO_RATE_PREVIEW", 2.0),
        0.10, key="EURO_RATE_PREVIEW"
    )
    st.caption("Utiliser le symbole EUROFUND pour ce support.")

    st.header("Frais d’entrée (%)")
    FEE_A = st.number_input(
        "Portefeuille 1 (Client)",
        0.0, 10.0, st.session_state.get("FEE_A", 3.0),
        0.10, key="FEE_A"
    )
    FEE_B = st.number_input(
        "Portefeuille 2 (Vous)",
        0.0, 10.0, st.session_state.get("FEE_B", 2.0),
        0.10, key="FEE_B"
    )

    st.header("Versements")
    with st.expander("Portefeuille 1 — Client"):
        M_A = st.number_input("Mensuel brut (€)", 0.0, 1_000_000.0, st.session_state.get("M_A", 0.0), 100.0, key="M_A")
        ONE_A = st.number_input("Ponctuel brut (€)", 0.0, 1_000_000.0, st.session_state.get("ONE_A", 0.0), 100.0, key="ONE_A")
        ONE_A_DATE = st.date_input("Date du ponctuel", value=st.session_state.get("ONE_A_DATE", pd.Timestamp("2024-07-01").date()), key="ONE_A_DATE")
    with st.expander("Portefeuille 2 — Vous"):
        M_B = st.number_input("Mensuel brut (€)", 0.0, 1_000_000.0, st.session_state.get("M_B", 0.0), 100.0, key="M_B")
        ONE_B = st.number_input("Ponctuel brut (€)", 0.0, 1_000_000.0, st.session_state.get("ONE_B", 0.0), 100.0, key="ONE_B")
        ONE_B_DATE = st.date_input("Date du ponctuel", value=st.session_state.get("ONE_B_DATE", pd.Timestamp("2024-07-01").date()), key="ONE_B_DATE")

    st.header("Répartition des versements")
    st.selectbox(
        "Mode",
        ["equal", "custom", "single"],
        index=["equal", "custom", "single"].index(st.session_state.get("ALLOC_MODE", "equal")),
        key="ALLOC_MODE",
        help="equal = parts égales • single = 100% sur une ligne (par défaut la première)."
    )

    st.header("Ajouter un fonds externe (optionnel)")
    _add_external_isin_block("A_lines")
    _add_external_isin_block("B_lines")

st.subheader("Construction des portefeuilles")

colA, colB = st.columns(2)

with colA:
    st.markdown("## Portefeuille 1 — Client")
    with st.expander("Ajouter des fonds recommandés"):
        _add_from_reco_block("A_lines")
    with st.expander("Saisie libre"):
        _add_line_form("A_lines", "Ajouter un fonds (client)")
    with st.expander("Coller un tableau"):
        _paste_table_block("A_lines")
    st.markdown("#### Lignes du portefeuille client")
    for i, ln in enumerate(st.session_state["A_lines"]):
        _line_card(ln, i, "A_lines")

with colB:
    st.markdown("## Portefeuille 2 — Vous")
    with st.expander("Ajouter des fonds recommandés"):
        _add_from_reco_block("B_lines")
    with st.expander("Saisie libre"):
        _add_line_form("B_lines", "Ajouter un fonds (vous)")
    with st.expander("Coller un tableau"):
        _paste_table_block("B_lines")
    st.markdown("#### Lignes du portefeuille vous")
    for i, ln in enumerate(st.session_state["B_lines"]):
        _line_card(ln, i, "B_lines")

custom_weights_A = {id(ln): 1.0 for ln in st.session_state.get("A_lines", [])}
custom_weights_B = {id(ln): 1.0 for ln in st.session_state.get("B_lines", [])}
single_target_A = id(st.session_state["A_lines"][0]) if st.session_state["A_lines"] else None
single_target_B = id(st.session_state["B_lines"][0]) if st.session_state["B_lines"] else None

dfA, brutA, netA, valA, xirrA, startA, fullA = simulate_portfolio(
    st.session_state.get("A_lines", []),
    st.session_state.get("M_A", 0.0),
    st.session_state.get("ONE_A", 0.0),
    st.session_state.get("ONE_A_DATE", pd.Timestamp("2024-07-01").date()),
    st.session_state.get("ALLOC_MODE", "equal"),
    custom_weights_A,
    single_target_A,
    st.session_state.get("EURO_RATE_PREVIEW", 2.0),
    st.session_state.get("FEE_A", 0.0)
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
    st.session_state.get("FEE_B", 0.0)
)

st.subheader("Comparateur de portefeuilles")

dates_for_min = [d for d in [startA, startB] if isinstance(d, pd.Timestamp)]
start_plot = min(dates_for_min) if dates_for_min else TODAY
idx = pd.bdate_range(start=start_plot, end=TODAY, freq="B")
chart_df = pd.DataFrame(index=idx)
if not dfA.empty:
    chart_df["Client"] = dfA.reindex(idx)["Valeur"].ffill()
if not dfB.empty:
    chart_df["Vous"] = dfB.reindex(idx)["Valeur"].ffill()
chart_df = chart_df.reset_index().rename(columns={"index": "Date"})
chart_df = chart_df.melt("Date", var_name="variable", value_name="Valeur (€)")

if chart_df.dropna().empty:
    st.info("Ajoutez des lignes dans les portefeuilles pour afficher le graphique.")
else:
    base = alt.Chart(chart_df).mark_line().encode(
        x=alt.X("Date:T", title="Date"),
        y=alt.Y("Valeur (€):Q", title="Valeur (€)"),
        color="variable:N"
    ).properties(height=360, use_container_width=True)
    st.altair_chart(base, use_container_width=True)

st.subheader("Synthèse chiffrée")
c1, c2, c3, c4, c5, c6 = st.columns(6)
with c1:
    st.metric("Investi BRUT (Client)", to_eur(brutA))
with c2:
    st.metric("Net investi (Client)", to_eur(netA))
with c3:
    st.metric("XIRR (Client)", f"{xirrA:.2f}%" if xirrA is not None else "—")
with c4:
    st.metric("Investi BRUT (Vous)", to_eur(brutB))
with c5:
    st.metric("Net investi (Vous)", to_eur(netB))
with c6:
    st.metric("XIRR (Vous)", f"{xirrB:.2f}%" if xirrB is not None else "—")

st.markdown("### Et si vous aviez investi avec nous ?")
gain_vs_client = (valB - valA) if (valA and valB) else 0.0
delta_xirr = (xirrB - xirrA) if (xirrA is not None and xirrB is not None) else None
st.success(
    f"Vous auriez gagné {to_eur(gain_vs_client)} de plus."
    + (f" Soit {delta_xirr:+.2f}% de performance annualisée supplémentaire." if delta_xirr is not None else "")
)

st.markdown("## Analyse de la diversification (données EODHD Fundamentals)")

EURO_RATE_CUR = st.session_state.get("EURO_RATE_PREVIEW", 2.0)
FEE_A_CUR = st.session_state.get("FEE_A", 0.0)
FEE_B_CUR = st.session_state.get("FEE_B", 0.0)

for label, port_key, fee_pct in [
    ("Portefeuille 1 — Client", "A_lines", FEE_A_CUR),
    ("Portefeuille 2 — Vous", "B_lines", FEE_B_CUR),
]:
    lines = st.session_state.get(port_key, [])
    if not lines:
        continue

    st.markdown(f"### {label}")
    agg = aggregate_portfolio_breakdowns(lines, EURO_RATE_CUR, fee_pct)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**Répartition par classe d'actifs**")
        _pie_chart_from_dict("", agg.get("asset_allocation", {}) or {})
    with col2:
        st.markdown("**Répartition géographique**")
        _pie_chart_from_dict("", agg.get("regions", {}) or {})
    with col3:
        st.markdown("**Répartition sectorielle**")
        _pie_chart_from_dict("", agg.get("sectors", {}) or {})

    st.markdown("**Top 10 lignes sous-jacentes du portefeuille**")
    _bar_chart_top_holdings("", agg.get("top10", []) or [])

positions_table("Portefeuille 1 — Client", "A_lines")
positions_table("Portefeuille 2 — Vous", "B_lines")

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
