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

FRAIS_ENTREE_DEFAULT = 0.0  # en % du montant investi

# Exemple d’univers de fonds (à adapter à ta base réelle)
UNIVERSE_FUNDS = {
    "R-co VALOR": {
        "isin": "FR0010588343",
        "name": "R-Co Valor C EUR",
        "type": "actions monde",
    },
    "Vivalor International": {
        "isin": "FR0010621910",
        "name": "Vivalor International",
        "type": "actions internationales",
    },
    "CARMIGNAC Investissement": {
        "isin": "FR0010148981",
        "name": "Carmignac Investissement A EUR Acc",
        "type": "actions monde",
    },
    "Fidelity World": {
        "isin": "LU0261953909",
        "name": "Fidelity Funds - World Fund A-Acc-EUR",
        "type": "actions monde",
    },
    "Clartan Valeurs": {
        "isin": "FR0010312921",
        "name": "Clartan Valeurs C",
        "type": "actions Europe / Monde",
    },
    "CARMIGNAC Patrimoine": {
        "isin": "FR0010135103",
        "name": "Carmignac Patrimoine A EUR Acc",
        "type": "mixte patrimonial",
    },
}

# ------------------------------------------------------------
# Paramètres EODHD
# ------------------------------------------------------------
EODHD_BASE_URL = "https://eodhd.com/api"
EODHD_API_TOKEN = st.secrets.get("EODHD_API_TOKEN", "")

if not EODHD_API_TOKEN:
    st.warning(
        "⚠️ Clé API EODHD manquante dans `st.secrets`. "
        "Ajoute `EODHD_API_TOKEN` dans ton fichier de configuration."
    )

# ------------------------------------------------------------
# Helpers généraux
# ------------------------------------------------------------


def to_eur(x: float | int | None) -> str:
    if x is None:
        return "—"
    return f"{x:,.0f} €".replace(",", " ").replace(".", ",")


def to_pct(x: float | int | None, decimals: int = 2) -> str:
    if x is None:
        return "—"
    return f"{x:.{decimals}f} %".replace(".", ",")


def safe_float(x: Any) -> Optional[float]:
    try:
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return None
        return float(x)
    except Exception:
        return None


# ------------------------------------------------------------
# EODHD : fonction générique d'appel API
# ------------------------------------------------------------

@st.cache_data(show_spinner=False, ttl=3600)
def eodhd_get(path: str, params: Optional[Dict[str, Any]] = None) -> Any:
    if not EODHD_API_TOKEN:
        return None
    if params is None:
        params = {}
    params = dict(params)
    params["api_token"] = EODHD_API_TOKEN
    params["fmt"] = "json"

    url = f"{EODHD_BASE_URL}{path}"
    resp = requests.get(url, params=params, timeout=20)
    resp.raise_for_status()
    try:
        return resp.json()
    except Exception:
        return None


# ------------------------------------------------------------
# EODHD Search : trouver un symbole à partir d'un nom ou d'un ISIN
# ------------------------------------------------------------

@st.cache_data(show_spinner=False, ttl=3600)
def eodhd_search(q: str) -> List[Dict[str, Any]]:
    if not q:
        return []
    res = eodhd_get("/search/" + q, params={"limit": 50})
    if isinstance(res, list):
        return res
    return []


# ------------------------------------------------------------
# EODHD Prices : série quotidienne (avec adjusted_close)
# ------------------------------------------------------------

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

    # Cas dict -> soit { "Stocks": {"Fund_%": 90}, ... } soit { "Stocks": 90, ... }
    if isinstance(raw, dict):
        for k, v in raw.items():
            label = str(k)
            val = None
            if isinstance(v, dict):
                # Noms possibles des champs de pourcentage
                for key_pct in ("Fund_%", "Assets_%", "Net_Assets_%", "Equity_%", "Weight", "Percent"):
                    if key_pct in v:
                        val = v[key_pct]
                        break
            else:
                # Valeur directe : {"Stocks": 90}
                val = v
            try:
                if val is not None:
                    out[label] = float(val)
            except Exception:
                continue

    # Cas liste -> chaque élément est normalement un dict
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

    # On enlève les zéros / négatifs
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

    Logique :
    - on essaie plusieurs symboles pour les fundamentals :
        1) le symbole utilisé pour les prix (ex. FR0011253624.EUFUND)
        2) la racine sans suffixe (ex. FR0011253624)
        3) les codes retournés par /search/{base}?type=fund
    - on ne retient un symbole que si au moins un des blocs de breakdown est présent.
    """

    def _try_symbol(sym: str) -> Optional[Dict[str, Any]]:
        """
        Teste un symbole sur /fundamentals.
        On ne renvoie le JSON que si on trouve au moins un bloc de breakdown.
        Sinon -> None (pour laisser la main au candidat suivant).
        """
        if not sym:
            return None
        js = eodhd_fundamentals(sym)
        if not js:
            return None

        # Si EODHD renvoie explicitement une erreur, on la remonte
        err_msg = js.get("error") or js.get("Error")
        if err_msg:
            st.caption(f"⚠️ EODHD Fundamentals indisponibles pour **{sym}** : {err_msg}")
            return None

        # Est-ce qu'on voit au moins une des clés "structurantes" dans le JSON ?
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
            # Pas de breakdown pour ce symbole -> on laisse la boucle essayer un autre candidat
            return None

        return js

    # 1) liste des symboles à tester
    candidates: List[str] = []
    base = (symbol or "").strip()
    if base:
        candidates.append(base)
        if "." in base:
            # ex. FR0011253624.EUFUND -> FR0011253624
            root = base.split(".", 1)[0]
            candidates.append(root)

        # On ajoute les symboles issus de la recherche EODHD, en filtrant sur les fonds
        try:
            search_res = eodhd_get(f"/search/{base}", params={"type": "fund"})
            if isinstance(search_res, list):
                for it in search_res:
                    code = it.get("Code")
                    exch = it.get("Exchange")
                    if code and exch:
                        candidates.append(f"{code}.{exch}")
                    elif code:
                        candidates.append(code)
        except Exception:
            pass

    # on dédoublonne en respectant l'ordre
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
        # Aucun symbole n'a donné de fondamentaux exploitables
        return {
            "asset_allocation": {},
            "regions": {},
            "sectors": {},
            "top10": [],
        }

    # On cherche les blocs dans tout le JSON
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

    # Normalisation douce des pourcentages (on garde les proportions)
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
    Ajoute dans 'target' le breakdown 'src' pondéré par w_line.
    """
    for k, v in src.items():
        if v is None:
            continue
        target[k] = target.get(k, 0.0) + w_line * (v / 100.0)


def aggregate_portfolio_breakdowns(
    lines: List[Dict[str, Any]],
    euro_rate: float,
) -> Dict[str, Any]:
    """
    Agrège les breakdowns de chaque ligne de portefeuille (si disponible via EODHD)
    en un breakdown global pondéré par le poids de chaque ligne.
    """
    alloc_tot: Dict[str, float] = {}
    regions_tot: Dict[str, float] = {}
    sectors_tot: Dict[str, float] = {}
    top10_tot: Dict[str, float] = {}

    # On commence par calculer la valeur de chaque ligne
    total_val = 0.0
    line_values: List[float] = []

    for ln in lines:
        amt = safe_float(ln.get("amount"))
        px = safe_float(ln.get("last_price"))
        if amt is None or px is None or px <= 0:
            line_values.append(0.0)
            continue
        val = amt * px
        total_val += val
        line_values.append(val)

    if total_val <= 0:
        return {
            "asset_allocation": {},
            "regions": {},
            "sectors": {},
            "top10": [],
        }

    for ln, val_line in zip(lines, line_values):
        if val_line <= 0:
            continue
        w_line = val_line / total_val

        sym_used = ln.get("sym_used", "")
        if not sym_used:
            # On tente de retrouver un symbole via l'ISIN ou le nom
            isin_or_name = ln.get("isin") or ln.get("name") or ""
            df_tmp, sym, _dbg = get_price_series(isin_or_name, None, euro_rate)
            if sym:
                sym_used = sym
                ln["sym_used"] = sym
        if not sym_used:
            continue

        bkd = get_fund_breakdowns(sym_used)
        alloc = bkd.get("asset_allocation", {}) or {}
        regions = bkd.get("regions", {}) or {}
        sectors = bkd.get("sectors", {}) or {}
        top10 = bkd.get("top10", []) or []

        _accumulate_weighted(alloc_tot, alloc, w_line)
        _accumulate_weighted(regions_tot, regions, w_line)
        _accumulate_weighted(sectors_tot, sectors, w_line)

        # Top10 agrégé
        for h in top10:
            nm = h.get("name")
            wt = safe_float(h.get("weight"))
            if not nm or wt is None:
                continue
            try:
                wt = float(wt)
            except Exception:
                continue
            top10_tot[nm] = top10_tot.get(nm, 0.0) + w_line * (wt / 100.0)

    # 3) Remise en pourcentage pour lisibilité
    def _as_percent(d: Dict[str, float]) -> Dict[str, float]:
        s = sum(d.values())
        if s <= 0:
            return {}
        return {k: 100.0 * v / s for k, v in d.items()}

    alloc_tot = _as_percent(alloc_tot)
    regions_tot = _as_percent(regions_tot)
    sectors_tot = _as_percent(sectors_tot)

    # Top 10 lignes sous-jacentes les plus importantes
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

# ------------------------------------------------------------
# Helpers pour récupérer les prix et symboles
# ------------------------------------------------------------

def _symbol_candidates(isin_or_name: str) -> List[str]:
    val = str(isin_or_name).strip()
    if not val:
        return []

    # On tente quelques variations classiques
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

    # Déduplication
    seen = set()
    out: List[str] = []
    for c in candidates:
        if c and c not in seen:
            seen.add(c)
            out.append(c)
    return out


@st.cache_data(show_spinner=False, ttl=3600)
def get_price_series(
    isin_or_name: str,
    start: Optional[pd.Timestamp],
    euro_rate: float,
) -> Tuple[pd.DataFrame, str, str]:
    """
    Retourne (df, symbole_retenu, debug_json)
    df contient une colonne 'Close' indexée par date.
    """
    val = str(isin_or_name).strip()
    debug = {"input": val}
    if not val:
        return pd.DataFrame(), "", json.dumps(debug)

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
# XIRR / TRI
# ------------------------------------------------------------

def xnpv(rate: float, cash_flows: List[Tuple[pd.Timestamp, float]]) -> float:
    t0 = cash_flows[0][0]
    return sum(
        cf / (1.0 + rate) ** ((t - t0).days / 365.0)
        for (t, cf) in cash_flows
    )


def xirr(cash_flows: List[Tuple[pd.Timestamp, float]], guess: float = 0.1) -> Optional[float]:
    if not cash_flows:
        return None
    try:
        rate = guess
        for _ in range(100):
            f = xnpv(rate, cash_flows)
            eps = 1e-7
            f_der = (xnpv(rate + eps, cash_flows) - f) / eps
            if abs(f_der) < 1e-12:
                break
            new_rate = rate - f / f_der
            if abs(new_rate - rate) < 1e-7:
                rate = new_rate
                break
            rate = new_rate
        return 100.0 * rate
    except Exception:
        return None

# ------------------------------------------------------------
# UI : Configuration de la page
# ------------------------------------------------------------

st.set_page_config(
    page_title="Simulation de portefeuilles Valority",
    layout="wide",
)

st.title("Simulation de portefeuilles Valority")

st.markdown(
    """
Cette application te permet de comparer **l'allocation actuelle** d'un client
avec une **allocation recommandée Valority**, en simulant les performances
historiques et la diversification.
"""
)

# ------------------------------------------------------------
# Saisie des paramètres globaux
# ------------------------------------------------------------

col1, col2, col3 = st.columns(3)

with col1:
    capital_initial = st.number_input(
        "Capital initial investi (€)",
        min_value=0.0,
        value=50000.0,
        step=1000.0,
        format="%.0f",
    )

with col2:
    frais_entree_pct = st.number_input(
        "Frais d'entrée (%)",
        min_value=0.0,
        max_value=5.0,
        value=FRAIS_ENTREE_DEFAULT,
        step=0.10,
        format="%.2f",
    )

with col3:
    euro_rate = st.number_input(
        "Taux de change EUR (pour conversion éventuelle)",
        min_value=0.5,
        max_value=2.0,
        value=1.0,
        step=0.01,
        format="%.2f",
        help="Si tous les fonds sont en EUR, laisse 1,00.",
    )

# ------------------------------------------------------------
# Tables d'allocation Client / Valority
# ------------------------------------------------------------

st.markdown("## 1️⃣ Allocation actuelle du client")

st.markdown(
    """
Renseigne les lignes de l'allocation actuelle du client.  
Tu peux saisir un **ISIN** ou le **nom du fonds**.
"""
)

if "A_lines" not in st.session_state:
    st.session_state["A_lines"] = []

if "B_lines" not in st.session_state:
    st.session_state["B_lines"] = []

def _new_line() -> Dict[str, Any]:
    return {
        "isin": "",
        "name": "",
        "amount": 0.0,
        "last_price": None,
        "sym_used": "",
        "debug": "",
    }

def _line_editor(key_prefix: str, lines: List[Dict[str, Any]]) -> None:
    """
    Édition d'une liste de lignes (allocation client ou Valority).
    """
    to_delete = []
    for i, ln in enumerate(lines):
        with st.expander(f"Ligne {i+1}", expanded=True):
            c1, c2, c3, c4 = st.columns([2, 3, 2, 1])
            with c1:
                ln["isin"] = st.text_input("ISIN", value=ln.get("isin", ""), key=f"{key_prefix}_isin_{i}")
            with c2:
                ln["name"] = st.text_input("Nom du fonds", value=ln.get("name", ""), key=f"{key_prefix}_name_{i}")
            with c3:
                ln["amount"] = st.number_input(
                    "Montant investi (€)",
                    min_value=0.0,
                    value=float(ln.get("amount") or 0.0),
                    step=1000.0,
                    format="%.0f",
                    key=f"{key_prefix}_amount_{i}",
                )
            with c4:
                if st.button("Supprimer", key=f"{key_prefix}_del_{i}"):
                    to_delete.append(i)

            # Affichage éventuel d'infos de debug
            if ln.get("debug"):
                with st.expander("Debug symbole EODHD", expanded=False):
                    st.code(ln["debug"], language="json")

    # Suppression des lignes marquées
    for idx in sorted(to_delete, reverse=True):
        if 0 <= idx < len(lines):
            lines.pop(idx)

    # Bouton d'ajout
    if st.button("Ajouter une ligne", key=f"{key_prefix}_add"):
        lines.append(_new_line())


_line_editor("A", st.session_state["A_lines"])

st.markdown("---")
st.markdown("## 2️⃣ Allocation recommandée Valority")

_line_editor("B", st.session_state["B_lines"])

# ------------------------------------------------------------
# Simulation des performances
# ------------------------------------------------------------

st.markdown("## 3️⃣ Simulation historique des portefeuilles")

st.markdown(
    """
On reconstruit l'historique des deux portefeuilles (Client vs Valority) à partir
des prix EODHD des supports saisis, puis on compare les courbes et les TRI (XIRR).
"""
)

# Helpers calculs lignes / portefeuilles

def compute_line_metrics(
    line: Dict[str, Any],
    fee_pct: float,
    euro_rate: float,
) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """
    Calcule, pour une ligne, le montant net investi, le prix d'achat moyen
    (approximatif) et la quantité d'unités.
    """
    amt = safe_float(line.get("amount"))
    if amt is None or amt <= 0:
        return None, None, None

    net_amt = amt * (1.0 - fee_pct / 100.0)
    buy_px = safe_float(line.get("last_price"))
    if buy_px is None or buy_px <= 0:
        isin_or_name = line.get("isin") or line.get("name")
        dfp, sym, dbg = get_price_series(isin_or_name, None, euro_rate)
        line["sym_used"] = sym
        line["debug"] = dbg
        if dfp.empty:
            return net_amt, None, None
        buy_px = float(dfp["Close"].iloc[0])
        line["last_price"] = buy_px

    qty_disp = net_amt / buy_px if buy_px > 0 else None
    return net_amt, buy_px, qty_disp


def build_portfolio_series(
    lines: List[Dict[str, Any]],
    fee_pct: float,
    euro_rate: float,
) -> Tuple[pd.Series, Dict[str, Any]]:
    """
    Construit l'historique de valeur d'un portefeuille.
    Retourne (serie_valeur, infos_debug).
    """
    all_series: List[pd.Series] = []
    debug: Dict[str, Any] = {"lines": []}

    for i, ln in enumerate(lines):
        net_amt, buy_px, qty = compute_line_metrics(ln, fee_pct, euro_rate)
        if net_amt is None or buy_px is None or qty is None:
            continue

        isin_or_name = ln.get("isin") or ln.get("name")
        df, sym, dbg = get_price_series(isin_or_name, None, euro_rate)
        ln["sym_used"] = sym
        ln["debug"] = dbg

        if df.empty:
            continue

        s = df["Close"] * qty
        s.name = f"{sym}"
        all_series.append(s)
        debug["lines"].append(
            {
                "index": i,
                "sym": sym,
                "qty": qty,
                "net_amt": net_amt,
                "first_date": str(df.index.min().date()),
                "last_date": str(df.index.max().date()),
            }
        )

    if not all_series:
        return pd.Series(dtype=float), debug

    # Alignement des séries
    df_all = pd.concat(all_series, axis=1).fillna(method="ffill")
    portfolio = df_all.sum(axis=1)
    portfolio.name = "Valeur portefeuille"
    return portfolio, debug


# Construction des deux séries
serie_A, debug_A = build_portfolio_series(st.session_state["A_lines"], frais_entree_pct, euro_rate)
serie_B, debug_B = build_portfolio_series(st.session_state["B_lines"], frais_entree_pct, euro_rate)

if serie_A.empty and serie_B.empty:
    st.info("Ajoute au moins une ligne dans l'un des portefeuilles pour lancer la simulation.")
else:
    df_plot = pd.DataFrame(
        {
            "date": pd.to_datetime(
                sorted(set(serie_A.index.to_list() + serie_B.index.to_list()))
            )
        }
    ).set_index("date")

    if not serie_A.empty:
        df_plot = df_plot.join(serie_A, how="left")
    if not serie_B.empty:
        df_plot = df_plot.join(serie_B, how="left")

    df_plot = df_plot.fillna(method="ffill").dropna(how="all")

    df_melt = df_plot.reset_index().melt(
        id_vars="date",
        var_name="Portefeuille",
        value_name="Valeur",
    )
    df_melt = df_melt.dropna(subset=["Valeur"])

    chart = (
        alt.Chart(df_melt)
        .mark_line()
        .encode(
            x="date:T",
            y=alt.Y("Valeur:Q", title="Valeur de portefeuille (€)"),
            color="Portefeuille:N",
            tooltip=["date:T", "Portefeuille:N", "Valeur:Q"],
        )
        .properties(width=900, height=400)
    )
    st.altair_chart(chart, use_container_width=True)

    # Calcul TRI (XIRR) sur les cash-flows : investissement initial + valeur finale
    cash_flows_A: List[Tuple[pd.Timestamp, float]] = []
    cash_flows_B: List[Tuple[pd.Timestamp, float]] = []

    if not serie_A.empty:
        first_date_A = serie_A.index.min()
        last_date_A = serie_A.index.max()
        total_invested_A = sum(
            safe_float(ln.get("amount") or 0.0) for ln in st.session_state["A_lines"]
        )
        if total_invested_A and total_invested_A > 0:
            cash_flows_A.append((first_date_A, -total_invested_A))
            cash_flows_A.append((last_date_A, float(serie_A.iloc[-1])))

    if not serie_B.empty:
        first_date_B = serie_B.index.min()
        last_date_B = serie_B.index.max()
        total_invested_B = sum(
            safe_float(ln.get("amount") or 0.0) for ln in st.session_state["B_lines"]
        )
        if total_invested_B and total_invested_B > 0:
            cash_flows_B.append((first_date_B, -total_invested_B))
            cash_flows_B.append((last_date_B, float(serie_B.iloc[-1])))

    xirr_A = xirr(cash_flows_A) if cash_flows_A else None
    xirr_B = xirr(cash_flows_B) if cash_flows_B else None

    colA, colB = st.columns(2)
    with colA:
        st.subheader("Portefeuille actuel du client")
        st.metric(
            "Valeur finale",
            to_eur(float(serie_A.iloc[-1])) if not serie_A.empty else "—",
        )
        st.metric(
            "TRI (XIRR)",
            to_pct(xirr_A) if xirr_A is not None else "—",
        )
    with colB:
        st.subheader("Portefeuille Valority")
        st.metric(
            "Valeur finale",
            to_eur(float(serie_B.iloc[-1])) if not serie_B.empty else "—",
        )
        st.metric(
            "TRI (XIRR)",
            to_pct(xirr_B) if xirr_B is not None else "—",
        )

    if xirr_A is not None and xirr_B is not None:
        delta_xirr = xirr_B - xirr_A
        st.metric(
            "Surperformance annualisée (Δ XIRR)",
            f"{delta_xirr:+.2f}%" if delta_xirr is not None else "—",
        )

    # Synthèse textuelle
    if not serie_A.empty and not serie_B.empty:
        valA = float(serie_A.iloc[-1])
        valB = float(serie_B.iloc[-1])
        gain_vs_client = max(valB - valA, 0.0)
        st.markdown(
            f"""
Aujourd’hui, avec votre allocation actuelle, votre portefeuille vaut **{to_eur(valA)}**.  
Avec l’allocation Valority, il serait autour de **{to_eur(valB)}**, soit environ **{to_eur(gain_vs_client)}** de plus."""
        )

# ------------------------------------------------------------
# Analyse de la diversification (Fundamentals)
# ------------------------------------------------------------

st.markdown("## Analyse de la diversification (données EODHD Fundamentals)")

if st.button("Analyser la diversification des deux portefeuilles"):
    if not st.session_state["A_lines"] and not st.session_state["B_lines"]:
        st.info("Ajoute des lignes dans au moins un portefeuille pour analyser la diversification.")
    else:
        agg_A = aggregate_portfolio_breakdowns(st.session_state["A_lines"], euro_rate)
        agg_B = aggregate_portfolio_breakdowns(st.session_state["B_lines"], euro_rate)

        def _pie_chart_from_dict(title: str, d: Dict[str, float]):
            if not d:
                st.caption(f"{title} : données indisponibles.")
                return
            df = pd.DataFrame(
                {
                    "Label": list(d.keys()),
                    "Poids": list(d.values()),
                }
            )
            chart = (
                alt.Chart(df)
                .mark_arc()
                .encode(
                    theta="Poids:Q",
                    color="Label:N",
                    tooltip=["Label:N", "Poids:Q"],
                )
                .properties(title=title)
            )
            st.altair_chart(chart, use_container_width=True)

        colA1, colB1 = st.columns(2)
        with colA1:
            st.subheader("Portefeuille client")
            _pie_chart_from_dict("Allocation d'actifs", agg_A.get("asset_allocation", {}) or {})
            _pie_chart_from_dict("Régions", agg_A.get("regions", {}) or {})
            _pie_chart_from_dict("Secteurs", agg_A.get("sectors", {}) or {})

        with colB1:
            st.subheader("Portefeuille Valority")
            _pie_chart_from_dict("Allocation d'actifs", agg_B.get("asset_allocation", {}) or {})
            _pie_chart_from_dict("Régions", agg_B.get("regions", {}) or {})
            _pie_chart_from_dict("Secteurs", agg_B.get("sectors", {}) or {})

        st.markdown("### Top 10 lignes sous-jacentes (portefeuille Valority)")

        top10_B = agg_B.get("top10", []) or []
        if not top10_B:
            st.caption("Top 10 indisponible (données non fournies par EODHD).")
        else:
            df_top10 = pd.DataFrame(top10_B)
            df_top10["Poids (%)"] = df_top10["weight"]
            st.dataframe(df_top10[["name", "Poids (%)"]].rename(columns={"name": "Ligne"}))


# ------------------------------------------------------------
# Debug avancé (facultatif)
# ------------------------------------------------------------

with st.expander("🛠 Debug avancé (symboles, séries, fundamentals)", expanded=False):
    st.write("### Debug portefeuille client")
    st.json(debug_A)
    st.write("### Debug portefeuille Valority")
    st.json(debug_B)
