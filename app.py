# =========================================
# 08) PORTEFEUILLE DE RÉFÉRENCE (poids, date d'investissement, VLP)
#     - Universe Generali (core + défensifs)
#     - Poids (0% possible)
#     - Date d’investissement initial
#     - VLP mensuels optionnels (montant, période)
#     - Perf & XIRR basés sur VL (EODHD)
# =========================================

import numpy as np
import pandas as pd
import plotly.express as px
from datetime import date, datetime
import calendar

# --- Universe Generali (core + défensifs)
UNIVERSE_GENERALI = [
    {"name": "R-co Valor C EUR", "isin": "FR0011253624", "type": "Actions Monde"},
    {"name": "Vivalor International", "isin": "FR0014001LS1", "type": "Actions Monde"},
    {"name": "CARMIGNAC Investissement A EUR", "isin": "FR0010148981", "type": "Actions Monde"},
    {"name": "FIDELITY Funds - World Fund", "isin": "LU0069449576", "type": "Actions Monde"},
    {"name": "CLARTAN Valeurs C", "isin": "LU1100076550", "type": "Actions Europe"},
    {"name": "CARMIGNAC Patrimoine", "isin": "FR0010135103", "type": "Diversifié patrimonial"},
    {"name": "SYCOYIELD 2030 RC", "isin": "FR001400MCQ6", "type": "Obligataire échéance"},
    {"name": "R-Co Target 2029 HY", "isin": "FR001400AWH8", "type": "Obligataire haut rendement"},
    {"name": "Fonds en euros Generali", "isin": None, "type": "Fonds Euro"},
]
DF_UNI = pd.DataFrame(UNIVERSE_GENERALI)

st.header("🟣 Portefeuille de référence (Generali Espace Invest 5)")

# --- Table des poids (éditable)
_w_init = pd.DataFrame(
    [{"Fonds": r["name"], "ISIN": r["isin"], "Type": r["type"], "Poids %": 0.0} for r in UNIVERSE_GENERALI]
)
st.markdown("Saisis les **poids (%)** par fonds (0% possible). La somme sera **normalisée** automatiquement.")
df_weights = st.data_editor(_w_init, use_container_width=True, num_rows="fixed")
df_weights["Poids %"] = pd.to_numeric(df_weights["Poids %"], errors="coerce").fillna(0.0)

# --- Paramètres d'investissement
colA, colB, colC = st.columns(3)
with colA:
    initial_date = st.date_input("Date d’investissement initial", value=date(2024, 1, 2))
with colB:
    initial_amount = st.number_input("Montant initial (€)", min_value=0.0, value=10000.0, step=500.0, format="%.2f")
with colC:
    euro_rate = st.number_input("Rendement annuel du Fonds € (si inclus) (%)", min_value=0.0, value=2.00, step=0.10, format="%.2f")

st.markdown("### Versements Libres Programmés (VLP) — *optionnels*")
col1, col2, col3, col4 = st.columns(4)
with col1:
    vlp_enable = st.checkbox("Activer VLP mensuels", value=False)
with col2:
    vlp_amount = st.number_input("Montant VLP mensuel (€)", min_value=0.0, value=300.0, step=50.0, format="%.2f", disabled=not vlp_enable)
with col3:
    vlp_start = st.date_input("Début VLP", value=initial_date, disabled=not vlp_enable)
with col4:
    vlp_end = st.date_input("Fin VLP", value=date.today(), disabled=not vlp_enable)

run_ref = st.button("🚀 Calculer la performance du portefeuille de référence", type="primary")

# --- Utils dates / contributions
def month_end(dt: date) -> date:
    last_day = calendar.monthrange(dt.year, dt.month)[1]
    return date(dt.year, dt.month, last_day)

def monthly_schedule(start: date, end: date) -> list[date]:
    if end < start:
        return []
    d = date(start.year, start.month, min(start.day, 28))
    out = []
    while d <= end:
        out.append(month_end(d))
        # incrément mois +1
        y = d.year + (d.month // 12)
        m = 1 if d.month == 12 else d.month + 1
        d = date(y, m, min(d.day, 28))
    return out

def to_ts(d: date) -> pd.Timestamp:
    return pd.Timestamp(d)

# --- Série synthétique pour Fonds Euro (croissance lissée à euro_rate)
@st.cache_data(ttl=6*3600, show_spinner=False)
def euro_fund_series(start_dt: date, end_dt: date, annual_rate: float) -> pd.Series:
    idx = pd.bdate_range(start=to_ts(start_dt), end=to_ts(end_dt), freq="B")
    if len(idx) == 0:
        return pd.Series(dtype=float)
    daily_rate = (1 + annual_rate/100.0) ** (1/252) - 1
    vals = (1 + daily_rate) ** np.arange(len(idx))
    s = pd.Series(vals, index=idx, name="Fonds € (synthetic)")
    # Échelle en "prix": base 100 au départ pour cohérence (sera ramené via prix)
    return s * 100.0

# --- Prix EODHD d'un fonds (via ISIN ou nom)
@st.cache_data(ttl=6*3600, show_spinner=False)
def load_price_series_generic(isin: str | None, name: str, from_dt: date) -> pd.Series:
    if isin:
        sym = resolve_symbol(isin)
    else:
        # Fonds euro (pas d’ISIN) -> série synthétique (ajustée ensuite)
        sym = None

    if sym:
        dfp = eod_prices(sym, from_dt.strftime("%Y-%m-%d"))
        if not dfp.empty:
            s = dfp["Close"].copy()
            s.name = name
            return s
    # fallback fonds euro synthétique (si name == fonds euro)
    if isin is None and "Fonds en euros" in name:
        return euro_fund_series(from_dt, date.today(), euro_rate).rename(name)
    return pd.Series(dtype=float)

# --- Simulation achats (lump sum + VLP) -> parts, valeur dans le temps
def simulate_reference_portfolio(dfw: pd.DataFrame,
                                 initial_dt: date,
                                 initial_amt: float,
                                 vlp_on: bool,
                                 vlp_amt: float,
                                 vlp_dt_start: date,
                                 vlp_dt_end: date) -> dict:
    # 1) Nettoyage & normalisation des poids
    dfw = dfw.copy()
    dfw["Poids %"] = pd.to_numeric(dfw["Poids %"], errors="coerce").fillna(0.0)
    total = dfw["Poids %"].sum()
    if total <= 0:
        st.error("Tous les poids sont à 0%. Mets au moins un poids > 0.")
        return {}
    dfw["poids_n"] = dfw["Poids %"] / total  # normalisation 100% même si saisie != 100

    # 2) Charger les séries de prix (depuis initial_dt)
    series_map: dict[str, pd.Series] = {}
    not_found = []
    for _, r in dfw.iterrows():
        if r["poids_n"] <= 0: 
            continue
        s = load_price_series_generic(r["ISIN"], r["Fonds"], initial_dt)
        if s.empty:
            not_found.append(r["Fonds"])
        else:
            series_map[r["Fonds"]] = s

    if not_found:
        st.warning("Données manquantes pour: " + ", ".join(not_found))
    if not series_map:
        st.error("Aucune VL disponible pour les fonds sélectionnés.")
        return {}

    # 3) Index commun (toutes les dates ouvrées observées)
    prices = pd.concat(series_map, axis=1).sort_index().ffill()
    # Restreindre à [initial_dt, today]
    prices = prices.loc[prices.index >= to_ts(initial_dt)]
    if prices.empty:
        st.error("Pas de données de VL après la date d’investissement.")
        return {}

    # 4) Calendrier des contributions
    cashflows: list[tuple[pd.Timestamp, float]] = []
    contribs: list[tuple[pd.Timestamp, float]] = []
    # - Lump sum initial (date exacte, si non-trading -> on prendra la 1ère VL >= date)
    cashflows.append((to_ts(initial_dt), -float(initial_amt)))
    contribs.append((to_ts(initial_dt), float(initial_amt)))

    if vlp_on and vlp_amt > 0:
        for d in monthly_schedule(vlp_dt_start, vlp_dt_end):
            ts = to_ts(d)
            if ts < prices.index.min():
                continue
            cashflows.append((ts, -float(vlp_amt)))
            contribs.append((ts, float(vlp_amt)))

    # 5) Simulation par parts
    # Parts par fonds
    parts = {f: 0.0 for f in prices.columns}
    # Historique valeur portefeuille
    port_values = []

    # fonction prix à une date: dernière VL <= d sinon 1ère >= d
    def px_on(fund: str, dts: pd.Timestamp) -> float | None:
        s = prices[fund]
        if dts in s.index:
            v = s.loc[dts]
            return float(v) if pd.notna(v) else None
        bef = s.loc[s.index <= dts]
        if not bef.empty:
            return float(bef.iloc[-1])
        aft = s.loc[s.index >= dts]
        if not aft.empty:
            return float(aft.iloc[0])
        return None

    # Achats aux dates de contribution
    for ts, amt in contribs:
        # Répartition selon poids_n AUJOURD’HUI (statique)
        for _, r in dfw.iterrows():
            if r["poids_n"] <= 0:
                continue
            fund = r["Fonds"]
            if fund not in prices.columns:
                continue
            px = px_on(fund, ts)
            if (px is None) or (px <= 0):
                continue
            parts[fund] += (amt * r["poids_n"]) / px

    # Valeur quotidienne (sur index des VL)
    for dts in prices.index:
        val = 0.0
        for f in prices.columns:
            px = prices.loc[dts, f]
            if pd.isna(px): 
                continue
            val += parts.get(f, 0.0) * float(px)
        port_values.append((dts, val))

    df_val = pd.DataFrame(port_values, columns=["Date", "Valeur"]).set_index("Date")
    total_investi = -sum(cf for _, cf in cashflows if cf < 0)
    final_val = float(df_val["Valeur"].iloc[-1])

    # XIRR (annualisé)
    cf_for_xirr = cashflows + [(df_val.index[-1], final_val)]
    irr = xirr(cf_for_xirr)

    out = {
        "prices": prices,
        "values": df_val,
        "investi": total_investi,
        "final": final_val,
        "pl": final_val - total_investi,
        "perf_pct": (final_val / total_investi - 1.0) * 100.0 if total_investi > 0 else np.nan,
        "xirr_pct": (irr * 100.0) if irr is not None else np.nan,
        "parts": parts,
        "cashflows": cashflows,
    }
    return out

if run_ref:
    res = simulate_reference_portfolio(
        df_weights, initial_date, initial_amount,
        vlp_enable, vlp_amount, vlp_start, vlp_end
    )
    if res:
        st.subheader("📈 Performance du portefeuille de référence")
        # courbe valeur
        fig = px.line(res["values"].reset_index(), x="Date", y="Valeur", title="Valeur du portefeuille (VL agrégées)")
        st.plotly_chart(fig, use_container_width=True)

        # métriques
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Investi total", to_eur(res["investi"]))
        c2.metric("Valeur actuelle", to_eur(res["final"]))
        c3.metric("P&L", to_eur(res["pl"]))
        c4.metric("Perf depuis origine", f"{res['perf_pct']:.2f}%")

        st.markdown(f"**XIRR (annualisé)** : {'' if pd.isna(res['xirr_pct']) else f'{res['xirr_pct']:.2f}%'}")

        # tableau parts + dernier prix
        last_date = res["values"].index[-1]
        last_px = res["prices"].loc[last_date]
        tbl = []
        for f in res["prices"].columns:
            qty = res["parts"].get(f, 0.0)
            px = float(last_px.get(f, np.nan))
            val = qty * px if pd.notna(px) else np.nan
            tbl.append({"Fonds": f, "Parts détenues": qty, "Dernier prix": px, "Valeur €": val})
        df_pos = pd.DataFrame(tbl)
        st.subheader("📄 Détail positions (estimées)")
        st.dataframe(
            df_pos.style.format({"Parts détenues": "{:.4f}", "Dernier prix": to_eur, "Valeur €": to_eur}, na_rep=""),
            use_container_width=True, hide_index=True
        )

        # base 100 comparée (portefeuille vs somme pondérée prix)
        base = res["values"].copy()
        base100 = base / base.iloc[0] * 100.0
        st.subheader("📊 Performance base 100")
        st.line_chart(base100.rename(columns={"Valeur": "Portefeuille (base 100)"}))
else:
    st.info("Remplis les poids, la date d’investissement (et VLP si besoin), puis clique **Calculer**.")
