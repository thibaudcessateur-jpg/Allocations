import streamlit as st
import pandas as pd
from dotenv import load_dotenv
from core.config import settings
from core.risk_profiles import RiskProfile, risk_buckets
from core.alloc_engine import build_allocation
from services.eodhd_client import EODHD
from core.schemas import ClientInput

load_dotenv()

st.set_page_config(page_title="Allocation CGP", page_icon="🦉", layout="wide")

st.title("🦉 Allocation CGP — Générateur d'allocations personnalisées")
st.caption("Données marché: EODHD | Usage professionnel privé")

# --- Sidebar — paramètres globaux ---
with st.sidebar:
    st.header("Paramètres")
    api_key_ok = True
    try:
        _ = settings.EODHD_API_KEY
    except Exception:
        api_key_ok = False
    st.write("Statut API:", "✅" if api_key_ok else "⚠️ clé manquante (EODHD_API_KEY)")

# --- Entrées client ---
col1, col2, col3 = st.columns(3)
with col1:
    age = st.number_input("Âge du client", min_value=18, max_value=99, value=40)
    horizon = st.selectbox("Horizon d'investissement", ["<3 ans", "3-5 ans", "5-8 ans", ">8 ans"], index=2)
with col2:
    profil_nom = st.selectbox("Profil de risque", list(risk_buckets.keys()), index=1)
    pea = st.checkbox("PEA (éligibilité titres)")
with col3:
    montant = st.number_input("Montant à allouer (€)", min_value=1000, step=1000, value=50000)
    contraintes_sp = st.multiselect(
        "Contraintes spécifiques",
        ["Exclure émergents", "Limiter techno à 25%", "Part défensive 30%", "Inclure produit structuré (16%)", "Inclure fonds en euros"]
    )

univers_default = [
    # Tickers EODHD — exemples: ETFs & indices (adapter selon univers réel du cabinet)
    {"ticker": "^FCHI", "name": "CAC 40 (indice)", "type": "index", "weight_cap": 0.20},
    {"ticker": "IWDA.AMS", "name": "iShares MSCI World (ETF)", "type": "etf", "weight_cap": 0.45},
    {"ticker": "ESE.PA", "name": "Amundi S&P 500 (ETF)", "type": "etf", "weight_cap": 0.45},
    {"ticker": "PEAEXUK.LSE", "name": "Developed Europe ex-UK (ETF)", "type": "etf", "weight_cap": 0.40},
    {"ticker": "BNP.PA", "name": "BNP Paribas (titre vif)", "type": "stock", "weight_cap": 0.10},
]

with st.expander("Univers d'investissement (démo)", expanded=False):
    st.caption("Vous pourrez brancher votre univers interne: ETF PEA, UC, SCPI, Structurés, etc.")
    st.dataframe(pd.DataFrame(univers_default))

# --- Bouton ---
if st.button("🧮 Générer l'allocation"):
    try:
        client = ClientInput(
            age=age, horizon=horizon, risk_profile=RiskProfile[risk_buckets[profil_nom]],
            amount=montant, pea=pea, constraints=contraintes_sp
        )
        api = EODHD(settings=settings)
        alloc_df, notes = build_allocation(client, univers_default, api)
        st.success("Allocation générée.")
        st.subheader("Proposition d'allocation")
        st.dataframe(alloc_df.style.format({"poids": "{:.1%}", "montant": "{:,.0f}"}))
        st.subheader("Notes de conformité & commentaires")
        for n in notes:
            st.markdown(f"- {n}")
    except Exception as e:
        st.error(f"Erreur: {e}")

st.divider()
st.caption("⚠️ Outil interne. Les performances passées ne préjugent pas des performances futures. Vérifier l'éligibilité PEA/assurance-vie et les documents réglementaires des supports.")
