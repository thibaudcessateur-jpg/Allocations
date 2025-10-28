import os
from typing import Optional, List, Dict, Any

import streamlit as st
import pandas as pd
from dotenv import load_dotenv

load_dotenv()
st.set_page_config(page_title="Allocation CGP — App", page_icon="🦉", layout="wide")

# ---------- Secrets helper ----------
def Secret_Token(name: str, default: Optional[str] = None) -> str:
    """
    Retrieve a secret/token from environment (preferred) or st.secrets.
    Raises a clear error if not found and no default provided.
    """
    # 1) OS env has priority
    v = os.getenv(name)
    if v and v.strip():
        return v.strip()
    # 2) Streamlit secrets
    try:
        v = st.secrets.get(name)  # type: ignore[attr-defined]
        if v and str(v).strip():
            return str(v).strip()
    except Exception:
        pass
    # 3) Fallback / error
    if default is not None:
        return default
    raise RuntimeError(f"Secret '{name}' is missing. Provide it via environment or st.secrets.")

# ---------- Page title ----------
st.title("🦉 Allocation CGP — Interface (univers Espace Invest 5)")
st.caption("Usage privé professionnel — univers renseigné, aucune allocation calculée pour l’instant.")

# ---------- Sidebar: status & secrets ----------
with st.sidebar:
    st.header("Paramètres")
    # EODHD non utilisé ici, simple contrôle de présence
    eodhd_ok = True
    try:
        _ = Secret_Token("EODHD_API_KEY")
    except Exception:
        eodhd_ok = False
    st.write("Clé EODHD:", "✅ détectée" if eodhd_ok else "⚠️ manquante (pas nécessaire à cette étape)")

    # GitHub repo info (private) pour lien de téléchargement
    gh_repo = os.getenv("GITHUB_REPO") or st.secrets.get("GITHUB_REPO", "")  # ex: thibaudcessateur-jpg/Allocations
    gh_branch = os.getenv("GITHUB_BRANCH") or st.secrets.get("GITHUB_BRANCH", "main")
    st.write("Repo GitHub:", gh_repo or "non défini")
    st.write("Branche:", gh_branch)

st.divider()

# ---------- Lien (optionnel) pour récupérer app.py depuis GitHub privé ----------
st.subheader("Télécharger ce script depuis le dépôt GitHub privé (optionnel)")
owner_repo = os.getenv("GITHUB_REPO") or st.secrets.get("GITHUB_REPO", "")
branch = os.getenv("GITHUB_BRANCH") or st.secrets.get("GITHUB_BRANCH", "main")
if owner_repo:
    api_url = f"https://api.github.com/repos/{owner_repo}/contents/app.py?ref={branch}"
    st.link_button("Ouvrir le fichier via l'API GitHub", api_url, use_container_width=True)
else:
    st.info("Définis `GITHUB_REPO` (ex: thibaudcessateur-jpg/Allocations) dans `.env` ou `st.secrets` pour activer le lien.")
st.code(
    'curl -s -H "Authorization: token $GITHUB_TOKEN" '
    '-H "Accept: application/vnd.github.raw+json" '
    f'https://api.github.com/repos/{owner_repo or "thibaudcessateur-jpg/Allocations"}/contents/app.py?ref={branch or "main"} '
    '> app.py',
    language="bash",
)
st.caption("Astuce : exporte ton token GitHub dans la variable d’environnement `GITHUB_TOKEN` côté machine locale.")

st.divider()

# ---------- Univers Espace Invest 5 (UC / obligations datées / fonds en euros) ----------
# Nota:
# - Tous les supports fournis ci-dessous ont "Versement libre désactivé" et "Versement libre programmé désactivé",
#   sauf mention explicite de contrainte particulière (ex. transferts programmés non éligibles).
# - Aucune logique d'allocation n'est appliquée ici.

UNIVERSE: List[Dict[str, Any]] = [
    # Actions Monde / Diversifiés actions
    {
        "name": "R-co Valor C EUR",
        "isin": "FR0011253624",
        "sri": 4,
        "sfdr": 8,
        "type": "UC Actions Monde",
        "versement_libre": False,
        "vlp": False,  # versement libre programmé
        "transferts_programmes_eligibles": True,
        "notes": "",
    },
    {
        "name": "Vivalor International",
        "isin": "FR0014001LS1",
        "sri": 4,
        "sfdr": None,  # non précisé dans le brief
        "type": "UC Actions Monde",
        "versement_libre": False,
        "vlp": False,
        "transferts_programmes_eligibles": False,  # ATTENTION - Non éligible aux transferts programmés
        "notes": "Non éligible aux transferts programmés",
    },
    {
        "name": "COMGEST Monde C",
        "isin": "FR0000284689",
        "sri": 4,
        "sfdr": 8,
        "type": "UC Actions Monde",
        "versement_libre": False,
        "vlp": False,
        "transferts_programmes_eligibles": True,
        "notes": "",
    },
    {
        "name": "Echiquier World Equity Growth",
        "isin": "FR0010859769",
        "sri": 4,
        "sfdr": 8,
        "type": "UC Actions Monde",
        "versement_libre": False,
        "vlp": False,
        "transferts_programmes_eligibles": True,
        "notes": "",
    },
    {
        "name": "Franklin Mutual Global Discovery",
        "isin": "LU0211333298",
        "sri": 4,
        "sfdr": 8,
        "type": "UC Actions Monde",
        "versement_libre": False,
        "vlp": False,
        "transferts_programmes_eligibles": True,
        "notes": "",
    },
    {
        "name": "CARMIGNAC INVESTISSEMENT A EUR",
        "isin": "FR0010148981",
        "sri": 4,
        "sfdr": 8,
        "type": "UC Actions Monde",
        "versement_libre": False,
        "vlp": False,
        "transferts_programmes_eligibles": True,
        "notes": "",
    },
    {
        "name": "Natixis - Thematics Meta A EUR",
        "isin": "LU1951204046",
        "sri": 5,
        "sfdr": 8,
        "type": "UC Thématique Monde",
        "versement_libre": False,
        "vlp": False,
        "transferts_programmes_eligibles": True,
        "notes": "",
    },
    {
        "name": "Pictet Global Megatrend Selection P",
        "isin": "LU0386882277",
        "sri": 4,
        "sfdr": 8,
        "type": "UC Thématique Monde",
        "versement_libre": False,
        "vlp": False,
        "transferts_programmes_eligibles": True,
        "notes": "",
    },
    {
        "name": "Morgan Stanley Gl Brands A",
        "isin": "LU0119620416",
        "sri": 4,
        "sfdr": 8,
        "type": "UC Actions Monde",
        "versement_libre": False,
        "vlp": False,
        "transferts_programmes_eligibles": True,
        "notes": "",
    },
    {
        "name": "FIDELITY FUNDS - WORLD FUND",
        "isin": "LU0069449576",
        "sri": 4,
        "sfdr": 8,
        "type": "UC Actions Monde",
        "versement_libre": False,
        "vlp": False,
        "transferts_programmes_eligibles": True,
        "notes": "",
    },
    {
        "name": "CLARTAN VALEURS",
        "isin": "LU1100076550",
        "sri": 4,
        "sfdr": 8,
        "type": "UC Actions Monde",
        "versement_libre": False,
        "vlp": False,
        "transferts_programmes_eligibles": True,
        "notes": "",
    },
    {
        "name": "CARMIGNAC PATRIMOINE",
        "isin": "FR0010135103",
        "sri": 3,
        "sfdr": 8,
        "type": "UC Diversifié (patrimonial)",
        "versement_libre": False,
        "vlp": False,
        "transferts_programmes_eligibles": True,
        "notes": "",
    },

    # Obligations datées / Fonds euro-croissance / monétaire assimilé si besoin
    {
        "name": "SYCOYIELD 2030 RC",
        "isin": "FR001400MCQ6",
        "sri": 2,
        "sfdr": 8,
        "type": "Obligataire daté (2024-2030)",
        "versement_libre": False,
        "vlp": False,
        "transferts_programmes_eligibles": True,
        "notes": "",
    },
    {
        "name": "R-Co Target 2029 HY",
        "isin": None,  # ISIN à compléter si besoin
        "sri": None,   # non fourni dans ton brief
        "sfdr": None,  # non fourni
        "type": "Obligataire daté HY (2029)",
        "versement_libre": False,
        "vlp": False,
        "transferts_programmes_eligibles": True,
        "notes": "",
    },
    {
        "name": "Fonds en euros AGGV",
        "isin": None,  # fonds en euros de Generali (AGGV) — pas d'ISIN standard
        "sri": 1,      # usuellement très faible (indicatif)
        "sfdr": None,
        "type": "Fonds en euros",
        "versement_libre": False,
        "vlp": False,
        "transferts_programmes_eligibles": True,
        "notes": "",
    },
]

st.subheader("Univers des supports — Espace Invest 5")
st.caption("Liste fournie par le conseiller. Aucune allocation appliquée à ce stade.")

df = pd.DataFrame(UNIVERSE)

# ---------- Filtres ----------
with st.expander("Filtres", expanded=True):
    colf1, colf2, colf3, colf4 = st.columns([1.2, 1, 1, 1.2])
    with colf1:
        type_sel = st.multiselect(
            "Type de support",
            options=sorted(df["type"].dropna().unique().tolist()),
            default=sorted(df["type"].dropna().unique().tolist()),
        )
    with colf2:
        sri_max = st.slider("SRI max", min_value=1, max_value=7, value=7)
    with colf3:
        sfdr_opts = ["Tout", 6, 8, 9]
        sfdr_sel = st.selectbox("SFDR", options=sfdr_opts, index=0)
    with colf4:
        excl_transferts_non_eligibles = st.checkbox("Exclure 'non éligible transferts programmés'", value=False)

# Appliquer filtres
mask = pd.Series([True] * len(df))
if type_sel:
    mask &= df["type"].isin(type_sel)
if sri_max is not None:
    mask &= df["sri"].fillna(99) <= sri_max
if sfdr_sel != "Tout":
    mask &= df["sfdr"] == sfdr_sel
if excl_transferts_non_eligibles:
    mask &= df["transferts_programmes_eligibles"] == True

view = df.loc[mask].copy()
view = view[
    ["name", "isin", "type", "sri", "sfdr", "versement_libre", "vlp", "transferts_programmes_eligibles", "notes"]
].sort_values(["type", "name"])

st.dataframe(
    view.rename(
        columns={
            "name": "Nom",
            "isin": "ISIN",
            "type": "Type",
            "sri": "SRI (1-7)",
            "sfdr": "SFDR",
            "versement_libre": "Versement libre",
            "vlp": "Versement libre programmé",
            "transferts_programmes_eligibles": "Elig. transferts programmés",
            "notes": "Notes",
        }
    ),
    use_container_width=True,
    hide_index=True,
)

# Export CSV
csv = view.to_csv(index=False).encode("utf-8")
st.download_button("Exporter la vue filtrée (CSV)", data=csv, file_name="univers_espace_invest5.csv", mime="text/csv")

st.divider()

# ---------- Placeholder formulaire (sans allocation) ----------
with st.form("client_form"):
    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.number_input("Âge du client", min_value=18, max_value=99, value=40)
    with col2:
        profil = st.selectbox("Profil de risque", ["Prudent", "Équilibré", "Dynamique", "Agressif"], index=1)
    with col3:
        montant = st.number_input("Montant à allouer (€)", min_value=1000, step=1000, value=50000)

    st.caption("⚠️ Aucune allocation n'est générée à ce stade — on la construira ensuite.")
    submitted = st.form_submit_button("Continuer (sans générer)")

if submitted:
    st.success("Formulaire enregistré. La logique d'allocation arrive à l’étape suivante.")

st.divider()
st.caption("⚠️ Version univers + filtres — pas d’allocation calculée pour l’instant.")
