# App Allocation CGP — Starter

Application privée pour CGP (France) permettant de générer des **allocations personnalisées** à partir des données **EODHD**.

## Pile technique
- **Streamlit** (UI locale rapide)
- **Python 3.10+**
- **Requests** (client API EODHD)
- **pandas / numpy** (données)
- **pydantic** (schémas)
- **PyPortfolioOpt** (optimisation — risque/retour, minimum variance, risk parity)
- **joblib** (cache) + **sqlite** simple (optionnel)

## Secrets & clés
- Mettre la clé **EODHD_API_KEY** dans votre `.env` (local) **et/ou** `st.secrets` si déploiement Streamlit.
- ⚠️ Ne jamais committer la clé. `.env` est ignoré par Git.

## Lancer en local
```bash
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env  # renseigner la clé EODHD
streamlit run app.py
```

## Qualité & CI
- **pre-commit** + **ruff** (lint/format) et **pytest**.
- Un workflow CI est fourni dans `.github/workflows/ci.yml` (facultatif, pour GitHub Actions).

## Déploiement
Usage interne uniquement (privé). Si besoin d'un partage, utiliser Streamlit Community Cloud avec `st.secrets`.
