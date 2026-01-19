diff --git a/app.py b/app.py
index a5772824f245811d5a6952335deefbc439d49029..0b72e06dec7b4013d747687c58f129000e26ba9c 100644
--- a/app.py
+++ b/app.py
@@ -1,86 +1,155 @@
 from __future__ import annotations
 
+import base64
 import json
+from io import BytesIO
 from datetime import date
 from typing import Any, Dict, List, Optional, Tuple
 
 import numpy as np
 import pandas as pd
 import requests
 import streamlit as st
 import altair as alt
+import matplotlib.pyplot as plt
+from weasyprint import HTML
 
 # ------------------------------------------------------------
 # Constantes & univers de fonds recommandés
 # ------------------------------------------------------------
 TODAY = pd.Timestamp.today().normalize()
 APP_TITLE = "Comparateur de portefeuilles"
 
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
     ("R-Co Target 2029 HY", "FR0014002XJ3"),
     ("Euro Bond 1-3 Years", "LU0321462953"),
 ]
 
+# Frais de gestion du contrat
+MGMT_FEE_EURO_PCT = 0.9
+MGMT_FEE_UC_PCT = 1.2
+
 # Libellés FR -> codes internes pour l'affectation des versements
 ALLOC_LABELS = {
     "Répartition égale": "equal",
     "Personnalisé": "custom",
     "Tout sur une ligne": "single",
 }
 
 
 # ------------------------------------------------------------
 # Utils format
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
 
 
+def contract_mgmt_fee_pct(isin_or_name: str) -> float:
+    code = str(isin_or_name or "").strip().upper()
+    if code == "EUROFUND":
+        return MGMT_FEE_EURO_PCT
+    return MGMT_FEE_UC_PCT
+
+
+def apply_management_fee(df: pd.DataFrame, fee_pct: float) -> pd.DataFrame:
+    if df.empty or fee_pct <= 0:
+        return df
+    df_adj = df.copy()
+    days = (df_adj.index - df_adj.index[0]).days.astype(float)
+    factor = np.power(1.0 - fee_pct / 100.0, days / 365.0)
+    df_adj["Close"] = df_adj["Close"].astype(float) * factor
+    return df_adj
+
+
+def _fig_to_base64_png(fig: plt.Figure) -> str:
+    buf = BytesIO()
+    fig.savefig(buf, format="png", bbox_inches="tight", dpi=150)
+    plt.close(fig)
+    return base64.b64encode(buf.getvalue()).decode("utf-8")
+
+
+def _portfolio_pie_chart_b64(df_lines: pd.DataFrame, title: str) -> str:
+    if df_lines is None or df_lines.empty:
+        return ""
+    if "Nom" not in df_lines.columns or "Valeur actuelle €" not in df_lines.columns:
+        return ""
+    values = df_lines[["Nom", "Valeur actuelle €"]].copy()
+    values = values[values["Valeur actuelle €"] > 0]
+    if values.empty:
+        return ""
+    fig, ax = plt.subplots(figsize=(5, 4))
+    ax.pie(
+        values["Valeur actuelle €"],
+        labels=values["Nom"],
+        autopct="%1.1f%%",
+        startangle=90,
+    )
+    ax.set_title(title)
+    ax.axis("equal")
+    return _fig_to_base64_png(fig)
+
+
+def _portfolio_value_chart_b64(df_client: pd.DataFrame, df_valority: pd.DataFrame) -> str:
+    if (df_client is None or df_client.empty) and (df_valority is None or df_valority.empty):
+        return ""
+    fig, ax = plt.subplots(figsize=(7, 3))
+    if df_client is not None and not df_client.empty:
+        ax.plot(df_client.index, df_client["Valeur"], label="Client")
+    if df_valority is not None and not df_valority.empty:
+        ax.plot(df_valority.index, df_valority["Valeur"], label="Valority")
+    ax.set_title("Évolution de la valeur des portefeuilles")
+    ax.set_xlabel("Date")
+    ax.set_ylabel("Valeur (€)")
+    ax.legend()
+    fig.autofmt_xdate()
+    return _fig_to_base64_png(fig)
+
+
 # ------------------------------------------------------------
 # XIRR
 # ------------------------------------------------------------
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
@@ -297,51 +366,52 @@ def suggest_alternative_funds(buy_date: pd.Timestamp, euro_rate: float) -> List[
             alternatives.append((name, isin, inception))
 
     return alternatives
 def correlation_matrix_from_lines(
     lines: List[Dict[str, Any]],
     euro_rate: float,
     years: int = 3,
     min_points: int = 30,
 ) -> pd.DataFrame:
     """
     Construit une matrice de corrélation des rendements quotidiens
     pour les lignes d'un portefeuille donné.
 
     - On récupère les VL quotidiennes via get_price_series
     - On restreint à 'years' années de données (fenêtre glissante)
     - On calcule les rendements journaliers (pct_change)
     - On renvoie corrélation de ces rendements.
     """
     series_map: Dict[str, pd.Series] = {}
     cutoff = TODAY - pd.Timedelta(days=365 * years)
 
     for ln in lines:
         label = ln.get("name") or ln.get("isin") or "Ligne"
         key = f"{label} ({ln.get('isin','')})"
 
-        df, _, _ = get_price_series(ln.get("isin") or ln.get("name"), None, euro_rate)
+        df_raw, sym, _ = get_price_series(ln.get("isin") or ln.get("name"), None, euro_rate)
+        df = apply_management_fee(df_raw, contract_mgmt_fee_pct(sym or ln.get("isin") or ln.get("name")))
         if df.empty:
             continue
 
         s = df["Close"].astype(float)
         s = s[s.index >= cutoff]
         if s.size < min_points:
             continue
 
         series_map[key] = s
 
     if len(series_map) < 2:
         return pd.DataFrame()
 
     df_prices = pd.DataFrame(series_map).dropna(how="all")
     if df_prices.shape[0] < min_points:
         return pd.DataFrame()
 
     returns = df_prices.pct_change().dropna(how="any")
     if returns.empty:
         return pd.DataFrame()
 
     corr = returns.corr()
     return corr
 
 
@@ -469,60 +539,61 @@ def simulate_portfolio(
             invalid_found = True
             ln["invalid_date"] = True
             ln["inception_date"] = inception
 
             alts = suggest_alternative_funds(d_buy, euro_rate)
             if alts:
                 alt_lines = [
                     f"- {name} ({isin}), historique depuis le {fmt_date(incep)}"
                     for name, isin, incep in alts
                 ]
                 alt_msg = "\n".join(alt_lines)
             else:
                 alt_msg = "Aucun fonds recommandé ne dispose d'un historique suffisant pour cette date."
 
             date_warnings.append(
                 f"[{portfolio_label}] {ln.get('name','(sans nom)')} "
                 f"({ln.get('isin','—')}) :\n"
                 f"- Date d'achat saisie : {fmt_date(d_buy)}\n"
                 f"- 1ère VL disponible : {fmt_date(inception)}\n\n"
                 f"Impossible de simuler ce fonds sur toute la période demandée.\n"
                 f"Propositions d'alternatives pour l'analyse historique :\n{alt_msg}"
             )
             continue
 
         ln["sym_used"] = sym
-        df = df_full
+        mgmt_fee = contract_mgmt_fee_pct(sym or isin_or_name)
+        df = apply_management_fee(df_full, mgmt_fee)
 
         if d_buy in df.index:
-            px_buy = float(df.loc[d_buy, "Close"])
+            px_buy = float(df_full.loc[d_buy, "Close"])
             eff_dt = d_buy
         else:
-            after = df.loc[df.index >= d_buy]
+            after = df_full.loc[df_full.index >= d_buy]
             if after.empty:
-                px_buy = float(df.iloc[-1]["Close"])
-                eff_dt = df.index[-1]
+                px_buy = float(df_full.iloc[-1]["Close"])
+                eff_dt = df_full.index[-1]
             else:
                 px_buy = float(after.iloc[0]["Close"])
                 eff_dt = after.index[0]
 
         px_manual = ln.get("buy_px", None)
         px_for_qty = float(px_manual) if (px_manual not in (None, "", 0, "0")) else px_buy
 
         price_map[key_id] = df["Close"].astype(float)
         eff_buy_date[key_id] = eff_dt
         buy_price_used[key_id] = px_for_qty
 
     if invalid_found and not price_map:
         return pd.DataFrame(), 0.0, 0.0, 0.0, None, TODAY, TODAY
     if not price_map:
         return pd.DataFrame(), 0.0, 0.0, 0.0, None, TODAY, TODAY
 
     start_min = min(eff_buy_date.values())
     start_full = max(eff_buy_date.values())
 
     bidx = pd.bdate_range(start=start_min, end=TODAY, freq="B")
     prices = pd.DataFrame(index=bidx)
     for key_id, s in price_map.items():
         prices[key_id] = s.reindex(bidx).ffill()
 
     qty_events = pd.DataFrame(0.0, index=bidx, columns=prices.columns)
@@ -690,51 +761,52 @@ def _line_card(line: Dict[str, Any], idx: int, port_key: str):
                             line["buy_px"] = float(str(new_px).replace(",", "."))
                         except Exception:
                             line["buy_px"] = ""
                     else:
                         line["buy_px"] = ""
                     line.pop("invalid_date", None)
                     line.pop("inception_date", None)
                     st.session_state[state_key] = False
                     st.success("Ligne mise à jour.")
                     st.experimental_rerun()
 
 def portfolio_summary_dataframe(port_key: str) -> pd.DataFrame:
     """
     Construit un DataFrame synthétique par ligne :
     Nom, ISIN, Net investi, Valeur actuelle, Perf € et Perf %.
     """
     fee_pct = st.session_state.get("FEE_A", 0.0) if port_key == "A_lines" else st.session_state.get("FEE_B", 0.0)
     euro_rate = st.session_state.get("EURO_RATE_PREVIEW", 2.0)
     lines = st.session_state.get(port_key, [])
 
     rows: List[Dict[str, Any]] = []
 
     for ln in lines:
         net_amt, buy_px, qty = compute_line_metrics(ln, fee_pct, euro_rate)
 
-        dfl, _, _ = get_price_series(ln.get("isin") or ln.get("name"), None, euro_rate)
+        dfl_raw, sym, _ = get_price_series(ln.get("isin") or ln.get("name"), None, euro_rate)
+        dfl = apply_management_fee(dfl_raw, contract_mgmt_fee_pct(sym or ln.get("isin") or ln.get("name")))
         if not dfl.empty:
             last_px = float(dfl["Close"].iloc[-1])
         else:
             last_px = np.nan
 
         val_now = qty * last_px if last_px == last_px else 0.0
         perf_abs = val_now - net_amt
         perf_pct = (val_now / net_amt - 1.0) * 100.0 if net_amt > 0 else np.nan
 
         rows.append(
             {
                 "Nom": ln.get("name", ""),
                 "ISIN / Code": ln.get("isin", ""),
                 "Net investi €": net_amt,
                 "Valeur actuelle €": val_now,
                 "Perf €": perf_abs,
                 "Perf %": perf_pct,
             }
         )
 
     df = pd.DataFrame(rows)
     return df
 # ------------------------------------------------------------
 # Tableau synthétique par ligne (un seul tableau par portefeuille)
 # ------------------------------------------------------------
@@ -744,51 +816,52 @@ def positions_table(title: str, port_key: str):
     Nom, ISIN, Date d'achat, Net investi, Valeur actuelle, Perf € et Perf %.
     """
     fee_pct = (
         st.session_state.get("FEE_A", 0.0)
         if port_key == "A_lines"
         else st.session_state.get("FEE_B", 0.0)
     )
 
     # ✅ Taux fonds euros par portefeuille (au lieu de EURO_RATE_PREVIEW)
     euro_rate = (
         st.session_state.get("EURO_RATE_A", 2.0)
         if port_key == "A_lines"
         else st.session_state.get("EURO_RATE_B", 2.5)
     )
 
     lines = st.session_state.get(port_key, [])
     rows: List[Dict[str, Any]] = []
 
     for ln in lines:
         buy_ts = pd.Timestamp(ln.get("buy_date"))
 
         # Montant net investi, VL d'achat et quantité
         net_amt, buy_px, qty = compute_line_metrics(ln, fee_pct, euro_rate)
 
         # ✅ IMPORTANT : on récupère la série "depuis buy_ts" pour éviter le mismatch EUROFUND
-        dfl, _, _ = get_price_series(ln.get("isin") or ln.get("name"), buy_ts, euro_rate)
+        dfl_raw, sym, _ = get_price_series(ln.get("isin") or ln.get("name"), buy_ts, euro_rate)
+        dfl = apply_management_fee(dfl_raw, contract_mgmt_fee_pct(sym or ln.get("isin") or ln.get("name")))
 
         if not dfl.empty:
             last_px = float(dfl["Close"].iloc[-1])
         else:
             last_px = np.nan
 
         # Valeur actuelle et performance
         val_now = qty * last_px if last_px == last_px else 0.0
         perf_abs = val_now - net_amt
         perf_pct = (val_now / net_amt - 1.0) * 100.0 if net_amt > 0 else np.nan
 
         rows.append(
             {
                 "Nom": ln.get("name", ""),
                 "ISIN / Code": ln.get("isin", ""),
                 "Date d'achat": fmt_date(ln.get("buy_date")),
                 "Net investi €": net_amt,
                 "Valeur actuelle €": val_now,
                 "Perf €": perf_abs,
                 "Perf %": perf_pct,
             }
         )
 
     st.markdown(f"### {title}")
     df = pd.DataFrame(rows)
@@ -810,51 +883,52 @@ def positions_table(title: str, port_key: str):
 
 
 # ------------------------------------------------------------
 # Analytics internes : retours, corrélation, volatilité
 # ------------------------------------------------------------
 
 def _build_returns_df(
     lines: List[Dict[str, Any]],
     euro_rate: float,
     years: int = 3,
     min_points: int = 60,
 ) -> pd.DataFrame:
     """
     Construit un DataFrame de rendements journaliers (pct_change)
     pour toutes les lignes du portefeuille avec un historique suffisant.
     Index = dates, colonnes = "Nom (ISIN)".
     """
     cutoff = TODAY - pd.Timedelta(days=365 * years)
     series_map: Dict[str, pd.Series] = {}
 
     for ln in lines:
         label = (ln.get("name") or ln.get("isin") or "Ligne").strip()
         isin = (ln.get("isin") or "").strip()
         key = f"{label} ({isin})" if isin else label
 
-        df, _, _ = get_price_series(ln.get("isin") or ln.get("name"), None, euro_rate)
+        df_raw, sym, _ = get_price_series(ln.get("isin") or ln.get("name"), None, euro_rate)
+        df = apply_management_fee(df_raw, contract_mgmt_fee_pct(sym or ln.get("isin") or ln.get("name")))
         if df.empty:
             continue
 
         s = df["Close"].astype(float)
         s = s[s.index >= cutoff]
         if s.size < min_points:
             continue
 
         series_map[key] = s
 
     if not series_map:
         return pd.DataFrame()
 
     df_prices = pd.DataFrame(series_map).dropna(how="any")
     if df_prices.shape[0] < min_points:
         return pd.DataFrame()
 
     returns = df_prices.pct_change().dropna(how="any")
     return returns
 
 
 def correlation_matrix_from_lines(
     lines: List[Dict[str, Any]],
     euro_rate: float,
     years: int = 3,
@@ -1592,50 +1666,53 @@ if mode in ("compare", "valority"):
 - Rendement total depuis le début : **—**
 """
             )
             st.markdown(
                 f"- Rendement annualisé (XIRR) : **{xirrB:.2f}%**"
                 if xirrB is not None
                 else "- Rendement annualisé (XIRR) : **—**"
             )
 
 
 
 def build_html_report(report: Dict[str, Any]) -> str:
     """
     Construit un rapport HTML exportable pour le client.
     Le contenu repose sur 'report', préparé plus bas dans le code.
     """
     as_of = report.get("as_of", "")
     synthA = report.get("client_summary", {})
     synthB = report.get("valority_summary", {})
     comp = report.get("comparison", {})
 
     dfA_lines = report.get("df_client_lines")
     dfB_lines = report.get("df_valority_lines")
     dfA_val = report.get("dfA_val")
     dfB_val = report.get("dfB_val")
+    chart_value_b64 = report.get("chart_value_b64", "")
+    pie_client_b64 = report.get("pie_client_b64", "")
+    pie_valority_b64 = report.get("pie_valority_b64", "")
 
     def _fmt_eur(x):
         try:
             return f"{x:,.2f} €".replace(",", " ").replace(".", ",")
         except Exception:
             return str(x)
 
     # Tables en HTML
     html_client_lines = dfA_lines.to_html(index=False, border=0, justify="left") if dfA_lines is not None else ""
     html_valority_lines = dfB_lines.to_html(index=False, border=0, justify="left") if dfB_lines is not None else ""
 
     if dfA_val is not None:
         html_A_val = dfA_val.to_html(index=False, border=0, justify="left")
     else:
         html_A_val = ""
 
     if dfB_val is not None:
         html_B_val = dfB_val.to_html(index=False, border=0, justify="left")
     else:
         html_B_val = ""
 
     html = f"""
 <!DOCTYPE html>
 <html lang="fr">
 <head>
@@ -1653,125 +1730,198 @@ h1, h2, h3 {{
 table {{
   border-collapse: collapse;
   width: 100%;
   margin: 8px 0 16px 0;
   font-size: 14px;
 }}
 th, td {{
   border: 1px solid #ddd;
   padding: 6px 8px;
 }}
 th {{
   background-color: #f5f5f5;
   text-align: left;
 }}
 .small {{
   font-size: 12px;
   color: #666;
 }}
 .block {{
   border: 1px solid #eee;
   border-radius: 8px;
   padding: 12px 16px;
   margin-bottom: 16px;
   background-color: #fafafa;
 }}
+img {{
+  max-width: 100%;
+  height: auto;
+}}
 </style>
 </head>
 <body>
 
 <h1>Rapport de portefeuille</h1>
 <p class="small">Date de génération : {as_of}</p>
 
 <h2>1. Synthèse chiffrée</h2>
 
 <div class="block">
   <h3>Situation actuelle — Client</h3>
   <ul>
     <li>Valeur actuelle : <b>{_fmt_eur(synthA.get("val", 0))}</b></li>
     <li>Montants réellement investis (net) : {_fmt_eur(synthA.get("net", 0))}</li>
     <li>Montants versés (brut) : {_fmt_eur(synthA.get("brut", 0))}</li>
     <li>Rendement total depuis le début : <b>{synthA.get("perf_tot_pct", 0):.2f} %</b></li>
     <li>Rendement annualisé (XIRR) : <b>{synthA.get("irr_pct", 0):.2f} %</b></li>
   </ul>
 </div>
 
 <div class="block">
   <h3>Simulation — Allocation Valority</h3>
   <ul>
     <li>Valeur actuelle simulée : <b>{_fmt_eur(synthB.get("val", 0))}</b></li>
     <li>Montants réellement investis (net) : {_fmt_eur(synthB.get("net", 0))}</li>
     <li>Montants versés (brut) : {_fmt_eur(synthB.get("brut", 0))}</li>
     <li>Rendement total depuis le début : <b>{synthB.get("perf_tot_pct", 0):.2f} %</b></li>
     <li>Rendement annualisé (XIRR) : <b>{synthB.get("irr_pct", 0):.2f} %</b></li>
   </ul>
 </div>
 
 <div class="block">
   <h3>Comparaison Client vs Valority</h3>
   <ul>
     <li>Différence de valeur finale : <b>{_fmt_eur(comp.get("delta_val", 0))}</b></li>
     <li>Écart de performance totale (Valority – Client) :
         <b>{comp.get("delta_perf_pct", 0):.2f} %</b></li>
   </ul>
 </div>
 
-<h2>2. Détail des lignes</h2>
+<h2>2. Composition du portefeuille</h2>
+
+<h3>Client</h3>
+{"<img src='data:image/png;base64," + pie_client_b64 + "' alt='Composition Client' />" if pie_client_b64 else "<p class='small'>Composition indisponible.</p>"}
+
+<h3>Valority</h3>
+{"<img src='data:image/png;base64," + pie_valority_b64 + "' alt='Composition Valority' />" if pie_valority_b64 else "<p class='small'>Composition indisponible.</p>"}
+
+<h2>3. Détail des lignes</h2>
 
 <h3>Portefeuille Client</h3>
 {html_client_lines}
 
 <h3>Portefeuille Valority</h3>
 {html_valority_lines}
 
-<h2>3. Historique de la valeur des portefeuilles</h2>
+<h2>4. Historique de la valeur des portefeuilles</h2>
+
+{"<img src='data:image/png;base64," + chart_value_b64 + "' alt='Évolution de la valeur' />" if chart_value_b64 else "<p class='small'>Graphique indisponible.</p>"}
 
 <h3>Client – Valeur du portefeuille par date</h3>
 {html_A_val}
 
 <h3>Valority – Valeur du portefeuille par date</h3>
 {html_B_val}
 
 <p class="small">
 Ce document est fourni à titre informatif uniquement et ne constitue pas un conseil en investissement
 personnalisé.
 </p>
 
 </body>
 </html>
 """
     return html
 
-    report_data = st.session_state.get("REPORT_DATA")
-    if report_data is not None:
-        html_report = build_html_report(report_data)
-        st.download_button(
-            "📄 Télécharger le rapport complet (HTML)",
-            data=html_report.encode("utf-8"),
-            file_name="rapport_portefeuille_valority.html",
-            mime="text/html",
-        )
+
+st.markdown("---")
+st.subheader("📄 Rapport client")
+
+df_client_lines = portfolio_summary_dataframe("A_lines")
+df_valority_lines = portfolio_summary_dataframe("B_lines")
+
+dfA_val = dfA.reset_index().rename(columns={"index": "Date"}) if not dfA.empty else None
+dfB_val = dfB.reset_index().rename(columns={"index": "Date"}) if not dfB.empty else None
+
+chart_value_b64 = _portfolio_value_chart_b64(dfA, dfB)
+pie_client_b64 = _portfolio_pie_chart_b64(df_client_lines, "Composition — Client")
+pie_valority_b64 = _portfolio_pie_chart_b64(df_valority_lines, "Composition — Valority")
+
+report_data = {
+    "as_of": fmt_date(TODAY),
+    "client_summary": {
+        "val": valA,
+        "net": netA,
+        "brut": brutA,
+        "perf_tot_pct": perf_tot_client or 0.0,
+        "irr_pct": xirrA or 0.0,
+    },
+    "valority_summary": {
+        "val": valB,
+        "net": netB,
+        "brut": brutB,
+        "perf_tot_pct": perf_tot_valority or 0.0,
+        "irr_pct": xirrB or 0.0,
+    },
+    "comparison": {
+        "delta_val": (valB - valA) if (valA is not None and valB is not None) else 0.0,
+        "delta_perf_pct": (
+            (perf_tot_valority - perf_tot_client)
+            if (perf_tot_client is not None and perf_tot_valority is not None)
+            else 0.0
+        ),
+    },
+    "df_client_lines": df_client_lines,
+    "df_valority_lines": df_valority_lines,
+    "dfA_val": dfA_val,
+    "dfB_val": dfB_val,
+    "chart_value_b64": chart_value_b64,
+    "pie_client_b64": pie_client_b64,
+    "pie_valority_b64": pie_valority_b64,
+}
+
+html_report = build_html_report(report_data)
+st.download_button(
+    "📄 Télécharger le rapport complet (HTML)",
+    data=html_report.encode("utf-8"),
+    file_name="rapport_portefeuille_valority.html",
+    mime="text/html",
+)
+
+try:
+    pdf_report = HTML(string=html_report).write_pdf()
+    st.download_button(
+        "📕 Télécharger le rapport complet (PDF)",
+        data=pdf_report,
+        file_name="rapport_portefeuille_valority.pdf",
+        mime="application/pdf",
+    )
+except Exception:
+    st.warning(
+        "La génération PDF est indisponible dans cet environnement. "
+        "Vérifiez les dépendances système de WeasyPrint.",
+    )
 
 # ------------------------------------------------------------
 # Bloc final : Comparaison OU "Frais & valeur créée"
 # ------------------------------------------------------------
 mode = st.session_state.get("MODE_ANALYSE", "compare")
 
 def _years_between(d0: pd.Timestamp, d1: pd.Timestamp) -> float:
     return max(0.0, (d1 - d0).days / 365.25)
 
 # ============================
 # CAS 1 — MODE COMPARAISON
 # ============================
 if mode == "compare":
     st.subheader("📌 Comparaison : Client vs Valority")
 
     gain_vs_client = (valB - valA) if (valA is not None and valB is not None) else 0.0
     delta_xirr = (xirrB - xirrA) if (xirrA is not None and xirrB is not None) else None
     perf_diff_tot = (
         (perf_tot_valority - perf_tot_client)
         if (perf_tot_client is not None and perf_tot_valority is not None)
         else None
     )
 
     with st.container(border=True):
         c1, c2, c3 = st.columns(3)
