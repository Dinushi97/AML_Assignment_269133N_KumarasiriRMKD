# Demo video script (about 4½ minutes)

The guideline asks for a **3–5 minute demo video showing the front-end system**, so about half of the video is a live app demo.
Record your screen (QuickTime → File → New Screen Recording, or Zoom/OBS) with your voice. Open the app at
`https://dengue-ews-kumarasiri.azurewebsites.net`, or run it locally with `streamlit run app/app.py`.

**Before recording:** open the app and leave the first page loaded (the first load takes a few seconds). Have the slide deck open in another window.

| Time | Show | Say |
|---|---|---|
| 0:00–0:20 | Slide 1 (title) | “Hello, I’m R.M.K.D. Kumarasiri, index 269133N. My assignment is a dengue early-warning system for Sri Lanka. For every district, it predicts whether there will be a high-transmission week two weeks from now. It uses a new interpretable algorithm, the Explainable Boosting Machine.” |
| 0:20–0:50 | Slides 2–3 (problem, data) | “Dengue causes large epidemics here; this dataset records over 174,000 cases in 2017. Health teams need lead time for fogging and clean-ups. The data is real and local: the Epidemiology Unit’s weekly reports for all 26 RDHS areas from 2006 to 2026, about 26,000 rows. It’s aggregate public data, with no personal information.” |
| 0:50–1:10 | Slide 5 (data quality) | “The EDA revealed hidden problems: Monaragala recorded zeros for the whole of 2020, and 22 weekly values were implausible. I treated these as missing and fixed or excluded them.” |
| 1:10–1:45 | Slides 8 and 10 (new algorithm, comparison) | “The new algorithm is the **Explainable Boosting Machine**. It learns one smooth curve per feature plus a few interactions, so every prediction is an exact sum of readable parts. I compared it with logistic regression, a decision tree, k-NN, Random Forest and XGBoost. EBM reaches PR-AUC 0.78 and 80% recall, against 0.59 for the persistence baseline, close to the best black boxes but fully explainable.” |
| 1:45–2:10 | Slide 13 (EBM glass box) | “Because it’s a glass box, we can read what it learned: risk rises steeply as recent cases approach the district’s threshold, nationwide waves raise every district, and above-threshold weeks matter most in big districts like Colombo. SHAP and LIME confirm the same drivers.” |
| **2:10–2:45** | **App → National overview tab** (latest week, 11 May 2026) | “This is the Streamlit app. For the latest report week it forecasts 21 of 26 districts as high-risk for the week of 25 May. Ratnapura, Matara and Galle have the highest risk.” *Hover over a bar to show its tooltip.* |
| **2:45–3:05** | **Sidebar → change the report week to 3 May 2025** | “I can go back to a week where the outcome is known. Here the model forecast 14 high-risk districts, 13 actually were, and accuracy for that week was 88%.” *Point to the accuracy figure.* |
| **3:05–3:45** | **District forecast tab → Colombo** (switch the week back to 11 May 2026) | “For Colombo the risk is about 60%: high. Below is the EBM’s exact explanation. The baseline plus these contributions adds up to exactly this probability. The four-week average, rising national cases and this week’s level push the risk up.” *Scroll to the explanation chart.* |
| **3:45–4:10** | **What-if:** in the table, change the last three weeks’ cases to **100** | “Users can enter their own numbers. If Colombo’s last three weeks had been 100 cases each, the risk drops to about 10%, low, and the explanation updates instantly.” |
| **4:10–4:25** | **Hold-out check tab** | “One week was held out from all training and testing. The saved model predicted high risk, and two weeks later Colombo really did report 308 cases, above its threshold of 291.” |
| 4:25–4:50 | Slide 16 (critical discussion), then slide 15 or the GitHub Actions page | “It’s less reliable where outbreaks are rare, such as Vavuniya, and it uses reported cases, so it should support the Epidemiology Unit, not replace it. The pipeline runs on Azure: GitHub Actions trains the EBM, checks it against the baseline, registers it and redeploys this app whenever I start it. Thank you.” |

**Tips**
* Speak at a calm pace. The script is about 590 words, which leaves time for the clicks.
* If the app is slow on Azure, record the demo part with the local app (`streamlit run app/app.py`); it's the same code and model.
* Keep the video between 3 and 5 minutes. To shorten it, skip the 0:50–1:10 row.
