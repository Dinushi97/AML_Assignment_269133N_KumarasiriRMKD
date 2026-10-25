# Demo video script (about 4½ minutes)

The guideline asks for a **3–5 minute demo video showing the front-end system**, so about half of the video is a live app demo.
Record your screen (QuickTime → File → New Screen Recording, or Zoom/OBS) with your voice. Open the app at
`https://dengue-ews-kumarasiri.azurewebsites.net`, or run it locally with `streamlit run app/app.py`.

**Before recording:** open the app and leave the first page loaded (the first load takes a few seconds). Have the slide deck open in another window.

| Time | Show | Say |
|---|---|---|
| 0:00–0:20 | Slide 1 (title) | “Hello, I’m R.M.K.D. Kumarasiri, index 269133N. My assignment is a machine-learning early-warning system for dengue in Sri Lanka. For every district, it predicts whether there will be a high-transmission dengue week two weeks from now, and it explains why.” |
| 0:20–0:50 | Slides 2–3 (problem, data) | “Dengue causes large epidemics here; this dataset records over 174,000 cases in 2017. Health teams need lead time for fogging and clean-ups. The data is real and local: the Epidemiology Unit’s weekly reports for all 26 RDHS areas from 2006 to 2026, about 26,000 rows, from the denguedatahub package. It’s aggregate public data, with no personal information.” |
| 0:50–1:15 | Slide 5 (data quality) | “The file had no missing values, but the EDA revealed hidden problems: Monaragala recorded zeros for the whole of 2020, and 22 weekly values were implausible, including one week where counts were shifted between districts. I treated these as missing and fixed or excluded them.” |
| 1:15–1:50 | Slides 9–11 (comparison, choice, algorithm) | “I tuned six models with Optuna and time-series cross-validation. The selected model is **LightGBM**, an algorithm not covered in lectures. It’s gradient boosting with histogram splits and leaf-wise growth. Unlike logistic regression or k-NN it captures thresholds and interactions without scaling. On the 2024–2026 test period it reaches PR-AUC 0.79 and recall 0.81, against 0.59 PR-AUC for the persistence baseline.” |
| 1:50–2:15 | Slide 13 (SHAP) | “SHAP shows the model relies on this week’s level and the four-week average relative to the threshold, then the national wave and the season, which matches dengue epidemiology.” |
| **2:15–2:50** | **App → National overview tab** (latest week, 11 May 2026) | “This is the Streamlit app. For the latest report week, it forecasts 18 of 26 districts as high-risk for the week of 25 May. Ratnapura, Galle and Matara have the highest risk.” *Hover over a bar to show its tooltip.* |
| **2:50–3:10** | **Sidebar → change the report week to 3 May 2025** | “I can go back to a week where the outcome is known. Here the model forecast 13 high-risk districts, and the accuracy for that week was 92%.” *Point to the accuracy figure.* |
| **3:10–3:50** | **District forecast tab → Colombo** (switch the week back to 11 May 2026) | “For Colombo the risk is about 75%: high. Below is the SHAP explanation for this single prediction. Red bars push the risk up, mainly because current cases are at the threshold.” *Scroll to the waterfall.* |
| **3:50–4:15** | **What-if:** in the table, change the last three weeks’ cases to **100** | “Users can enter their own numbers. If Colombo’s last three weeks had been 100 cases each, the risk drops to about 29%, low, and the explanation updates instantly.” |
| **4:15–4:30** | **Hold-out check tab** | “One week was held out from all training and testing. The saved model predicted high risk, and two weeks later Colombo really did report 308 cases, above its threshold of 291.” |
| 4:30–4:50 | Slide 16 (critical discussion), then slide 15 or the GitHub Actions page | “It’s less reliable where outbreaks are rare, such as Vavuniya, and it uses reported cases, so it should support the Epidemiology Unit, not replace it. The whole pipeline runs automatically on Azure: GitHub Actions trains, gates, registers and redeploys the model every week. Thank you.” |

**Tips**
* Speak at a calm pace. The script is about 560 words, which leaves time for the clicks.
* If the app is slow on Azure, record the demo part with the local app (`streamlit run app/app.py`); it's the same code and model.
* Keep the video between 3 and 5 minutes. To shorten it, skip the 0:50–1:15 row.
