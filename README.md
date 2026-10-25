# Loan Delinquency Prediction

A Data Scientist I case study. It ranks loan applicants by their relative risk of becoming delinquent, using third-party identity and digital-footprint attributes, and gives each test applicant a 1–100 risk score.

## Results

| | PR-AUC | ROC-AUC |
|---|---|---|
| Holdout (2,998 records, untouched until the end) | **0.206** | **0.665** |
| Random baseline | 0.088 | 0.500 |

- **Final model:** logistic regression on 21 features, about 2.3× better than random on PR-AUC.
- **Riskiest 5%:** these applicants were delinquent about four times as often as average (34% vs 9%).
- **Strongest signals:** no WhatsApp on the phone number, billing type, not being found in the vendor's email–name lookup, and a weaker online presence.
- **Main limitations:**
  - The training data is 94% Airtel customers, against 45% in the test set.
  - The data collection date is unknown, so leakage can't be ruled out.
  - The features describe identity and digital presence, not ability to repay.

Full details are in the report and in `docs/findings_and_decisions.md`.

## Approach

1. **Data quality.** The text "NaT" was used as a missing-value placeholder, and −1 meant "not found in the vendor network" (59–84% of some columns). Constant columns were dropped. Phone numbers had been truncated in the source file, so they were not used.
2. **Split before analysis.** A stratified 70/30 split: all exploration and model selection used the 70% development set, and the 30% holdout was used only for the final evaluation.
3. **Feature engineering.**
   - Email and phone platform-presence shares.
   - "Not found" flags.
   - Ordinal footprint codes.
   - Fixed category levels.
   
   Every step uses only the row's own values, so it is leakage-safe.
4. **Validation.**
   - Stratified 5-fold cross-validation, with PR-AUC as the primary metric and ROC-AUC alongside.
   - Preprocessing runs inside a scikit-learn pipeline.
   - A change was adopted only if it improved PR-AUC by more than the fold-to-fold noise (about 0.03).
5. **Model comparison.**
   - Logistic regression against Random Forest, HistGradientBoosting and CatBoost.
   - The selected 22 features against all 84 usable columns.
   - Also tested: permutation importance, class weighting, sensitivity checks, an ablation and a rank-average ensemble.
   
   None beat logistic regression by more than the noise, so the simplest model was kept. Removing the network-operator feature cost nothing, which gave the final 21 features.
6. **Scoring.** The score is the percentile of predicted risk within a fixed reference distribution (out-of-fold predictions on the training data). It measures relative risk, not a calibrated probability.

## Repository structure

```
├── notebooks/
│   ├── eda.ipynb            # exploratory data analysis (working notebook)
│   ├── fe.ipynb             # feature development and checks (working notebook)
│   ├── features.py          # feature-engineering functions
│   └── modeling.ipynb       # model experiments (working notebook)
├── submission/
│   ├── Model_code_Atharva_Divekar.ipynb   # complete, self-contained final workflow
│   └── Report_Atharva_Divekar.docx        # case study report
├── docs/
│   └── findings_and_decisions.md          # findings and full decision log
├── requirements.txt
└── README.md
```

`submission/Model_code_Atharva_Divekar.ipynb` is the authoritative workflow and reproduces every number in the report. The other notebooks are the working record of how each decision was reached.

## How to run

1. Create an environment (tested with Python 3.14.7) and install the dependencies:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
2. Place `Assignment_Train.xlsx` and `Assignment_Test.xlsx` in a `data/` folder at the repository root.
3. Open `submission/Model_code_Atharva_Divekar.ipynb` and run all cells. All random steps use seed 42. The notebook writes the scored test file, `Assignment_Test_Atharva_Divekar.csv`, next to itself.

## Data

The case-study data is **not included** in this repository. It was provided for the assessment and contains personal information (names and phone numbers). The scored output files are not included for the same reason.

## Use of AI

AI assistance (Anthropic's Claude, through Claude Code) was used as a tutor and pair programmer, as the assessment allowed. It explained concepts, proposed and reviewed code, ran checks, and helped draft the documentation. All analytical decisions were made by me, and every reported number comes from code in this repository.
