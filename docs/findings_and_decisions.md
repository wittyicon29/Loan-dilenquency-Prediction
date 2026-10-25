# Loan Delinquency Case Study: Findings and Decision Log

Working record of the analysis so far: Stages 0–4. Exploratory analysis (EDA), feature engineering and the baseline are complete.
Last updated: 2026-10-21.

How entries are labelled:
- **Observed**: a fact read directly from an output.
- **Interpretation**: what we think it means.
- **Assumption**: something we treat as true without proof.
- **Decision**: a choice we made, with its reason.

Results marked *(not yet in notebook)* came from read-only checks run outside `notebooks/eda.ipynb`. Reproduce them in the notebook before relying on them in the report.

---

## 1. Project and environment

| Item | Detail |
|---|---|
| Data files | `data/Assignment_Train.xlsx`, `data/Assignment_Test.xlsx`, `data/Assignment_FeatureDictionary.xlsx` (unchanged) |
| Brief | `docs/DATA SCIENTIST I _ RECRUITMENT CASE STUDY.docx` |
| Notebook | `notebooks/eda.ipynb` (runs from `notebooks/`, so data paths start with `../data/`) |
| Python | 3.14.7 in the project's `.venv` |
| Core packages | pandas 3.0.6, scikit-learn 1.9.1, numpy 2.5.4, scipy 1.18.1, matplotlib 3.11.2, seaborn 0.13.2, openpyxl 3.1.5, pyarrow 25.0.1 |
| Also installed | catboost 1.2.10, lightgbm 4.7.0, xgboost 3.4.1, optuna 5.0.0, mlflow 3.17.1 |

Notes:
- `requirements.txt` has no version pins. Pin them with `pip freeze` before submission.
- `mlflow` is installed but not listed in `requirements.txt`. Either uninstall it or add it.
- pandas 3 behaves differently from older versions in three ways that matter here:
  - text columns load with dtype `str`;
  - Copy-on-Write is always on;
  - `select_dtypes(include="object")` also selects `str` columns unless you pass `exclude="str"`.

## 2. The brief: points that shape the work

- Feature engineering is the main focus. The brief asks for a small set of well-motivated features.
- The brief promises "applicant, loan, repayment, and third-party attributes". The data contains only third-party identity attributes (see §3).
- The 1–100 score must express **relative** risk. It must not be called a calibrated probability unless calibration is shown.
- Deliverables, in one ZIP: `Model code_{FullName}`, `Assignment_Test_{FullName}.csv`, `Report_{FullName}`.
- Inconsistencies in the brief:
  - It refers to `Assignment_Test.csv`, but the supplied file is `.xlsx`.
  - The model-code filename has no extension.
  - It doesn't say whether the 1–100 score must be an integer.

## 3. Feature dictionary

**Observed**
- One sheet with 115 entries. Columns: `feature_name, description, business_meaning, category, feature_group, impact_direction, good_range, bad_range`.
- Every feature comes from one third-party identity provider, "Bureau". Groups: Social (44), Identity (30), Telecom (17), Fraud (15), Digital Footprint (13).
- **The target is not defined anywhere in the dictionary.** There are no applicant, loan or repayment variables.
- Problems within the dictionary itself:
  - `Phone Social Advance.whatsapp` is listed twice.
  - There is a typo: `Emial GIN.FraudCount`.
  - `nameEmailMatch` is described as a 0–100 score but labelled "Binary indicator".
  - The good = 1 direction for `numberHasPortingHistory` and `multiplePhoneAttached` is questionable.
  - For `L2Count`, the business meaning says >50 is risky, while `bad_range` says >5.
- `-1` is documented, for `Email Attributes.digitalAge` only, as meaning "absent from the Bureau network".
- `revocationHistory` only covers Jan 2022 onwards, so a value of `false` partly means "unknown".

**Interpretation**
- These features were designed to detect identity fraud, not credit risk. Expect modest predictive power.
- Treat the dictionary's impact directions and good/bad ranges as the vendor's claims, to be checked against the data.

**Fairness and proxy risks**
- Matrimonial platforms (`shaadi`, `jeevansaathi`) can act as proxies for marital status or community.
- Region encodes location.
- Platform presence partly reflects income and urban access.

## 4. Train/test schema vs dictionary

- Train has 10,000 rows × 94 columns; test has 5,000 × 93. `Target` is the only extra column in train. The column order is identical, and no column name is duplicated.
- UID is unique in both train (1–10,000) and test.
- 34 columns match the dictionary only after fixing spelling. **Decision:** use the data's column names.
- 7 columns in the data are undocumented: `name`, `domainUpdatedDate`, `ndrScore`, `unrScore`, `phoneNumber`, `Email GIN.nameDiversity`, `Mobile GIN.nameDiversity`.
- 29 dictionary entries are absent from the data, including `identityConfidence`, `identityRisk`, `digitalFootprint`, the verification dates and `imsi`.

## 5. Target

- `Target` takes the values 0 and 1, plus 8 cells containing the **literal text "NaT"**. These are scattered (UIDs 3333, 3982, 4530, 4786, 6976, 7092, 8228, 9596).
- After those 8 rows are removed: 9,992 rows, of which 9,114 are 0 and 878 are 1. **The positive rate is 8.79%.**
- A model that predicts "nobody is delinquent" would score 91.2% accuracy. Accuracy is therefore not a useful metric here.
- **Assumption:** `Target = 1` means delinquent.
- **Decision:** drop the 8 unlabelled rows and never impute the target.
- Delinquency rate across UID order:
  - 5 of 20 groups fall outside the 95% chance band, against about 1 expected.
  - A chi-square test of equal rates across groups gives χ² = 35.87, df = 19, p = 0.011.
  - There is no trend (ρ = −0.21, p = 0.39).
  - *(The tests are not yet in the notebook.)*
  - Interpretation: no drift, but possibly a batch effect (hypothesis only).

## 6. The "NaT" missing-value placeholder

- The text `"NaT"` is used as the missing-value placeholder throughout the dataset. **Fix:** load with `pd.read_excel(..., na_values=["NaT"])`. Train and test must both be loaded this way.
- After the fix, the 92 features split as: float64 54, int64 22, str 15, object 1. The object column is `UPIname`, which contains one integer value.
- Missing values come in three kinds: *not applicable*, *lookup failed*, and *truly unknown*.

## 7. Train/holdout split

- **Decision:** split before any feature-vs-target analysis, using a stratified 70:30 split with `random_state=42`.
  - Result: `dev` has 6,994 rows (positive rate 0.0879); `holdout` has 2,998 rows (0.0877).
  - Why 70:30: the holdout keeps about 263 positives, which gives a more stable estimate. Holdout positives help *evaluation*, not learning.
- Rules:
  - Profiling that doesn't use the target may use all of `train`.
  - Feature-vs-target analysis uses `dev` only.
  - The holdout is used once, at the end.
- **Decision:** keep the stratified random split. We found no drift and no evidence of repeat borrowers.
- Lesson: a stale kernel state once double-split the data. Use Restart + Run All, and assert the expected row counts.

## 8. Feature profiling (all of `train`, no target)

### 8.1 Missingness
| Feature | Missing | Note |
|---|---|---|
| `domainUpdatedDate` | 100% | Drop |
| email / phone `paytm` | 99.8% / 99.7% | Drop |
| `revocationMonth` | 98.8% | Structurally missing (only recycled numbers have it). Months run Jan 2022 – Jun 2023. |
| email / phone `skype` | 96.6% / 95.9% | Very sparse |
| `portedFrom*` | 53–58% | Mostly "never ported" |
| 8 `Email GIN.*` numeric columns | 0.4% each | Probably the same rows (not yet verified) |

- Porting consistency check: 53.0% of rows have `numberHasPortingHistory = 0`, which matches the 53.0% missing country code. However, `portedFromNetworkName` is missing in 58.4% of rows, about 5% more.

### 8.2 Constant columns (drop)
`Email GIN.fraud`, `Email GIN.fraudCount`, `Mobile GIN.fraud`, `Mobile GIN.fraudCount`, `Phone Name Attributes.ccrp`, `isPhoneReachable`, `isValidPhoneNumber`, `roaming`, `portedFromNetworkcountryCodeIso2` (always 91 when present).
- **Hypothesis:** every fraud flag is 0, possibly because flagged applicants were filtered out before approval. That would be **selection bias**, and belongs in the limitations section.

### 8.3 The −1 sentinel
- Six features are 59–84% −1 even though pandas reports 0% missing: `phoneNumber` 84.2%, `emailNameDigitalAge` 80.4%, `nameMatchScore` 74.6%, `ndrScore` and `unrScore` 74.0% each, `phoneNameDigitalAge` 58.8%.
- Several more columns have a minimum of −1 *(not yet quantified in the notebook)*.
- `ndrScore` and `unrScore` are −1 in exactly the same 7,399 rows. This suggests one email–name lookup failed for those rows.
- **Assumption:** −1 means "not found / no data" in every field.

### 8.4 Outliers
- `Phone Name Attributes.digitalage` has one row at 36,411 days (≈ 99.7 years). The next-highest value is 6,182 days. The extreme value is invalid.
- Phone ages bunch around 5,190 days (≈ 14.2 years). **Hypothesis:** this is a ceiling set by the data source.

### 8.5 Categorical features
- Some categories are rare, for example `domainRiskLevel` Review (15) and Very Low (6), network IDEA (2) and MTNL (6).
- Some features are near-constant: `domainRiskLevel` is 98.7% Moderate, `domaincorporate` 99.3% No, and network 94.3% Airtel.
- Some labels sit off the scale: `Review`, `NO_RECORDS`, `No Data`. Billing type has both NaN and `unknown`. Capitalisation is inconsistent.
- Crosstab of footprint against recommendation:
  - All 24 NO_RECORDS rows are recommendation `Bad`.
  - The 79 `No Data` rows appear at every footprint level.
  - **No overlap** between the two groups, so they are different populations.
  - The vendor's recommendation appears to be partly derived from footprint.

### 8.6 Duplicates and repeated people
- There are no exact duplicates across all 92 features, and no conflicting labels.
- `phoneNumber` has been truncated to 6 significant digits in the source file. In the 9,992 labelled rows, the 1,578 non-−1 values contain only 913 distinct numbers (99.9% end in ≥ 6 zeros). The 1,016 "shared phone" rows are an artefact of the truncation, not repeated people. *(Reproduced in the submission notebook, section 2b.)*
- Name + UPI name matches 745 rows, mostly common names. Adding region and phone age leaves at most about 8 possible repeat pairs. *(Not yet in the notebook.)*
- **Limitation:** there is no reliable person identifier.

### 8.7 Identifiers and PII
- `UID` is not a predictor.
- `name`, `UPIname` and `phoneNumber` are personal data. Don't print raw rows containing them in the submission.

## 9. Feature vs target (`dev` only)

Each finding below gives group sizes, and draws conclusions only where the 95% confidence intervals don't overlap.

### 9.1 Binary flags
- **Strongest:** `Phone Social Advance.whatsapp = 0` (n = 433): **32.8%** delinquent vs 7.1% for value 1 (lift 3.7). The 433 people are 6% of dev but contain 23% of delinquents.
- Most platform flags point the same way: being absent goes with higher delinquency. The pattern is consistent across email and phone, for example Amazon +6.4 points (email) and +6.7 points (phone), and phone Flipkart +8.3 points.
- No real difference: `jeevansaathi`, `isWABusiness`, `businessNameDetected`, `facebook`, `shaadi`.
- Noise from groups that are too small: `fraudNetwork` (5–8 people with value 1), `paytm`, `skype`.
- For flags, a missing value behaves about like the average.
- `businessNameDetected` (email) and `businessNameDetected` (phone) are identical in 100% of rows.

### 9.2 Engineered feature: platform share
- `email_platform_share` / `phone_platform_share` = platforms registered ÷ platforms known. `paytm` and `skype` are excluded. Computed per row, so there is no leakage and it is identical for test.
- Everyone has at least 12 known email platforms and 7 known phone platforms, so the share is reliable. Medians: 0.35 (email) and 0.46 (phone).
- Delinquency by quintile of share:
  - email: 13.6% → 10.0% → 7.9% → 5.4% → 6.1%
  - phone: 15.6% → 8.7% → 7.1% → 5.7% → 5.8%
  - Lowest vs highest quintile is a real difference. Q4 vs Q5 is not (the intervals overlap). **The relationship is not linear.**

### 9.3 Categorical features
| Feature | Finding |
|---|---|
| `footprint` | HIGH 7.1% < MEDIUM 12.4% ≤ LOW 15.8%. NO_RECORDS has only 17 people. |
| `phoneFootprintStrengthOverall` | Very High 3.9% → High 6.3% → Medium 9.2% → Low 10.3%: clean ordering |
| `numberBillingType` | postpaid 3.8% vs prepaid 9.3% (the dictionary's claim holds). **unknown: 52.8% (19/36).** Missing 11.0%. |
| `emailFinalRecommendation` | "Check" 13.4% vs "Good" 8.4%. The vendor's "Bad" (8.7%) is **not** riskier. |
| `currentNetworkName` | Vodafone 18.8% (69), Jio 11.5% (295), Airtel 8.6% |
| `currentNetworkRegion` | ranges from Chennai 2.7% (364) to Orissa 15.0% (107). **Fairness concern.** |
| `identityTrust` (email/mobile) | High ≈ 7.5% vs Medium ≈ 9.3%. Weak. |
| `domainRiskLevel`, `domaincorporate` | No usable signal |

### 9.4 Numeric features
- **Email ages:** older means safer, then the rate levels off.
  - `Email Attributes.digitalAge`: 10.3% / 12.1% → 5.5% / 4.7%.
  - `Email Name digitalage`: 13.0% → 5.2%.
- **Phone ages:**
  - `Phone Name digitalage` (age of the phone number) shows **no pattern**.
  - `phoneNameDigitalAge` (age of the phone + name pairing) runs 9.7% → 4.7%.
  - So what matters is how long the number has belonged to this person, not how old the number is.
- **"Not found" (−1) is consistently riskier** than found values:
  - `nameMatchScore`: 9.9% when not found, vs 3.6–7.5% when found
  - `ndr` / `unr`: 9.9% vs ~5.5%
  - `emailNameDigitalAge`: 9.7% vs 3.7–6.6%
  - `phoneNameDigitalAge`: 10.1% vs 4.7–9.7%
  - `Email Name digitalage`: 12.5% vs 5.2–13.0%
  - Because "not found" doesn't sit at the low end of the scale, leaving −1 in place would mislead a linear model. **This is evidence for D12 option (c).**
- `uniqueHits`: strong and clean, 14.0% (0–3 hits) → 5.7%.
- `nameEmailMatch`: mild, 10.8% → 7.2%.
- GIN features (l1/l2 Confidence, Count, nameDiversity): little or no signal.
- **Oddities** (recorded as found, without explanation):
  - For `nameMatchScore`, a perfect match (7.5%) is riskier than a partial match (3.8%). The intervals barely touch.
  - `Email Attributes.digitalAge` has 181 days as a very common value, which looks like a vendor default.

### 9.5 Redundancy (Spearman correlation, −1 excluded)
- `Email Name digitalage` ↔ `emailNameDigitalAge`: ρ = 0.91 where both exist. Keep `digitalage` (about 77% coverage vs about 20%).
- `Phone Name digitalage` ↔ `phoneNameDigitalAge`: ρ = 0.69. Keep `phoneNameDigitalAge`, since only it shows a signal.
- `ndr` ↔ `unr`: ρ = −0.32. They share their missing rows but hold different values.
- The two email-age sources agree only moderately (ρ = 0.47).

## 10. Test set vs train

- 5,000 rows, UID unique, same columns as train (minus `Target`). Loaded with `na_values=["NaT"]`.
- **Distribution shift in billing type:**

  | | missing | postpaid | prepaid | unknown |
  |---|---|---|---|---|
  | train | 4.4% | 14.9% | 80.2% | 0.5% |
  | test | 22.8% | 7.4% | 68.9% | 1.0% |

  *(The normalised shares are not yet in the notebook.)*
- −1 is about 5–6 points more common in test for the name-match and age features. Fewer test applicants were found in the vendor network.
- `revocationMonth` runs up to Jun 2023 in train and Aug 2023 in test. **So the Bureau data was pulled no earlier than mid-2023, and test is likely a later batch.**
- **Limitation:** the holdout comes from the training batch, so its score may overstate performance on test.

## 11. Decision log

| # | Status | Item | Reason |
|---|---|---|---|
| D1 | Assumption | The score is used at loan origination (application scoring) | Your hypothesis; nothing in the data contradicts it |
| D2 | Assumption | `Target = 1` means delinquent | Minority class, plausible rate |
| D3 | Decided | Drop the 8 unlabelled rows; never impute the target | No supervised signal |
| D4 | Decided | Load train **and test** with `na_values=["NaT"]` | "NaT" is the placeholder used throughout |
| D5 | Decided | Use the data's column names | 34 columns differ from the dictionary only in spelling |
| D6 | Decided | Stratified 70:30 split, seed 42, made before target analysis | Keeps the holdout unseen and gives a stable estimate |
| D7 | Decided | Use the holdout once; do target analysis on `dev` only | Avoids an optimistic evaluation |
| D8 | Decided | Keep the stratified random split | No drift; no repeat borrowers |
| D9 | Assumption | −1 means "not found / no data" | §8.3 |
| D10 | Decided | UID is not a predictor | Brief |
| D11 | Planned (confirm) | Select the model by cross-validation within `dev` | |
| D12 | **Settled: (c)** | −1 → NaN plus "not found" flags | Cross-validation PR-AUC: (a) 0.233 vs (c) 0.235. The gap (0.002) is far smaller than the fold std (≈ 0.03), so the two are equal. (c) is chosen for interpretability and because of the EDA evidence (§9.4). |
| D13 | Moot | Phone-age outlier | `Phone Name digitalage` is not in the feature set |
| D14 | Candidate | Drop `domainUpdatedDate`, email/phone `paytm` and the 9 constant columns | No information |
| D15 | Recommended (confirm) | Drop `phoneNumber` | Truncated, 84% −1, PII |
| D16 | Open | `name` / `UPIname` (PII): exclude, or derive a name-match feature? | Stage 3 |
| D17 | **Decided** | Ordinal encoding per feature, plus an "unrated" flag where off-scale levels exist (`footprint_no_records`) | Compact, and matches the ordered patterns in §9.3 |
| D18 | **Decided** | Levels with fewer than ~100 people in dev → "other", using fixed lists chosen from dev. Exception: billing keeps `unknown` and `missing` (sensitivity check). | The same encoding applies to dev, holdout and test |
| D19 | Open | Class imbalance: class weights vs resampling vs neither | Compare on validation PR-AUC |
| D20 | Rejected | NO_RECORDS ≈ No Data | No overlap |
| D21 | Finding | WhatsApp = 0 → 32.8% delinquent | §9.1 |
| D22 | Assumption / limitation | Bureau data was collected at application time. If it was collected after outcomes, some flags may be leakage. | Collection date unknown |
| D23–D24 | Finding | Absence from platforms → higher risk; tiny groups are noise | §9.1 |
| D25 | **Built** | `email_platform_share`, `phone_platform_share` | §9.2 |
| D26 | Finding | The platform share's relationship to risk is non-linear (levels off at the top) | §9.2 |
| D27 | Open | Does the phone share add anything beyond the WhatsApp flag? | Check during modelling |
| D28 | Open | Keep `isWABusiness` in the phone share? | |
| D29–D30 | Finding | Footprint and billing patterns | §9.3 |
| D31 | **Limitation** | WhatsApp = 0 and billing "unknown" may reflect events after default | Leakage risk |
| D32 | **Decided** | Sensitivity check: build the model with and without the suspect signals (WhatsApp = 0, billing type) | Leakage can't be ruled out |
| D33 | Finding | The vendor's "Bad" is not riskier; "Check" is | §9.3 |
| D34 | **Decided** | Region is excluded from the main model. One comparison run with region included will report what this fairness choice costs. | Rates range from 2.7% to 15.0%; proxy risk |
| D35 | Finding | Email and phone `businessNameDetected` are identical. Drop one, or both (no signal). | §9.1 |
| D36 | Finding | `domainRiskLevel` and `domaincorporate`: no signal | §9.3 |
| D37 | Finding | Email ages: older is safer. Phone number age: no signal. Phone + name age: older is safer. | §9.4 |
| D38 | Finding | "Not found" is consistently riskier than found | Supports D12 (c) |
| D39 | Oddity | A perfect `nameMatchScore` is riskier than a partial one | §9.4 |
| D40 | Finding | `uniqueHits` is strong and clean | §9.4 |
| D41 | Finding | GIN features carry little or no signal | §9.4 |
| D42 | Oddity | 181 days is a very common email age | §9.4 |
| D43–D46 | Finding | Redundancy results | §9.5 |
| D47–D50 | Finding | Test-set check: billing shift, more "not found", later batch | §10 |
| D51 | **Limitation** | The holdout may overstate performance on test | §10 |
| D52 | **Decided** | Include billing type in the with/without sensitivity check | Shift plus leakage risk |
| D53 | **Decided** | Individual platform flags: keep only WhatsApp, alongside the two shares | Fewer, interpretable features. WhatsApp's signal is far stronger than the rest. |
| D54 | **Decided** | Code structure: `eda.ipynb` (exploration), `fe.ipynb` (feature development and checks), `features.py` (single source of feature code), `modeling.ipynb` (modelling) | One copy of the feature code |
| D55 | Oddity | `Email Name digitalage` equals `phoneNameDigitalAge` exactly in 20.5% of dev rows where both are real, yet the overall correlation is only ρ = 0.22. Keep both. | Possibly a shared "first seen" date at the vendor (hypothesis) |
| D56 | **Decided** | Stratified 5-fold CV on dev, shuffled, seed 42 | About 123 delinquents per validation fold (10 folds would give about 61) |
| D57 | **Decided** | A false negative (missed delinquent) is the costlier error per case | It can lose much of the principal, while a false positive loses the interest margin. False positives are more numerous, so the threshold is a business decision. |
| D58 | **Decided** | Model comparison: PR-AUC (primary) + ROC-AUC. Final evaluation: recall, precision and the confusion matrix at a chosen threshold. | PR-AUC focuses on the rare class (random ≈ 0.088). The AUCs need no threshold. |
| D59 | **Result** | Baseline logistic regression (c): **PR-AUC 0.235 ± 0.034, ROC-AUC 0.700 ± 0.029**. Dummy: 0.088 / 0.500. | §14 |
| D60 | **Decided** | Add an "all usable columns" experiment. Exclusions: UID, PII, region, constants, `domainUpdatedDate`, `revocationMonth`. | Tests whether feature selection left signal unused |
| D61 | **Decided** | Model set: logistic regression, Random Forest, HistGradientBoosting, plus one external library (CatBoost recommended; to confirm) | Two tree families plus one independent check. Avoids selecting on noise. |
| D62 | **Decided** | Feature importance by permutation (drop in PR-AUC), used for insight only, not for selecting features on the same data | Built-in tree importance favours high-cardinality features |
| D63 | **Decided** | Optional ensemble step: a rank average of logistic regression and the best tree model. Tried only if their out-of-fold predictions correlate below ~0.9; kept only if it beats the best single model by more than ~0.03. No stacking. | Complexity has to earn its place |

## 12. Feature-engineering plan

This was the starting point drafted from the EDA findings. It has now been implemented, with the decisions D17, D18, D34 and D53 applied (see §13). The "still to decide" items at the end were resolved as follows: region excluded; ordinal encoding; rare levels grouped; `isWABusiness` kept inside the phone share; individual platform flags dropped except WhatsApp. Class imbalance is still open.

**Engineered features**
| Feature | Built from | Rationale |
|---|---|---|
| `email_platform_share`, `phone_platform_share` | Platform flags (excluding paytm and skype) | Built. Broader presence goes with lower risk. Non-linear. |
| `email_name_not_found` (0/1) | `nameMatchScore`, `ndrScore` and `unrScore` = −1 (the same lookup) | "Not found" is riskier (§9.4). One flag for one lookup. |
| `phone_name_not_found` (0/1) | `phoneNameDigitalAge` = −1 | Same reason |
| `email_not_found` (0/1) | `Email Name digitalage` = −1 | Same reason |
| Ordinal versions of `footprint`, `phoneFootprintStrengthOverall` and `emailFootprintStrength`, plus "unrated" flags | Text levels | The patterns are ordered (§9.3). Each feature gets its own mapping. |

**Raw features to carry forward** (with −1 converted to NaN if option (c) is confirmed)
`whatsapp` (sensitivity check), `Email Attributes.digitalAge`, `Email Name digitalage`, `phoneNameDigitalAge`, `nameMatchScore`, `nameEmailMatch`, `uniqueHits`, `numberBillingType` (sensitivity check), `emailFinalRecommendation`, `currentNetworkName` (grouping rare levels), `numberHasPortingHistory`, first/last name match flags.

**Drop candidates**
- No information: the constant columns, `domainUpdatedDate`, `paytm`, `skype`.
- Redundant or no signal: `emailNameDigitalAge` (redundant), `Phone Name digitalage` (no signal), `businessNameDetected` (both copies), `domainRiskLevel`, `domaincorporate`, GIN features, `portedFrom*`.
- Identifiers and PII: `phoneNumber`, `name`, `UPIname`, `UID`.
- Evidence only, not a feature: `revocationMonth`.

**Still to decide:** region (fairness), ordinal vs one-hot encoding, rare categories, `isWABusiness`, class imbalance, and whether to keep the individual platform flags alongside the shares.

## 13. Implemented feature set (`notebooks/features.py`)

`build_features(data, sentinel_mode="flag")` applies four steps in order and returns UID, the model features, and Target (if present):
1. `handle_sentinels`: creates three "not found" flags, then replaces −1 with NaN in four columns. In "raw" mode it changes nothing.
2. `add_platform_share`: adds the email and phone platform shares.
3. `encode_ordinals`: maps footprint LOW/MEDIUM/HIGH to 0–2 (NO_RECORDS → NaN, plus the flag `footprint_no_records`); maps phone strength Low→Very High to 0–3; maps email strength Very Low→Very High to 0–3 (that scale has no Medium).
4. `clean_categories`: fixed levels per category.
   - billing: prepaid / postpaid / unknown / missing
   - network: Airtel / Reliance Jio / other
   - recommendation: Good / Check / other

**The 22 features in "flag" mode** (19 in "raw" mode, which has no not-found flags):

| Group | Features |
|---|---|
| Numeric (11) | 4 digital-age and score columns with −1 handled, `nameEmailMatch`, `uniqueHits`, 2 platform shares, 3 ordinal codes |
| Flags (8) | `whatsapp`, `numberHasPortingHistory`, first and last name match, 3 not-found flags, `footprint_no_records` |
| Categorical (3) | billing type, network, recommendation |

**Checks that pass:**
- The flag shares match the EDA (0.749 / 0.229 / 0.589).
- No −1 remains in the numeric features.
- The ordinal NaN counts are as expected (17 / 0 / 0). The check caught a "Very low" typo during development.
- The category counts match the EDA.
- dev, holdout and test have identical columns.
- **Every test row is kept, in its original order, with the same UID.**
- `features.py` gives output identical to the notebook version, in both modes and on all three datasets.

Missing values that remain after feature engineering, filled by the imputer inside the pipeline:

| Feature | dev | test |
|---|---|---|
| `nameMatchScore` | 74.9% | 80.8% |
| `phoneNameDigitalAge` | 58.9% | 62.1% |
| `Email Name digitalage` | 22.9% | 27.6% |
| `whatsapp` | 5.8% | 5.2% |

For these rows the imputed value means little, so the "not found" flags carry the information.

## 14. Baseline (Stage 4)

**Pipeline:** a `ColumnTransformer` handles each column group separately:
- numeric → median imputation, then standard scaling
- flags → most-frequent imputation
- categorical → `OneHotEncoder(handle_unknown="ignore")`

This feeds `LogisticRegression(max_iter=1000)`. No class weights yet (D19). All preprocessing is learned inside each CV training fold, so no information leaks from the validation folds.

| Model | PR-AUC (mean ± std) | ROC-AUC (mean ± std) |
|---|---|---|
| Dummy (predicts the base rate) | 0.088 ± 0.000 | 0.500 ± 0.000 |
| Logistic regression, −1 raw (a) | 0.233 ± 0.030 | 0.697 ± 0.026 |
| Logistic regression, −1 flag (c) | **0.235 ± 0.034** | **0.700 ± 0.029** |

**Interpretation**
- The model clearly beats random: PR-AUC is about 2.7× the base rate. Its performance is moderate, which is consistent with identity and fraud signals rather than credit history.
- The difference between (a) and (c) is well inside the fold-to-fold noise, so D12 → (c).
- **Yardstick for Stage 5:** the fold std of PR-AUC is about 0.03. Treat changes smaller than that as noise.

## 14a. Stage 5 plan (agreed 2026-10-21)

**1. A 2 × 2 comparison of feature set and model, plus extra tree models.** Each comparison changes one thing at a time, so we can tell whether a gain comes from the features or from the model.

| Model | Selected 22 features | All usable columns |
|---|---|---|
| Logistic regression | baseline (0.235) | to run |
| Random Forest (bagging) | to run | to run |
| HistGradientBoosting (boosting, scikit-learn) | to run | to run |
| CatBoost (native categorical features, ordered boosting); to confirm vs XGBoost | to run | to run |

- **"All usable columns"** means every column except: UID; PII (`name`, `UPIname`, `phoneNumber`); region columns (D34); constant columns; `domainUpdatedDate`; `revocationMonth` (D60).
- **Why not every tree library:** with a fold std of about 0.03, many near-identical models invite selecting on noise. Tuning many models also risks overfitting the CV folds.
- **Rule:** if models are within the noise, choose the simplest and most interpretable. Logistic regression stays in the running.

**2. Permutation importance on the best "all columns" model**, measured as the drop in PR-AUC when a column is shuffled on validation folds.
- It is used for **insight**: which signals matter, and whether our 22 features miss anything.
- A feature it suggests adding is tested as a normal ablation. Don't select features by importance and then score them on the same dev data.
- Correlated features share credit (e.g. the identical businessName copies).
- Importance shows association, not causation.

**3. Class weighting** (D19).

**4. Sensitivity checks:** without WhatsApp / billing (D32, D52), and with region (D34).

**5. Ablation:** do the platform shares add value beyond WhatsApp? (D27)

**6. Choose the final model** and interpret its coefficients or importances. Only after that, use the holdout once (D7).

**7. Ensemble check (optional, conditional).**
- Random Forest and boosting are already ensembles of trees. This step tests whether combining *different* model types helps.
  1. Compute the correlation between the out-of-fold predictions of logistic regression and the best tree model.
  2. Only if that correlation is below ~0.9 (the models make different mistakes), try a **simple average of their ranks** (soft voting).
  3. Keep the average only if it beats the best single model by more than ~0.03 PR-AUC.
- Stacking is not used: with 615 positives, it adds a learned layer that needs nested CV and is hard to justify.
- Rank averaging fits our output, which is a rank-based 1–100 score, not a calibrated probability.

## 15. Open questions

- Verify that the 35 missing Email GIN rows are the same rows.
- Move the read-only checks marked above into the notebook.
- Housekeeping:
  - pin `requirements.txt` and resolve `mlflow`;
  - correct the notebook note in cell 20 (holdout positives don't "help the model learn");
  - update cell 13's "49 mixed-type" note;
  - update cell 40's note (the share is a decimal, not an integer count);
  - relabel plot 2's legend as "Flag = 1 / 0".

## 16. Interview questions to be ready for

1. Your features come from an identity-verification vendor. Why should they predict delinquency at all?
2. Why didn't you impute the missing targets?
3. A model that predicts nobody is delinquent gets 91% accuracy. What did you use instead, and why?
4. Why did you split before exploring feature–target relationships, and why 70:30?
5. Five of twenty UID groups fell outside the 95% band. Does that mean the data has a problem?
6. Why is "0% missing" misleading for `nameMatchScore`?
7. How did you check for duplicates, and why wasn't the exact-duplicate check enough?
8. A duplicate check flagged 1,016 rows sharing phone numbers. What did you do?
9. Why not encode `Review` or `NO_RECORDS` as 0 or 1 on the risk scale?
10. Every fraud flag is 0. What might that tell you about how the data was collected?
11. Your strongest signal is "no WhatsApp". How do you know it isn't leakage?
12. Groups 4 and 5 have delinquency rates of 5.4% and 6.1%. Is group 5 riskier?
13. Why did you build a platform *share* rather than an "on any platform" flag?
14. The test set has 23% missing billing type, against 4% in train. What does that mean for your results?
15. Why did you use a `Pipeline` instead of imputing and scaling dev before cross-validation?
16. Your PR-AUC is 0.235. Is that good? What do you compare it with?
17. Options (a) and (c) scored almost the same. Why did you choose (c)?
18. Why 5 folds and not 10?
19. Why is PR-AUC your primary metric rather than accuracy or ROC-AUC?
20. Why must the not-found flags be created before replacing −1 with NaN?
21. Why are the category levels fixed lists instead of being recomputed on each dataset?
