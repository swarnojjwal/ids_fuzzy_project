# Fuzzy Logic Intrusion Detection System (Mamdani vs Sugeno)

A Streamlit dashboard that detects intrusions in the NSL-KDD data set using **only fuzzy logic**
(no machine-learning algorithms). Two independent fuzzy inference systems are implemented from scratch with NumPy:

| | Mamdani | Sugeno (zero-order) |
|---|---|---|
| AND / OR | min / max | product / probabilistic sum |
| Rule output | fuzzy set (Low, Medium, High) | constant (0-100) |
| Implication | min (clip output set) | not needed |
| Aggregation | max | not needed |
| Defuzzification | centroid | weighted average |

## Run

```
pip install -r requirements.txt
streamlit run app.py
```
Everything (preprocessing, both fuzzy models, rules, evaluation, charts, UI, CSS) is inside **app.py**.
The data set is in `data/`. Optional folder `saved_config/` stores rule/membership settings.

## Pages
1. **Dataset Overview** - data quality, class/family distribution, feature relevance, preprocessing.
2. **Rules & Membership** - membership plots, rule table with meanings, rule analysis, editor (enable, weight, outputs, new rules, MF parameters, save/load).
3. **Mamdani vs Sugeno** - all metrics, confusion matrices, ROC/AUC, prediction time, auto-generated verdict.
4. **Prediction** - every dataset feature as an input, real example loader, risk gauges, disagreement highlight, activated rules; CSV batch upload and download.
5. **Live Monitor** - *simulation / uploaded-data monitor* (NOT real network capture): counts, risk trace, history, trends.
6. **Analytics** - score distributions, rule activation heat-map, attack types, trends.

## How the system was built (no data leakage)
* The training file is split 70 % / 30 % (fixed seed). Rules were studied on the 70 % part; decision borders
  (Normal | Suspicious | Malicious) are tuned on the 30 % validation part. **The test file was never used for tuning.**
* Features were ranked by single-feature AUC. `num_failed_logins` (used in the sample project) has zero relevance here.
  Used: serror_rate, rerror_rate, count, same_srv_rate, diff_srv_rate, dst_host_srv_count, dst_host_count, logged_in, src_bytes.
* The rule analysis tab shows, for each rule, what share of the records it fires on are real attacks (training split).
  A rule that fired mostly on normal traffic (D5) was removed; a bad-precision rule (N4) is kept but down-weighted and is flagged "review" on purpose.

## Real results on the official test file (22,544 records; Suspicious+Malicious = alert)
| Metric | Mamdani | Sugeno |
|---|---|---|
| Accuracy | 0.793 | 0.746 |
| Precision | 0.933 | 0.929 |
| Recall | 0.685 | 0.600 |
| F1 | 0.790 | 0.729 |
| Specificity | 0.935 | 0.939 |
| False-positive rate | 0.065 | 0.061 |
| AUC | 0.835 | 0.907 |
| Time per 1000 records | ~1.3 ms | ~0.45 ms |

On the validation split accuracy was 0.924 (Mamdani) and 0.919 (Sugeno).
The drop on the test file is real and expected: the test file holds **17 attack types never seen in training**,
and R2L attacks (login-based, e.g. warezclient, guess_passwd) look like normal sessions in these features
(alert rate only 8 % / 2 %). DoS and Probe attacks are detected well (74-88 %).
Numbers shown in the app are recalculated from the data every time; they are never hard-coded.

## Notes
* Mamdani gives the better accuracy / F1 / recall here; Sugeno has the higher AUC, slightly fewer false alarms and is ~3x faster.
* The Mamdani centroid can never reach exactly 0 or 100 (it stays between about 14 and 86); Sugeno uses the full range.
* All values are normalised to 0..1 before fuzzification, so membership parameters in the editor are on a 0-1 scale.
