# Data

`data/raw/credit_default.csv` is created by `python scripts/download_data.py` and is **not committed**.

- **Dataset:** Default of Credit Card Clients
- **Source:** UCI Machine Learning Repository, dataset id 350
  https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients
- **Creator:** I-Cheng Yeh (2009). DOI: https://doi.org/10.24432/C55S3H
- **License:** Creative Commons Attribution 4.0 International (CC BY 4.0). Credit the creator when you share or adapt it.
- **Size (per UCI):** 30,000 rows, 23 features, binary target `default payment next month`. UCI lists no missing values.
- **Region and period:** Taiwan, April to September 2005.

## Schema used in this project

| Column | Meaning | Used as model input? |
|---|---|---|
| LIMIT_BAL | Credit limit (NT$) | yes |
| SEX, AGE, MARRIAGE | Demographics | **no, audit only** |
| EDUCATION | 1 graduate, 2 university, 3 high school, 4 other | yes |
| PAY_1..PAY_6 | Repayment status, month 1 = September 2005 (most recent) | yes |
| BILL_AMT1..6 | Bill statement amount | yes |
| PAY_AMT1..6 | Amount paid | yes |
| default | Target, 1 = default next month | target |

The original file names the first repayment column `PAY_0` and skips `PAY_1`; `creditguard.data` renames it to `PAY_1` so months run 1 to 6.
Column names also differ between the UCI download (`X1..X23`, `Y`) and the Kaggle mirror. `standardize_columns` handles both.
