# Data

This repository uses the **Telco Customer Churn** sample dataset distributed on Kaggle and attributed there to IBM Sample Data Sets.

- Source: https://www.kaggle.com/datasets/blastchar/telco-customer-churn
- File: `WA_Fn-UseC_-Telco-Customer-Churn.csv`
- Snapshot used: Kaggle dataset version downloaded 2026-10-08
- Size: 7,043 customers and 21 columns
- Kaggle license label: **Data files © Original Authors**
- IBM context: the records describe a fictional telecommunications company; `Churn` identifies customers who left during the last month.

The source CSV is downloaded by `python scripts/download_data.py` and is not committed. The public Kaggle download does not require credentials. Do not infer real-world customer behavior or protected-class effects from this fictional sample.

Refresh the file with:

```bash
python scripts/download_data.py
```
