# Model card

## Intended use

Rank customers by estimated churn risk so a retention team can prioritize a limited outreach capacity. The model is a portfolio demonstration, not a production decision system.

## Model selection

Logistic Regression, Random Forest, and Gradient Boosting are compared using stratified out-of-fold predictions on 80% of the data. Average precision (PR-AUC) is the selection metric because churn is the minority class. The decision threshold is chosen from development predictions only, then evaluated once on the untouched 20% test set.

## Result

Gradient Boosting was selected. On the test set it achieved **0.663 PR-AUC**, **0.845 ROC-AUC**, **0.559 precision**, and **0.725 recall** at the development-selected threshold. Targeting the highest-risk 20% of test customers captured **49.7% of churners**, a **2.48× lift** over random targeting.

## Limitations

- The IBM sample represents a fictional telecommunications company, not a current operating population.
- The data is a static snapshot; temporal drift and real campaign outcomes cannot be tested.
- Associations and feature importance do not establish why a customer churned.
- Protected-class, fairness, privacy, offer cost, and contact-policy reviews are required before real use.
- Risk scores should prioritize human-reviewed outreach, not deny service or determine pricing.

## Monitoring needed in production

Track input drift, churn prevalence, PR-AUC, calibration, recall at the actual outreach capacity, false-positive contact burden, campaign uplift, and subgroup performance. Retrain or pause when these measures deteriorate.

