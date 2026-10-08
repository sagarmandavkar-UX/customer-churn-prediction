from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks/customer_churn_analysis.ipynb"


def main() -> None:
    notebook = nbf.v4.new_notebook()
    notebook["metadata"] = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12"},
    }
    cells = [
        nbf.v4.new_markdown_cell(
            "# Customer Churn Prediction\n\n"
            "## tl;dr\n\n"
            "Gradient Boosting produced **0.663 test PR-AUC** and **0.845 ROC-AUC**. "
            "At the development-selected threshold it recovered **72.5% of churners**. "
            "If outreach is limited to the highest-risk 20% of customers, the model captures **49.7% of churners** "
            "with **2.48× lift** over random targeting."
        ),
        nbf.v4.new_markdown_cell(
            "## Context & Methods\n\n"
            "The goal is risk ranking for retention outreach. Logistic Regression, Random Forest, and Gradient Boosting "
            "are compared using stratified out-of-fold average precision on the development partition. The selected model "
            "is evaluated once on an untouched stratified test set.\n\n"
            "### Key Assumptions\n\n"
            "- `Churn = Yes` identifies a customer who left in the last month.\n"
            "- The static sample is used for classification, not causal inference.\n"
            "- A 20% outreach budget is illustrative and should be replaced by real campaign capacity and economics."
        ),
        nbf.v4.new_code_cell(
            "from pathlib import Path\n"
            "import json, os, subprocess, sys\n"
            "import pandas as pd\n"
            "from IPython.display import Image, display\n\n"
            "ROOT = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()\n"
            "env = os.environ.copy()\n"
            "env['PYTHONPATH'] = str(ROOT / 'src')\n"
            "subprocess.run([sys.executable, str(ROOT / 'scripts/run_analysis.py')], cwd=ROOT, env=env, check=True, capture_output=True)\n"
            "summary = json.loads((ROOT / 'reports/summary.json').read_text())\n"
            "summary"
        ),
        nbf.v4.new_markdown_cell("## Data\n\nThe IBM sample contains 7,043 fictional telco customers. Eleven blank `TotalCharges` values are converted to missing and imputed inside each training fold, preventing leakage."),
        nbf.v4.new_code_cell(
            "data = pd.read_csv(ROOT / 'data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv')\n"
            "pd.DataFrame({'rows': [len(data)], 'columns': [data.shape[1]], 'duplicate_customer_ids': [data.customerID.duplicated().sum()], 'churn_rate': [(data.Churn == 'Yes').mean()]})"
        ),
        nbf.v4.new_markdown_cell("## Results\n\n### Contract pattern\n\nMonth-to-month contracts have the highest observed churn rate. This is an association, not evidence that contract type causes churn."),
        nbf.v4.new_code_cell("display(Image(filename=ROOT / 'reports/figures/churn_by_contract.png', width=800))"),
        nbf.v4.new_markdown_cell("### Model discrimination\n\nPrecision–recall curves emphasize performance on the minority churn class; the dashed line is the test-set churn prevalence."),
        nbf.v4.new_code_cell("display(Image(filename=ROOT / 'reports/figures/precision_recall_curves.png', width=800))\npd.read_csv(ROOT / 'reports/model_comparison.csv').round(3)"),
        nbf.v4.new_markdown_cell("### Model interpretation\n\nPermutation importance measures the decrease in test average precision when a raw input is shuffled. It is predictive importance, not a causal explanation."),
        nbf.v4.new_code_cell("display(Image(filename=ROOT / 'reports/figures/feature_importance.png', width=800))\npd.read_csv(ROOT / 'reports/feature_importance.csv').head(10).round(4)"),
        nbf.v4.new_markdown_cell("### Retention targeting\n\nRisk ranking is more actionable than accuracy. The curve shows how much observed churn is captured as outreach capacity expands."),
        nbf.v4.new_code_cell("display(Image(filename=ROOT / 'reports/figures/targeting_gains.png', width=800))\npd.read_csv(ROOT / 'reports/targeting_gains.csv').round(3)"),
        nbf.v4.new_markdown_cell(
            "## Takeaways\n\n"
            "- Use the model to prioritize a human-reviewed retention list, not as an automatic adverse decision.\n"
            "- A 20% outreach budget captures roughly half of observed churners in the test set.\n"
            "- Validate on a real, time-ordered customer cohort and measure incremental campaign uplift before deployment.\n"
            "- Source: [Kaggle — Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)."
        ),
    ]
    notebook["cells"] = cells
    NOTEBOOK.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(notebook, NOTEBOOK)
    print(NOTEBOOK)


if __name__ == "__main__":
    main()

