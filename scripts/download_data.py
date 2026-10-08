from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile
from io import BytesIO

ROOT = Path(__file__).resolve().parents[1]
URL = "https://www.kaggle.com/api/v1/datasets/download/blastchar/telco-customer-churn"
DESTINATION = ROOT / "data/raw"


def main() -> None:
    DESTINATION.mkdir(parents=True, exist_ok=True)
    with urlopen(URL) as response:
        archive = BytesIO(response.read())
    with ZipFile(archive) as bundle:
        bundle.extract("WA_Fn-UseC_-Telco-Customer-Churn.csv", DESTINATION)
    print(f"Downloaded dataset to {DESTINATION}")


if __name__ == "__main__":
    main()

