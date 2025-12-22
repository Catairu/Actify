import gdown
import sys
from pathlib import Path
from zipfile import ZipFile

# ==============================================================================
# 1. CONFIGURATION AND PATHS
# ==============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
zip_path = DATA_DIR / "data.zip"
url = "https://drive.google.com/uc?id=1GSCASKZxyNsEQEVwCpbqlYQjBK23oeck"

sentinel_file = DATA_DIR / "UCI HAR Dataset"

# ==============================================================================
# 2. SCRIPT EXECUTION
# ==============================================================================

if sentinel_file.exists():
    print("Dataset found. Skipping setup.")
    sys.exit()

DATA_DIR.mkdir(parents=True, exist_ok=True)

# Download the dataset and extract the zip file contents
gdown.download(url, str(zip_path), quiet=False)
with ZipFile(zip_path, "r") as zip_ref:
    zip_ref.extractall(DATA_DIR)

print("Extraction complete! Removing the zip file...")
zip_path.unlink()

print("Setup completed successfully!")
