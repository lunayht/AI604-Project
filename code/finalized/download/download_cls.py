"""
    Download helper for BreakHis Dataset
"""

import gdown

DATA_URL = "https://drive.google.com/u/1/uc?id=1bd60s98V3W00R_N7tvOE5pfoDaf8lpfv&export=download"
OUTPUT = "combined.tar.gz"

if __name__ == "__main__":
    gdown.download(DATA_URL, OUTPUT, quiet=False)