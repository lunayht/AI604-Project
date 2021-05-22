cd timm-fork-install && pip install -e . && cd ..
mkdir -p data
echo "Downloading BreakHis Dataset from Google Drive (combined.tar.gz) ..."
gdown https://drive.google.com/u/1/uc?id=1bd60s98V3W00R_N7tvOE5pfoDaf8lpfv&export=download
