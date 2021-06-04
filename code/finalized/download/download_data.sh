#!/bin/bash
DATASET_PATH=dataset
mkdir -p $DATASET_PATH

echo "Download BreakHis (Classification) Dataset ..."
python -c "import gdown; gdown.download('https://drive.google.com/u/1/uc?id=1bd60s98V3W00R_N7tvOE5pfoDaf8lpfv&export=download', 'combined.tar.gz', quiet=False)"
tar -xvxf combined.tar.gz --directory $DATASET_PATH
mv $DATASET_PATH/combined $DATASET_PATH/breakhis && rm combined.tar.gz

mkdir -p $DATASET_PATH/crowdsourcing

echo "Downloading CrowdSourcing (Segmentation) Dataset ..."
python download/download_seg.py
mv $DATASET_PATH/annotations $DATASET_PATH/crowdsourcing
mv $DATASET_PATH/images $DATASET_PATH/crowdsourcing
mv $DATASET_PATH/masks $DATASET_PATH/crowdsourcing
mv $DATASET_PATH/meta $DATASET_PATH/crowdsourcing

echo "Preprocessing CrowdSourcing (Segmentation) Dataset ..."
python download/preprocess_seg.py