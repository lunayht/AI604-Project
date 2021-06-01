#!/bin/bash
DATASET_PATH=dataset
mkdir -p $DATASET_PATH

echo "Download BreakHis (Classification) Dataset ..."
python download/download_cls.py
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
python download/train_test_split.py