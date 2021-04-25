#!/bin/sh

echo Installing Boto3...
pip3 install boto3 --user

echo Download CrowdSourcing Dataset: Images, Masks, Metadata...
python3 download_from_s3.py