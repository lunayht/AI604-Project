"""
Run this script to download CrowdsourcingDataset-Amgadetal2019 from S3.
"""

import boto3
import os

AWS_ACCESS_KEY_ID = "AKIASJXRCGLKGJYCUFDL"
AWS_SECRET_KEY = "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"
BUCKET_NAME = "ai604"
PREFIXS = [
    "dataset/images/",
    "dataset/masks/",
    "dataset/meta/"
]

s3 = boto3.resource('s3',aws_access_key_id=AWS_ACCESS_KEY_ID,aws_secret_access_key=AWS_SECRET_KEY)
bucket = s3.Bucket(BUCKET_NAME)

for prefix in PREFIXS:
    objects = bucket.objects.filter(Prefix=prefix)
    total = 3 if (prefix == "dataset/meta/") else 151
    
    if not os.path.isdir(prefix):
        os.makedirs(prefix)
    
    print("\nStart downloading " + prefix + "...")
    for i, obj in enumerate(objects):
        path, filename = os.path.split(obj.key)
        if filename == '':
            continue
        bucket.download_file(obj.key, obj.key)
        progress = i*100/total
        print("Progress: %.2f"%progress + " | Data: " + filename)