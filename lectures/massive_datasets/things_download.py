# file: things_download.py
import boto3, os
s3 = boto3.client('s3', region_name='us-east-1')
bucket = 'things-data'
for obj in s3.list_objects_v2(Bucket=bucket, Prefix='MEG/')['Contents']:
    s3.download_file(bucket, obj['Key'], os.path.basename(obj['Key']))