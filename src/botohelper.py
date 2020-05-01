# Step 1: Create a Connection to S3


# Boto 3
import boto3
boto3_connection = boto3.resource('s3')

# Check contents of existing buckets


# Boto 3
def print_s3_contents_boto3(connection):
    for bucket in connection.buckets.all():
        for key in bucket.objects.all():
            print(key.key)

# print_s3_contents_boto3(boto3_connection)

# Step 2: Create a Bucket


import os
username = os.environ['USER']
bucket_name = username + "-terminal-from-boto3"
boto3_connection.create_bucket(Bucket=bucket_name)
# Step 3: Make a file


# make a file (could use an existing file, but make one quick from the command line)
echo 'Hello world from boto3!' > hello-boto.txt


# Step 4: Upload the file to s3


s3_client = boto3.client('s3')
#or
s3 = boto3.resource('s3')


s3_client.upload_file('hello-boto.txt', bucket_name, 'hello-remote.txt')


# did it make it?
print_s3_contents_boto3(boto3_connection)
# Step 5: Download a file from s3


s3_client.download_file('ajzcap2', 'energy_dataset.csv', 's3_energy.csv')
print(open('hello-back-again.txt').read())