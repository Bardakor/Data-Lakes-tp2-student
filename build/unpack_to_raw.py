import os
import pandas as pd
import boto3

# Define LocalStack endpoint and credentials
LOCALSTACK_ENDPOINT_URL = "http://localhost:4566"
AWS_ACCESS_KEY_ID = "root"  # Dummy credentials for LocalStack
AWS_SECRET_ACCESS_KEY = "root" # Dummy credentials for LocalStack
AWS_REGION = "us-east-1" # Default region


def unpack_data(input_dir, bucket_name, output_file_name):
    """
    Unpacks and combines multiple CSV files from train, test, and dev subfolders into a single CSV file,
    then uploads the combined file to the specified S3 bucket.

    Parameters:
    input_dir (str): Path to the directory containing the train, test, and dev subfolders.
    bucket_name (str): Name of the S3 bucket to upload the combined file to.
    output_file_name (str): Name of the combined CSV file to be uploaded to S3.
    """
    s3 = boto3.client(
        's3',
        endpoint_url=LOCALSTACK_ENDPOINT_URL,
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        region_name=AWS_REGION
    )
    data_frames = []

    # Iterate through train, test, and dev subfolders
    for subfolder in ['train', 'test', 'dev']:
        subfolder_path = os.path.join(input_dir, subfolder)
        if os.path.exists(subfolder_path) and os.path.isdir(subfolder_path):
            for file_name in os.listdir(subfolder_path):
                file_path = os.path.join(subfolder_path, file_name)
                print(f"Reading {file_path}")
                data = pd.read_csv(
                    file_path,
                    names=['sequence', 'family_accession', 'sequence_name', 'aligned_sequence', 'family_id']
                )
                data_frames.append(data)
        else:
            print(f"Subfolder {subfolder_path} does not exist or is not a directory.")

    # Combine all data frames into a single data frame
    if data_frames:
        combined_data = pd.concat(data_frames, ignore_index=True)
        print("All files combined successfully.")

        # Save the combined data to a CSV file
        combined_csv_path = f"/tmp/{output_file_name}"  # Save locally before uploading
        combined_data.to_csv(combined_csv_path, index=False)
        print(f"Combined file saved locally at {combined_csv_path}.")

        # Upload the combined file to the S3 bucket
        s3.upload_file(combined_csv_path, bucket_name, output_file_name)
        print(f"Uploaded combined file to bucket '{bucket_name}' with name '{output_file_name}'.")
    else:
        print("No valid files found to process.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Unpack, combine, and upload protein data")
    parser.add_argument("--input_dir", type=str, required=True, help="Path to input directory")
    parser.add_argument("--bucket_name", type=str, required=True, help="Name of the S3 bucket")
    parser.add_argument("--output_file_name", type=str, required=True, help="Name of the output file for S3")
    args = parser.parse_args()

    unpack_data(args.input_dir, args.bucket_name, args.output_file_name)
