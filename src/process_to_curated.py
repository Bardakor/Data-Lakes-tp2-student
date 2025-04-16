import io
import pandas as pd
import boto3
import os # Added os import if needed elsewhere, or remove if not used
from transformers import AutoTokenizer

# Define LocalStack endpoint and credentials
LOCALSTACK_ENDPOINT_URL = "http://localhost:4566"
AWS_ACCESS_KEY_ID = "root"  # Dummy credentials for LocalStack
AWS_SECRET_ACCESS_KEY = "root" # Dummy credentials for LocalStack
AWS_REGION = "us-east-1" # Default region


def tokenize_sequences(bucket_staging, bucket_curated, input_file, output_file, model_name="facebook/esm2_t6_8M_UR50D"):
    """
    Tokenizes protein sequences from the staging bucket and uploads processed data to the curated bucket.

    Steps:
    1. Downloads the staging data file from the staging bucket.
    2. Tokenizes the 'sequence' column using a pre-trained tokenizer.
    3. Stores tokenized sequences alongside other relevant columns.
    4. Uploads the tokenized data to the curated bucket.

    Parameters:
    bucket_staging (str): Name of the staging S3 bucket.
    bucket_curated (str): Name of the curated S3 bucket.
    input_file (str): Name of the input file in the staging bucket.
    output_file (str): Name of the output file in the curated bucket.
    model_name (str): Name of the Hugging Face model to use for tokenization.
    """
    # Initialize S3 client
    s3 = boto3.client(
        's3',
        endpoint_url=LOCALSTACK_ENDPOINT_URL,
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        region_name=AWS_REGION
    )

    # Step 1: Download staging data
    print(f"Downloading {input_file} from staging bucket...")
    response = s3.get_object(Bucket=bucket_staging, Key=input_file)
    data = pd.read_csv(io.BytesIO(response['Body'].read()))

    # Ensure the 'sequence' column exists
    if "sequence" not in data.columns:
        raise ValueError("The input data must contain a 'sequence' column.")

    # Step 2: Load tokenizer
    print(f"Loading tokenizer for {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Step 3: Tokenize sequences
    print("Tokenizing sequences...")
    tokenized_data = []
    for sequence in data["sequence"]:
        tokens = tokenizer(sequence, truncation=True, padding="max_length", max_length=1024, return_tensors="np")
        tokenized_data.append(tokens["input_ids"][0])  # Extract token IDs as a flat array

    # Convert tokenized data into a DataFrame
    tokenized_df = pd.DataFrame(tokenized_data)
    tokenized_df.columns = [f"token_{i}" for i in range(tokenized_df.shape[1])]

    # Merge tokenized sequences with metadata
    print("Merging tokenized sequences with metadata...")
    metadata = data.drop(columns=["sequence"])  # Drop the original sequence column
    processed_data = pd.concat([metadata, tokenized_df], axis=1)

    # Step 4: Save processed data locally
    local_output_path = f"/tmp/{output_file}"
    processed_data.to_csv(local_output_path, index=False)
    print(f"Processed data saved locally at {local_output_path}.")

    # Step 5: Upload to curated bucket
    print(f"Uploading {output_file} to curated bucket...")
    with open(local_output_path, "rb") as f:
        s3.upload_fileobj(f, bucket_curated, output_file)

    print(f"Processed data successfully uploaded to curated bucket as {output_file}.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Process data from staging to curated bucket")
    parser.add_argument("--bucket_staging", type=str, required=True, help="Name of the staging S3 bucket")
    parser.add_argument("--bucket_curated", type=str, required=True, help="Name of the curated S3 bucket")
    parser.add_argument("--input_file", type=str, required=True, help="Name of the input file in the staging bucket")
    parser.add_argument("--output_file", type=str, required=True, help="Name of the output file in the curated bucket")
    parser.add_argument("--model_name", type=str, default="facebook/esm2_t6_8M_UR50D", help="Tokenizer model name")
    args = parser.parse_args()

    tokenize_sequences(args.bucket_staging, args.bucket_curated, args.input_file, args.output_file, args.model_name)
