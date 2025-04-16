# Data Lakes & Data Integration 

This repository is designed to help students learn about data lakes and data integration pipelines using Python, Docker, LocalStack, and DVC. Follow the steps below to set up and run the pipeline.

---

## 1. Prerequisites

### Install Docker
Docker is required to run LocalStack, a tool simulating AWS services locally.

1. Install Docker:
```bash
sudo apt update
sudo apt install docker.io
```

2. Verify Docker installation:
```bash
docker --version
```

3. Install AWS CLI
AWS CLI is used to interact with LocalStack S3 buckets.

```bash
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip
sudo ./aws/install
```

4. Verify that the installation worked

```bash
aws --version
```

5. Configure AWS CLI for LocalStack

```bash
aws configure
```

Enter the following values:
* AWS Access Key ID: root
* AWS Secret Access Key: root
* Default region name: us-east-1
* Default output format: json

**Note:** This configuration is primarily used by the AWS CLI itself (e.g., for creating buckets) and by DVC to interact with the LocalStack remote. The Python scripts in this project are configured to connect to LocalStack directly using the endpoint URL and dummy credentials defined within the code.

6. Create LocalStack S3 buckets:

```bash
aws --endpoint-url=http://localhost:4566 s3 mb s3://raw
aws --endpoint-url=http://localhost:4566 s3 mb s3://staging
aws --endpoint-url=http://localhost:4566 s3 mb s3://curated
```

7. Install DVC
DVC is used for data version control and pipeline orchestration.

```bash
pip install dvc
```

```bash
dvc remote add -d localstack-s3 s3://
dvc remote modify localstack-s3 endpointurl http://localhost:4566
```

## 2. Repository Setup
Install Python Dependencies

```bash
pip install -r build/requirements.txt
```

Start LocalStack

```bash
bash scripts/start_localstack.sh
```

Download the Dataset

```bash
pip install kaggle 
kaggle datasets download googleai/pfam-seed-random-split
```

Move the dataset into a data/raw folder.

## 3. Running the Pipeline

Unpack the dataset into a single CSV file in the raw bucket:

```bash
python build/unpack_to_raw.py --input_dir data/raw --bucket_name raw --output_file_name combined_raw.csv
```

Preprocess the data to clean, encode, split into train/dev/test, and compute class weights:

```bash
python src/preprocess_to_staging.py --bucket_raw raw --bucket_staging staging --input_file combined_raw.csv --output_prefix preprocessed
```

Prepare the data for model training by tokenizing sequences:

```bash
python src/process_to_curated.py --bucket_staging staging --bucket_curated curated --input_file preprocessed_train.csv --output_file tokenized_train.csv
```

## 4. Running the Entire Pipeline with DVC
The pipeline stages are defined in dvc.yaml. Run the pipeline using:

```bash
dvc repro
```

## 5. Notes
Ensure LocalStack is running before executing any pipeline stage.
This pipeline illustrates a basic ETL flow for a data lake, preparing data from raw to curated for AI model training.
If you encounter any issues, ensure Docker, AWS CLI, and DVC are correctly configured.


python build/unpack_to_raw.py --input_dir ./data/raw --bucket_name raw --output_file_name combined_raw.csv   

```bash
python build/unpack_to_raw.py --input_dir ./data/raw --bucket_name raw --output_file_name combined_raw.csv   
```

```bash
python src/preprocess_to_staging.py --bucket_raw raw --bucket_staging staging --input_file combined_raw.csv --output_prefix preprocessed 
```
