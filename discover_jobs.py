import os
import json
import requests
from datetime import datetime
from dotenv import load_dotenv
from google.cloud import storage, bigquery

load_dotenv()

USAJOBS_API_KEY = os.getenv("USAJOBS_API_KEY")
USAJOBS_EMAIL = os.getenv("USAJOBS_EMAIL")
ADZUNA_APP_ID = os.getenv("ADZUNA_APP_ID")
ADZUNA_APP_KEY = os.getenv("ADZUNA_APP_KEY")
GCS_BUCKET = os.getenv("GCS_BUCKET_NAME")
BQ_DATASET = os.getenv("BQ_DATASET_ID")
GCP_PROJECT = os.getenv("GCP_PROJECT_ID")

SEARCH_TERMS = ["Data Engineer", "Hadoop Developer", "GCP Data Engineer"]


def fetch_usajobs(keyword):
    url = "https://data.usajobs.gov/api/search"
    headers = {
        "Host": "data.usajobs.gov",
        "User-Agent": USAJOBS_EMAIL,
        "Authorization-Key": USAJOBS_API_KEY,
    }
    params = {"Keyword": keyword, "ResultsPerPage": 25}
    resp = requests.get(url, headers=headers, params=params)
    resp.raise_for_status()
    data = resp.json()
    jobs = []
    for item in data.get("SearchResult", {}).get("SearchResultItems", []):
        job = item.get("MatchedObjectDescriptor", {})
        jobs.append({
            "source": "usajobs",
            "title": job.get("PositionTitle"),
            "company": job.get("OrganizationName"),
            "location": job.get("PositionLocationDisplay"),
            "url": job.get("PositionURI"),
            "description": job.get("UserArea", {}).get("Details", {}).get("JobSummary", ""),
            "posted_date": job.get("PublicationStartDate"),
            "search_term": keyword,
        })
    return jobs


def fetch_adzuna(keyword):
    url = "https://api.adzuna.com/v1/api/jobs/us/search/1"
    params = {
        "app_id": ADZUNA_APP_ID,
        "app_key": ADZUNA_APP_KEY,
        "what": keyword,
        "results_per_page": 25,
    }
    resp = requests.get(url, params=params)
    resp.raise_for_status()
    data = resp.json()
    jobs = []
    for item in data.get("results", []):
        jobs.append({
            "source": "adzuna",
            "title": item.get("title"),
            "company": item.get("company", {}).get("display_name"),
            "location": item.get("location", {}).get("display_name"),
            "url": item.get("redirect_url"),
            "description": item.get("description"),
            "posted_date": item.get("created"),
            "search_term": keyword,
        })
    return jobs


def upload_to_gcs(all_jobs):
    client = storage.Client(project=GCP_PROJECT)
    bucket = client.bucket(GCS_BUCKET)
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    filename = f"raw/jobs_{timestamp}.json"
    blob = bucket.blob(filename)
    blob.upload_from_string(json.dumps(all_jobs, indent=2), content_type="application/json")
    print(f"Uploaded {len(all_jobs)} jobs to gs://{GCS_BUCKET}/{filename}")
    return filename


def load_to_bigquery(all_jobs):
    client = bigquery.Client(project=GCP_PROJECT)
    table_id = f"{GCP_PROJECT}.{BQ_DATASET}.jobs_raw"

    schema = [
        bigquery.SchemaField("source", "STRING"),
        bigquery.SchemaField("title", "STRING"),
        bigquery.SchemaField("company", "STRING"),
        bigquery.SchemaField("location", "STRING"),
        bigquery.SchemaField("url", "STRING"),
        bigquery.SchemaField("description", "STRING"),
        bigquery.SchemaField("posted_date", "STRING"),
        bigquery.SchemaField("search_term", "STRING"),
    ]

    job_config = bigquery.LoadJobConfig(
        schema=schema,
        write_disposition="WRITE_APPEND",
        source_format=bigquery.SourceFormat.NEWLINE_DELIMITED_JSON,
    )

    ndjson_data = "\n".join(json.dumps(job) for job in all_jobs)
    load_job = client.load_table_from_file(
        file_obj=__import__("io").StringIO(ndjson_data),
        destination=table_id,
        job_config=job_config,
    )
    load_job.result()
    print(f"Loaded {len(all_jobs)} rows into {table_id}")


def main():
    all_jobs = []
    for term in SEARCH_TERMS:
        print(f"Searching USAJobs for: {term}")
        all_jobs.extend(fetch_usajobs(term))
        print(f"Searching Adzuna for: {term}")
        all_jobs.extend(fetch_adzuna(term))

    print(f"Total jobs fetched: {len(all_jobs)}")

    if all_jobs:
        upload_to_gcs(all_jobs)
        load_to_bigquery(all_jobs)
    else:
        print("No jobs found.")


if __name__ == "__main__":
    main()
