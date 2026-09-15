import os
from dotenv import load_dotenv
from google.cloud import bigquery
import vertexai
from vertexai.language_models import TextEmbeddingModel

load_dotenv()

GCP_PROJECT = os.getenv("GCP_PROJECT_ID")
BQ_DATASET = os.getenv("BQ_DATASET_ID")

vertexai.init(project=GCP_PROJECT, location="asia-south1")
model = TextEmbeddingModel.from_pretrained("text-embedding-004")

RESUME_SUMMARY = """
Senior Data Engineer with 10 years of experience across Wells Fargo, Visa,
American Express, and General Electric. Expertise in GCP, BigQuery, Dataproc,
Cloud Composer, Airflow, Spark, PySpark, Hadoop, HDFS, Hive, Sqoop, Python, SQL.
"""


def get_embedding(text):
    embeddings = model.get_embeddings([text])
    return embeddings[0].values


def fetch_jobs():
    client = bigquery.Client(project=GCP_PROJECT)
    query = f"""
        SELECT title, company, location, url, description, source, search_term
        FROM `{GCP_PROJECT}.{BQ_DATASET}.jobs_raw`
    """
    return list(client.query(query).result())


def create_embeddings_table(client, table_id):
    schema = [
        bigquery.SchemaField("title", "STRING"),
        bigquery.SchemaField("company", "STRING"),
        bigquery.SchemaField("location", "STRING"),
        bigquery.SchemaField("url", "STRING"),
        bigquery.SchemaField("source", "STRING"),
        bigquery.SchemaField("search_term", "STRING"),
        bigquery.SchemaField("embedding", "FLOAT64", mode="REPEATED"),
    ]
    table = bigquery.Table(table_id, schema=schema)
    client.create_table(table, exists_ok=True)


def main():
    client = bigquery.Client(project=GCP_PROJECT)
    table_id = f"{GCP_PROJECT}.{BQ_DATASET}.jobs_embeddings"
    create_embeddings_table(client, table_id)

    print("Embedding resume...")
    resume_vector = get_embedding(RESUME_SUMMARY)
    print(f"Resume embedding length: {len(resume_vector)}")

    jobs = fetch_jobs()
    print(f"Embedding {len(jobs)} job descriptions...")

    rows = []
    for i, job in enumerate(jobs):
        desc = job.description or job.title or ""
        if not desc.strip():
            continue
        try:
            vector = get_embedding(desc[:2000])
        except Exception as e:
            print(f"Skipping job {i} due to error: {e}")
            continue
        rows.append({
            "title": job.title,
            "company": job.company,
            "location": job.location,
            "url": job.url,
            "source": job.source,
            "search_term": job.search_term,
            "embedding": vector,
        })
        if (i + 1) % 10 == 0:
            print(f"  Processed {i + 1}/{len(jobs)}")

    errors = client.insert_rows_json(table_id, rows)
    if errors:
        print(f"Errors inserting rows: {errors}")
    else:
        print(f"Inserted {len(rows)} job embeddings into {table_id}")


if __name__ == "__main__":
    main()
