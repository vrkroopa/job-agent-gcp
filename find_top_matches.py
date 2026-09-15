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


def get_resume_embedding():
    embeddings = model.get_embeddings([RESUME_SUMMARY])
    return embeddings[0].values


def find_top_matches(resume_vector, top_n=10):
    client = bigquery.Client(project=GCP_PROJECT)
    table_id = f"`{GCP_PROJECT}.{BQ_DATASET}.jobs_embeddings`"

    query = f"""
        SELECT
            base.title,
            base.company,
            base.location,
            base.url,
            base.source,
            distance
        FROM VECTOR_SEARCH(
            TABLE {table_id},
            'embedding',
            (SELECT @resume_vector AS embedding),
            top_k => {top_n},
            distance_type => 'COSINE'
        )
        ORDER BY distance ASC
    """

    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ArrayQueryParameter("resume_vector", "FLOAT64", resume_vector)
        ]
    )

    results = client.query(query, job_config=job_config).result()
    return list(results)


def main():
    print("Generating resume embedding...")
    resume_vector = get_resume_embedding()

    print("Searching for top matching jobs...")
    matches = find_top_matches(resume_vector, top_n=10)

    print(f"\nTop {len(matches)} matches:\n")
    for i, row in enumerate(matches, 1):
        print(f"{i}. {row.title} — {row.company} ({row.location})")
        print(f"   Source: {row.source} | Similarity distance: {row.distance:.4f}")
        print(f"   {row.url}\n")


if __name__ == "__main__":
    main()
