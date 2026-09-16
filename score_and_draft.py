import os
import json
from datetime import datetime
from dotenv import load_dotenv
from google.cloud import bigquery
import vertexai
from vertexai.language_models import TextEmbeddingModel
import anthropic

load_dotenv()

GCP_PROJECT = os.getenv("GCP_PROJECT_ID")
BQ_DATASET = os.getenv("BQ_DATASET_ID")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

vertexai.init(project=GCP_PROJECT, location="asia-south1")
embed_model = TextEmbeddingModel.from_pretrained("text-embedding-004")
claude = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

RESUME_SUMMARY = """
Senior Data Engineer with 10 years of experience across Wells Fargo, Visa,
American Express, and General Electric. Expertise in GCP, BigQuery, Dataproc,
Cloud Composer, Airflow, Spark, PySpark, Hadoop, HDFS, Hive, Sqoop, Python, SQL.
"""

SCORE_THRESHOLD = 70


def get_resume_embedding():
    embeddings = embed_model.get_embeddings([RESUME_SUMMARY])
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
    return list(client.query(query, job_config=job_config).result())


def get_job_description(title, company):
    """Fetch the full description from jobs_raw since jobs_embeddings doesn't store it."""
    client = bigquery.Client(project=GCP_PROJECT)
    query = f"""
        SELECT description
        FROM `{GCP_PROJECT}.{BQ_DATASET}.jobs_raw`
        WHERE title = @title AND company = @company
        LIMIT 1
    """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("title", "STRING", title),
            bigquery.ScalarQueryParameter("company", "STRING", company),
        ]
    )
    results = list(client.query(query, job_config=job_config).result())
    return results[0].description if results else ""


def score_job(title, company, description):
    prompt = f"""You are evaluating how well a candidate's resume matches a job posting.

RESUME SUMMARY:
{RESUME_SUMMARY}

JOB TITLE: {title}
COMPANY: {company}
JOB DESCRIPTION:
{description[:3000]}

Respond ONLY with a JSON object in this exact format, no other text:
{{"score": <integer 0-100>, "reasoning": "<one sentence explanation>"}}
"""
    response = claude.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=200,
        messages=[{"role": "user", "content": prompt}],
    )
    text = response.content[0].text.strip()
    text = text.replace("```json", "").replace("```", "").strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"score": 0, "reasoning": "Could not parse response"}


def draft_application(title, company, description):
    prompt = f"""Write a concise, tailored cover letter opening paragraph (3-4 sentences)
for this job, based on the candidate's resume. Be specific about relevant skills,
not generic.

RESUME SUMMARY:
{RESUME_SUMMARY}

JOB TITLE: {title}
COMPANY: {company}
JOB DESCRIPTION:
{description[:3000]}

Write only the paragraph, no preamble or explanation.
"""
    response = claude.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=300,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text.strip()


def main():
    print("Generating resume embedding...")
    resume_vector = get_resume_embedding()

    print("Finding top matches...")
    matches = find_top_matches(resume_vector, top_n=10)

    os.makedirs("applications", exist_ok=True)
    report_lines = [f"# Daily Job Report — {datetime.now().strftime('%Y-%m-%d')}\n"]

    for i, job in enumerate(matches, 1):
        print(f"\n[{i}/{len(matches)}] Scoring: {job.title} — {job.company}")
        description = get_job_description(job.title, job.company)

        if not description.strip():
            print("  No description found, skipping scoring.")
            continue

        result = score_job(job.title, job.company, description)
        score = result.get("score", 0)
        reasoning = result.get("reasoning", "")
        print(f"  Score: {score}/100 — {reasoning}")

        report_lines.append(f"## {i}. {job.title} — {job.company}")
        report_lines.append(f"- Location: {job.location}")
        report_lines.append(f"- Source: {job.source}")
        report_lines.append(f"- URL: {job.url}")
        report_lines.append(f"- **Fit score: {score}/100** — {reasoning}\n")

        if score >= SCORE_THRESHOLD:
            print("  Drafting application...")
            draft = draft_application(job.title, job.company, description)
            safe_name = "".join(c if c.isalnum() else "_" for c in f"{job.company}_{job.title}")[:60]
            draft_path = f"applications/{safe_name}.txt"
            with open(draft_path, "w") as f:
                f.write(f"{job.title} — {job.company}\n{job.url}\n\n{draft}")
            report_lines.append(f"  Draft saved: `{draft_path}`\n")
            print(f"  Draft saved to {draft_path}")

    report_filename = f"daily_report_{datetime.now().strftime('%Y-%m-%d')}.md"
    with open(report_filename, "w") as f:
        f.write("\n".join(report_lines))
    print(f"\nReport saved to {report_filename}")


if __name__ == "__main__":
    main()
