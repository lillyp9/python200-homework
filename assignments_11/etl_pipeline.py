import os
import json
from datetime import date
import requests
from dotenv import load_dotenv
from openai import OpenAI
from azure.storage.blob import ContainerClient
from azure.identity import DefaultAzureCredential
from prefect import flow, task, get_run_logger

#===Constants and Configurations ====
load_dotenv("/home/lilly/python200-homework/.env")

ACCOUNT_URL = "https://odalissctd2026sa.blob.core.windows.net"
CONTAINER = "pipeline-data"
MAX_RECORDS = 24

SYSTEM_PROMPT = (
    "You are classifying hourly weather conditions for outdoor running. "
    "Given a temperature in Celsius and a precipitation amount in mm, "
    "classify the conditions as exactly one of: good, marginal, or bad. "
    "Reply with that one word -- only no punctuation, no explanation."
)

VALID_LABELS = {"good", "marginal", "bad"}

#======== ETL Pipeline Taks and Flow =====
@task(retries=2, retry_delay_seconds=10)
def extract(latitude: float, longitude: float) -> dict:
    url = (
        f"https://api.open-meteo.com/v1/forecast?latitude={latitude}&longitude={longitude}&hourly=temperature_2m,precipitation&forecast_days=7"
    )
    #request URl and raises statuses (check for webisite errors and stops if it fails) 
    response = requests.get(url)
    response.raise_for_status()
    print(f"Extracted forecast data for ({latitude}, {longitude})")
    return response.json()


@task
def transform(data: dict, max_records: int) -> list:
    #connects to OpenAI  using API key 
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    hourly = data["hourly"] #gets the hourly data from the response 

    records = [] #placeholder
     #loops through hourly weather ata , stops early if it hits max 
    for i in range(min(max_records, len(hourly["time"]))):
        records.append({
            "time": hourly["time"][i],
            "temperature_2m": hourly["temperature_2m"][i],
            "precipitation": hourly["precipitation"][i],
        })

    enriched = [] #placeholder 
     #classify each record
    for i, record in enumerate(records):
        user_msg = (
            f"Temperature: {record['temperature_2m']}C, "
            f"Precipitation: {record['precipitation']}mm"
        )
        #sends the response to the OpenAI API and gets the label and reads the system prompt 
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ]
        )
        #clean the text from the response , removing extra whitespace and make it lowercase  
        raw_label = response.choices[0].message.content.strip().lower()
        #Check if the response is one of the valid labels.
        label = raw_label if raw_label in VALID_LABELS else "unknown"
        #Saves the final weather record with the new label
        enriched.append({**record, "conditions": label})
        if (i + 1) % 6 == 0:
            print(f"  Classified {i + 1}/{len(records)} records")

    print(f"Transform complete: {len(enriched)} records enriched")
    return enriched


@task
def load(records: list, blob_path: str) -> None:
    logger = get_run_logger()
    #uses DefaultAzureCredentials to get the credentials from the cloud.
    credential = DefaultAzureCredential()
    container = ContainerClient(ACCOUNT_URL, CONTAINER, credential=credential)
     #convert the records to json and encode it to bytes
    payload = json.dumps(records).encode("utf-8")
    #Upoad the file to a specified blob path in Azure Storage, overwriting if it already exists.
    container.upload_blob(blob_path, payload, overwrite=True)
    logger.info(f"Loaded {len(records)} records to {blob_path}")
    print(f"Loaded {len(payload)} bytes to {blob_path}")


@flow(log_prints=True)
def etl_pipeline(latitude: float = 35.2271, longitude: float = -80.8431):
    today = date.today().isoformat()
    #clean blob path with the date to store the results 
    blob_path = f"final/{today}/weather_etl.json"
    #exttract the data for the given latitude and longitude
    data = extract(latitude, longitude)
    #passes the raw data 
    enriched = transform(data, max_records=MAX_RECORDS)
    #loads the enriched data to Azure Storage
    load(enriched, blob_path)
    print(f"Pipeline complete. Results at {blob_path}")

 #This allows the pipeline to be run as a script, which is useful for testing and deployment.
if __name__ == "__main__":
    etl_pipeline()