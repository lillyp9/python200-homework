import requests 
import os
import pandas as pd
import json 
from dotenv import load_dotenv
from datetime import date
from openai import OpenAI
from azure.storage.blob import ContainerClient
from azure.identity import DefaultAzureCredential 

load_dotenv(""
"/home/lilly/python200-homework/.env")
#=====Constants 
Account_URL = "https://odalissctd2026sa.blob.core.windows.net"
Container = "pipeline-data"
Valid_labels = {"good","marginal", "bad"}
System_prompt = (
    "You are classifying hourly weather conditions for outdoor running. "
    "Given a temperature in Celsius and a precipitation amount in mm, "
    "classify the conditions as exactly one of: good, marginal, or bad. "
    "Reply with that one word only -- no punctuation, no explanation."
)

def make_user_message(record):
    return(
        f"Temperature: {record['temperature_2m']}C,"
        f"Precipitation: {record['precipitation']}mm"
    )

def reshape_hourly(data):
    #reshape the "hourly" parallel lists into a list of per-hour record dict
    hourly = data["hourly"]
    records = [] #placeholder 
    for i in range(len(hourly["time"])): #condition 
        records.append({
            "time": hourly["time"][i],
            "temperature_2m": hourly["temperature_2m"][i],
            "precipitation": hourly["precipitation"][i],
        })
    return records

    #====== connect to Blob storage 
credential = DefaultAzureCredential()
container = ContainerClient(Account_URL, Container, credential=credential)

today = date.today().isoformat()
blob_path = f"raw/{today}/weather.json"

    #==== Step 1 : Read with fallback in case of errors , use fallback dataset
try:
    raw = container.download_blob(blob_path).readall()
    data = json.loads(raw.decode("utf-8"))
    print(f"Loaded weather data from {blob_path}")
    #Catch error if the blob not found 
except Exception as e:
    print(f"Could not load {blob_path} ({e})")
    print("Fetching fresh weather data from API as fallback")
    import requests
    url = "https://api.open-meteo.com/v1/forecast?latitude=35.2271&longitude=-80.8431&hourly=temperature_2m,precipitation&forecast_days=7"
    resp = requests.get(url)
    resp.raise_for_status()
    data = resp.json()

records = reshape_hourly(data)
print(f"dataLoaded {len(records)} hourly records")       

 #Only the first 24 records (one day of hourly data)
records = records[:24]

#======Step 2 Trasform 
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
enriched = [] #placeholder
#loop through the records and classify using LLM, print progress every 6 records
for i, record in enumerate(records):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": System_prompt},
            {"role": "user", "content": make_user_message(record)},
        ]
    )
    raw_label = response.choices[0].message.content.strip().lower()
    label = raw_label if raw_label in Valid_labels else "unknown"
    enriched.append({**record, "conditions": label})
    if (i + 1) % 6 == 0:
        print(f"  Processed {i + 1} records")

#====== Step 3: Write 
processed_path = f"processed/{today}/weather_classified.json"
container.upload_blob(
    processed_path,
    json.dumps(enriched).encode("utf-8"),
    overwrite=True
)
print(f"Uploaded to {processed_path}")

#Step 4: Spot-Check 
downloaded = container.download_blob(processed_path).readall()
processed_data = json.loads(downloaded.decode("utf-8"))
df = pd.DataFrame(processed_data)

print("\nLabel distribution:")
print(df["conditions"].value_counts())
print("\nFirst 5 rows:")
print(df.head(5))

#Step 5: Save Output 
os.makedirs("outputs", exist_ok=True)
with open("outputs/first_10_records.json", "w") as f:
    json.dump(enriched[:10], f, indent=2)
print("\nSaved first 10 records to outputs/first_10_records.json")

#================ Reflection 
"""The LLM was a good fit for this task, however retrieving the dataset needed turn out to be diffcult , instead i relay on the fallback dataset from the API, which isnt ideal but allowed me to complete the project.
It did a good job grabbing all the relevant data , upload and process the data , the later classifying weather good , bad  or marginal. 
I do think deterministic code could be better because thhe inputs are just two numbers , temperature, and precipitation. Having it shorter.
Given with the rules , the output was consistent.I dont think i would have lost or gain anthing with a few changes such as "temperature > 10 and precipitation < 1 → good") it would just bea different apparach but similar output.Hypthetically speaking. """

