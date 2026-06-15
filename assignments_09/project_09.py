import requests 
import pandas as pd
import json 
from datetime import date
from azure.storage.blob import ContainerClient
from azure.identity import DefaultAzureCredential 

#Constants 
Account_URL = "https://odalissctd2026sa.blob.core.windows.net"
Container = "pipeline-data"

#====== Step 1 : Extract 

#Default location NC
latitude = 35.2271
longitude = -80.8431

def extract_weather_data():
    #script
    """Fetch 7 days of hourly weather data froom Open-Meteo API.
    Return the JSON response as a dictionary """
    url = f"https://api.open-meteo.com/v1/forecast?latitude={latitude}&longitude={longitude}&hourly=temperature_2m,precipitation&forecast_days=7"

    #Catch early errors 
    response = requests.get(url)
    response.raise_for_status()
    return response.json()

#==== Main =======
if __name__ == "__main__":
    #Load and Connect to Blob Storage 
    credential = DefaultAzureCredential()
    container = ContainerClient(
        account_url=Account_URL,
        container_name=Container,
        credential=credential
    )

    print("Extract weather data")
    weather_data = extract_weather_data()

    #======== Step 2: Serialize 
    json_bytes = json.dumps(weather_data).encode("utf-8")

    #====== Step 3: Load 

    today = date.today().isoformat()
    blob_path = f"raw/{today}/weather.json"
    container.upload_blob(blob_path, json_bytes, overwrite=True)
    print(f"Upload {blob_path} ({len(json_bytes)} bytes)")

    #======= Step 4 : Verify 
    for blob in container.list_blobs():
        print(f"{blob.name}({blob.size} bytes)")
        print(f"\nBlob in containers with the list each name and size")

    #====== Step 5 : Read back 
    #download the blob that you just upoaded 
    downloaded = container.download_blob(blob_path).readall()
    data = json.loads(downloaded)

    #uses panda datafame and printout the first 5 rows 
    df = pd.DataFrame(data["hourly"])
    print(df.head())


    #Save download json for mentor to see 
    with open("outputs/weather_raw.json", "w") as f:
        f.write(downloaded.decode("utf-8"))
        print("\nSaved to outputs/weather_raw.json")
        print("fully completed")

