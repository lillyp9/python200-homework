from prefect import task, flow 
import pandas as pd 
import numpy as np 

@task
def create_series(arr):
    return pd.Series(arr, name="values")

@task
def clean_data(series):
    cleaned_series = series.dropna()
    return cleaned_series

@task
def summarize_data(series):
    summary = {
        "mean" : series.mean(),
        "median": series.median(),
        "std": series.std(),
        "mode": series.mode()[0]
    }

    return summary 


@flow
def data_pipeline(arr):
    series = create_series(arr)
    clean = clean_data(series)
    summary = summarize_data(clean)
    return summary

if __name__ == "__main__":
    arr = np.array([12.0, 15.0, np.nan, 14.0, 10.0, np.nan, 18.0, 14.0, 16.0, 22.0, np.nan, 13.0])
    result = data_pipeline(arr)
    for key , values in result.items():
        print(f"{key} :{values}")
   

#Question 1 
#It only took a couple hours instead of minutes of manually in put for the  code to run, not to mention its small function that call the whole section/part of the code.However i will see I did have trouble , having the flow run as it keep breaking down to to a fail of dependency , writing the code was easy , the running was the issue.
# Question 2 
# The prefect is useful when you just want something to update and load , insteaad of manualling having it show on another window having to kill then run the port again. 