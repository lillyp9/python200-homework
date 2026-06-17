#========= Perfect Orchestration ========

#====== Perfect Question 1 
#what is the difference between a @task and a @flow in Prefect? You have a helper function that converts a temperature from Celsius to Fahrenheit -- a pure, in-memory calculation with no I/O. Would you decorate it with @task? Why or why not?
"""The difference between @tas and @flow  is that @tas is a single focused unit of work, while @flow calls tasks in order , and managing the run
In order to do the Celsius to Fahrenheit with @task is not needed as it is a in memory calculation with no I/O, no failure risk, nothing """

#Question 2 : 
#Write the decorator (just the decorator line, not the full function) for a task named call_api that retries up to 3 times with a 30-second delay between attempts.
"""@task(retries=3, retry_delay_seconds=30)
def call_api():
    pass
"""

#Question 3 :
#You run your pipeline and the Prefect UI shows: extract is Completed, transform is Failed, load never ran. In a comment block, describe: where in the UI do you look to understand what went wrong, and what specific information would you expect to find there?
"""I would look at the failed transform task and open the Logs tab , Im  excpet to find the exception traceback showing what error occured. You can see where the pippeline broke."""

#======== Production Patterns 

#====== Question 1 :
#In a comment block, explain what raise_for_status() does and why it is better than writing if response.status_code != 200: print("error") in a pipeline task. What happens to downstream tasks in each case when the API returns a 500 error?
"""It stops the task immediately and raises the expection . Then with the raise_for_status on error 500, the task fails and downstream tasks  the transform and load never run. With the the print() The code keeps going and passes the empty data which could caused failures down the line."""
#Production Question 2
#Your pipeline uploads results to final/{today}/weather_etl.json with overwrite=True. The pipeline crashes halfway through the transform step. You fix the bug and re-run it from the beginning. In a comment block, explain: what does overwrite=True protect you from in this scenario, and what would happen without it?
"""On a re-run, the blob from a previous attempt may already
exist. overwrite=True lets the new run replace it. Without it the upload would fail because the blob already exists, so your fixed re-run would crash at the load step ."""
#Production Question 3
#Write a task stub -- just the function signature, decorator, and a single log line -- that uses get_run_logger() to log an INFO message saying how many records were loaded. The function should accept records (a list) and blob_path (a string) as arguments.
from prefect import task, get_run_logger
@task 
def load(records: list, blob_path: str):
    logger = get_run_logger()
    logger.info(f"Loaded {len(records)} records to {blob_path}")

    