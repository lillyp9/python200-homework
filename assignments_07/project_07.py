
import os
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from dotenv import load_dotenv

from smolagents import CodeAgent, OpenAIServerModel, tool

#load api 
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
print("API key loaded.")
#paths of files 
DATA_PATH = Path("/home/lilly/python200-homework/assignments_01/outputs/merged_happiness.csv")
OUTPUT_DIR = Path("/home/lilly/python200-homework/assignments_07/outputs")
OUTPUT_DIR.mkdir(exist_ok=True)


#global dataframe shared by all tools
df = None
#======================Task 1: Define your Tool=====
@tool
def load_happiness_data() -> dict:
    """ Load the World Happniess dataset.
    merged CSV first, if thats missing, falls back to loading every yearly CSV from the happiness_project resource folder and 
    concatenating them. The result is stored in global df. 

    Returns:
        A dict with "shape" (rows,cols) and "columns" (list of column names).
    """
    #Store in global df 
    global df 
    #Read the csv
    df = pd.read_csv(DATA_PATH)
    return {"shape": list(df.shape), "columns": df.columns.tolist()}

@tool
def summarize_column(column: str) -> dict:
    """ Return the descirptive statistics for a single column.
    Args:
        column: The column name to describe.

    Returns:
    A dict from pandas describe(), or an error dict if found invalid
    """
    #condition
    if df is None:
        return{"error": "No data loaded. Call the load_happiness_data first."}
    if column not in df.columns:
         return {"error": f"'{column}' is not a column. Options: {df.columns.tolist()}"}
    return df[column].describe().to_dict()

@tool 
def compute_correlation(col1: str, col2: str) -> dict:
    """Compute the Pearson correlation between two numeric columns.

    Args:
        col1: First column name.
        col2: Second column name.

    Returns:
        A dict with col1, col2, pearson_r, and p_value (rounded to 4 decimals).
    """
    if df is None:
        return {"error": "No data loaded. Call load_happiness_data first."}
    if col1 not in df.columns:
        return {"error": f"'{col1}' is not a column. Options: {df.columns.tolist()}"}
    if col2 not in df.columns:
        return {"error": f"'{col2}' is not a column. Options: {df.columns.tolist()}"}

    clean = df[[col1, col2]].dropna()
    r, p = pearsonr(clean[col1], clean[col2])
    return {
        "col1": col1,
        "col2": col2,
        "pearson_r": round(float(r), 4),
        "p_value": round(float(p), 4),
    }

@tool
def get_top_n_countries(column: str, year: int, n: int = 5) -> dict:
    """Return the top N countries ranked by a given column for a specific year.

    Args:
        column: The column to rank by (example: "Happiness score").
        year: The year to filter to (example: 2019).
        n: How many countries to return. Defaults to 5.

    Returns:
        A dict with a "results" list of {country, value} pairs.
    """
    #condition
    if df is None:
        return {"error": "No data loaded. Call load_happiness_data first."}
    if column not in df.columns:
        return {"error": f"'{column}' is not a column. Options: {df.columns.tolist()}"}
    if "year" not in df.columns:
        return {"error": "Dataset has no 'year' column."}

    year_df = df[df["year"] == year]
    if year_df.empty:
        return {"error": f"No data for year {year}. Available: {sorted(df['year'].unique())}"}

    top = year_df.nlargest(n, column)[["Country", column]]
    return {
        "results": [
            {"country": row["Country"], column: row[column]}
            for _, row in top.iterrows()
        ]
    } 

#==================== Task 2 : Build the Agent
model = OpenAIServerModel(api_key= api_key, model_id="gpt-4o-mini")

SYSTEM_PROMPT = """
You are a data anaylst assitant for the World Happiness dataset.
Use the available tools for loading data,summarizing columns, computing correlations,
and ranking the countries. Write Python code directly only when the tools are not sufficent for the task at hand.
For example, when creating a custom plots or computing somethiing the tools dont cover it.
Be nice but not over-friendly , act as if you are talking to a student.
When you write the code , you must EXECUTE it, not just describe it. Never show the code block as the final answer, allways run the code so the result may show.
When saving plots, save them to '/home/lilly/python200-homework/assignments_07/outputs/'.
Confirm the file was written by checking it exists after saving.

The full DataFrame is available as the variable `df` inside your code.
Use df directly when you need to write plotting code or do something the tools don't cover.
"""


agent = CodeAgent(
    tools=[load_happiness_data, summarize_column, compute_correlation, get_top_n_countries],
    model=model,
    instructions=SYSTEM_PROMPT,
    additional_authorized_imports=["pandas", "matplotlib.pyplot", "scipy.stats"],
    max_steps=8,
)

print("Agent ready.")

#============ Running the Project 
if __name__ == "__main__":

#=============================== Task 3 : Run Queries
    queries = [
    "Load the happiness data and tell me its shape and column names.",
    "Summarize the happiness_score column.",
    "What is the correlation between gdp_per_capita and happiness_score? Is it statistically significant?",
    "Show me the top 5 happiest countries in 2020.",
    "Plot happiness_score over the years as a line chart, with one line per region. Save the plot to outputs/happiness_by_region.png.",
]
# Pre-load the data
    print("\nPre-loading data")
    load_happiness_data()

    for query in queries:
        print(f"\n--- Query: {query} ---")
        response = agent.run(query, reset=False, additional_args={"df": df})
        print(response)

#Note When doing thhis assignment the CodeAgent couldnt grab the information it needed , so it created its own matplotlib code . Just adding a line to the system prompt and reloading the data ,  letting the model know that the full dataframe is in the variable 'df', 
#The model was able to pass thrpugh and obtain the information that was needed.Without that oe detail the agent sandbox can pnly see the dict that has load_happiness_data returns (just the shape + columns) and it'll hallucinate and create its own data to fullfill the rrequest. 


#=================== Task 4 : Create my own questions 
    my_query_1 = "Whats the range of GDP per capital in the dataset? What are the lowest and highest values ?"
    print(f"\n My Query: {my_query_1}")
    response_1 = agent.run(my_query_1, reset=False, additional_args={"df": df})
    print(response_1)
#This would triggered the tool that would be used which is summarize_columns with the "GDP per capital" and have the min and max return. However when running the code , it did Not summarize_column , it wrote pandas code  ─ Executing parsed code: ─────────────────────────────────────────────────────────────── 
#gdp_min = df['GDP per capita'].min()                                                    
#gdp_max = df['GDP per capita'].max() The min came back as 0.0  and the max 10.0, it doenst looks like reaal GDP figures, it shows that the data itself isnt clean which means the quality of the data from week 1 merge is compromised.

#Theres no need for custome code since , we just need to retrieve it.

#Question2 
    my_query_2 = "How many countries had a happiness score above 7 in 2020? List the countries names."
    print(f"My Query 2: {my_query_2}")
    response_2 = agent.run(my_query_2, reset=False, additional_args={"df": df})
    print(response_2)
#This did not call a tool, the agent wrote a direct pandas filter and return the countries with their names. 
#CodeAgent outperforms ToolCalling Agent , the question may not be cleared or match perfectly with a tool the mode can call, so theres flexibility .

#===================task 5 Reflection =====================
#Question 1 : 
#In Query 3 the agent gave a pearson_r of 0.6313 and a p_value of 0.0,
# and called it statistically significant. The p_value is 0 because my
# tool rounds to 4 decimals so the real value is just a very tiny
# number. The agent used the p-value right, but it didnt say what
# threshold it was using. Usually significance is p < 0.05 so 0.0
# obviously clears it, but the agent could of explained that part
# better instead of just saying significant.

# Question 2:
#From Task 4 , Query 1 surprised me because the GDP per capita came back with
#min 0.0 and max 10.0, which doesnt look like real GDP numbers. The
#agent just answered the question with what it found, it didnt notice the data looked off. The data probably has some normalized or placeholder values left over from the week 1 merge.
#If this was in a real life situation m a analyst would flagged it and look into it.


#Question 3 
#One tool that would make this agent more useful is a filter tool that takes a column, a operator, and a value.
#Right now my tools cant filter by a condition, so the agent has to write its own pandas code for stuff like "how many countries had happiness above 7". A filter tool would make those answers come
#from a real tool instead of code, which is more predictable.

