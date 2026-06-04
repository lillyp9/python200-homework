import os 
import json
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI
from scipy.stats import pearsonr
import pandas as pd
import matplotlib 
matplotlib.use("Agg")
import matplotlib.pyplot as plt 



#load env file and openAI
load_dotenv()
client = OpenAI()

def celsius_to_fahrenheit(celsius: float) -> str:
    """Convert a Celsius temperature to Fahrenheit and return it as a formatted string."""
    fahrenheit = (celsius * 9 / 5) + 32
    return f"{celsius}°C is {fahrenheit}°F"

#JSON schema dictionary
celsius_to_fahrenheit_schema = {
    "type" : "function",
    "function": {
        "name": "celsius_to_fahrenheit", #name tool the model going to use when it wants to call it
        "description": "Convert a Celsius temperature to Fahrenheit.",
        "parameters": {    
            "type": "object",
            "properties" : {    #parameters.properties 
                "celsius": {
                    "type": "number",
                    "description" : "The temperature in degree Celsius." #The model will read when decide whether to call this tool
                }
            },
            "required": ["celsius"] #agruments the model must provide
        }
    }
}

#Direct calls result 
print("Question 1 results:")
print(" ", celsius_to_fahrenheit(0))
print(" ", celsius_to_fahrenheit(100))
print(" ", celsius_to_fahrenheit(-40))

#Question 2:
from datetime import datetime 

#define the function
def get_current_time() -> str:
    #return the current date and time as a string
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

get_current_time_schema = {
    "type":"function",
    "function": {
        "name": "get_current_time",
        "description": "Get the current time and date.",
        "parameters": {         #parameters . properties 
            "type": "object",
            "properties": {}
        }
    }
}

def run_agent(user_message):
    # send request to the model with get_current_time as the only tool, run the tool
    # if the model calls it, and return the final answer.
    tools = [get_current_time_schema] #name tool 
    messages = [{"role": "user", "content": user_message}]

    #First API call ,model decides whether to call a tool
    response = client.chat.completions.create(
        model = "gpt-4o-mini",
        messages = messages,
        tools = tools
    )
    messages = response.choice[0].messages.content #calling the frist and best generated repsonse

    #If tool not call then return the answer directly  
    if not messages.tool_calls:
        return messages.content 
    
    #run the tool and feed the results back 
    messages.append(messages)
    for tool_call in messages.tool_calls:
        if tool_call.function.name == "get-current_time": #condition
            result = get_current_time()
        else:
            result = "Unknown tool"
        messages.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": str(result)
        })

    #Second API call. Model uses the tool result to write the fiinal answer.
    final = client.chat.completions.create(
        model = "gpt-4o-mini",      
        messages = messages,
        tools = tools
    )
    return final.choices[0].message.content  #return the content of the response from the AI


#Question 2 Prediction:
#1. Will calling run_agent("Convert 100 degrees Celsius to Fahrenheit") trigger a tool call? Why or why not?
# No calling run_agent should not be triggered. The only tool called is the get_current_time, which has nothing to do with the temperature conversiion.
#2. How many API calls will be made to answer this query?
#Just one API call , the model saw the quetson and saw no tool relevant so answer directly. 

#======= Question 3 
#Update tools  list and update run_agent to dispatch it when the model request it.
def run_agent(user_message):
    #send request to the model with answer.
    # both tools avaible, dispatch any and return the final answer.
    tools = [get_current_time_schema, celsius_to_fahrenheit_schema]
    messages = [{"role": "user", "content": user_message}]

    #First API call : keep in mind model will decide whether to call a tool
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        tools=tools
    )
    response_message = response.choices[0].message

    #If no tool was called, return the answer directly
    if not response_message.tool_calls:
        return response_message.content
    
    #Run whichever tool the model requested 
    messages.append(response_message)
    for tool_call in response_message.tool_calls:
        function_name = tool_call.function.name
        function_args = json.loads(tool_call.function.arguments)

        if function_name == "get_current_time":
            result = get_current_time()
        elif function_name == "celsius_to_fahrenheit":
            result = celsius_to_fahrenheit(**function_args)
        else:
            result = f"Unknown tool: {function_name}"

        messages.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": str(result)
        })

    #Second API call now the model uses the result to write the final answer
    final = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        tools=tools
    )
    return final.choices[0].message.content

print("\nQuestion 3 results")

response_a = run_agent("What is 37 degrees Celsius in Fahrenheit ?")
print("Response A:", response_a)
#The model did call the tool , the request asked for a specific Celsius-to-Fahrenheit conversion, the model called it with celsius=37.
# It got back 37 Celsuis to 98.6 Fahrenheit which was the output.

response_b = run_agent("What is the boiling point of water in plain English ?")
print("Response B:", response_b)
#No tool was called , It asking for a question not a conversion. So the model answered right away.


#================== Lesspn 3: Multi-Tool Agent
class CsvManager:
    def __init__(self, resources_dir: Path):
        self.resources_dir = resources_dir
        self.df = None
        self.csv_name = None

    # --- Small internal helpers --------------------------------------

    def _normalize_csv_name(self, filename: str) -> str:
        if not filename.lower().endswith(".csv"):
            return filename + ".csv"
        return filename

    def _available_csv_files(self) -> list[str]:
        if not self.resources_dir.exists():
            return []
        return sorted(
            [
                p.name
                for p in self.resources_dir.iterdir()
                if p.is_file() and p.suffix.lower() == ".csv"
            ]
        )

    def _ensure_loaded(self):
        if self.df is None:
            files = self._available_csv_files()
            example = files[0] if files else "your_file.csv"
            return {
                "error": (
                    "No CSV is loaded yet. First load one from resources/. "
                    f"For example: load_csv '{example}'."
                )
            }
        return None

    # --- Tools (public methods) --------------------------------------

    def list_csv_files(self):
        """
        List available CSV files in resources/.
        """
        files = self._available_csv_files()
        if not files:
            return {
                "message": (
                    "No CSV files found in resources/. "
                    "Create a resources/ folder and put one or more .csv files inside it."
                ),
                "files": [],
            }
        return {"files": files}

    def load_csv(self, filename: str):
        """
        Load a CSV file from resources/ and make it the active dataset.

        filename can be "bike_commute" or "bike_commute.csv".
        """
        filename = self._normalize_csv_name(filename)
        path = self.resources_dir / filename

        if not path.exists():
            return {
                "error": f"Could not find '{filename}' in resources/.",
                "available_files": self._available_csv_files(),
            }

        self.df = pd.read_csv(path)
        self.csv_name = filename

        return {
            "message": f"Loaded {filename} with shape {self.df.shape}.",
            "columns": self.df.columns.tolist(),
        }

    def get_columns(self):
        """
        Return column names for the currently loaded CSV.
        """
        error = self._ensure_loaded()
        if error:
            return error
        return self.df.columns.tolist()

    def summarize_columns(self, columns: list[str] | None = None):
        """
        Return basic summary stats for one or more columns.

        If columns is None, summarize all columns.
        Uses pandas.describe(include="all") to stay simple and readable.
        """
        error = self._ensure_loaded()
        if error:
            return error

        if columns is None:
            data = self.df
        else:
            missing = [c for c in columns if c not in self.df.columns]
            if missing:
                return {"error": f"These columns are not in the data: {missing}"}
            data = self.df[columns]

        summary = data.describe(include="all").transpose().round(3)
        return summary.to_dict()

    def describe_column(self, column: str):
        """
        Simple summary for a single column using pandas.describe().
        """
        error = self._ensure_loaded()
        if error:
            return error

        if column not in self.df.columns:
            return {"error": f"'{column}' is not a column. Options: {self.df.columns.tolist()}"}

        s = self.df[column]
        summary = s.describe().to_dict()

        cleaned = {}
        for key, value in summary.items():
            if isinstance(value, (int, float)):
                cleaned[key] = round(value, 3)
            else:
                cleaned[key] = value

        return cleaned

    def plot_data(self, y: str, x: str | None = None, plot_type: str = "line"):
        """
        Plot from the active CSV.
    
        - If x is None: plot y vs row index.
        - If x is provided: plot y vs x.
        """
        error = self._ensure_loaded()
        if error:
            return error
    
        if plot_type not in ["scatter", "line"]:
            return "Error: I can only do 'scatter' or 'line'."
    
        if y not in self.df.columns:
            return f"Error: column '{y}' is not in {self.df.columns.tolist()}"
    
        # If someone accidentally passes x == y, treat it like "plot y"
        if x == y:
            x = None
    
        # Scatter needs x
        if plot_type == "scatter" and x is None:
            return "Error: scatter plots need both x and y columns."
    
        title_csv = self.csv_name or "current CSV"
    
        if x is None:
            ax = self.df[y].plot(kind="line")
            ax.set_title(f"{title_csv} | Line plot: {y} vs row index")
            plt.show()
            return f"Plotted {y} vs row index as a line plot."
    
        if x not in self.df.columns:
            return f"Error: column '{x}' is not in {self.df.columns.tolist()}"
    
        ax = self.df.plot(x=x, y=y, kind=plot_type)
        ax.set_title(f"{title_csv} | {plot_type.title()} plot: {y} vs {x}")
        plt.show()
        
        return f"Plotted {y} vs {x} as a {plot_type}."

print("Class defined")
#============== Lesson 3 : Extend Csvmanager with computer_correlation ===
def compute_correlation(self, col1: str, col2:str):
    """
    Compute the Pearson correlation between two columns in the loaded DataFrame.
    Returns the correlation coefficient and p-value.
    """   
    error = self._ensure_loaded()
    if error:
        return error
    #condition for the two columns
    if col1 not in self.df.columns:
        return{"error": f"'{col1}' is not a column. Options: {self.df.columns.tolist()}"}
    if col2 not in self.df.columns:
                return{"error": f"'{col2}' is not a column. Options: {self.df.columns.tolist()}"}

    r, p = pearsonr(self.df[col1], self.df[col2])
    return {
        "col1": col1,
        "col2": col2,
        "pearson_r": round(float(r), 4),
        "p_value": round(float(p), 4),
    }
#have to replace my two CsvManager since when i run it , it doesnt pass through the compute_correlation
#so create a new method to the class 
CsvManager.compute_correlation = compute_correlation
#Single instances 
RESOURCES_DIR = Path("/home/lilly/python-200/lessons/07_AI_agents/resources")
csv_manager = CsvManager(RESOURCES_DIR)

node_tools = {
    "list_csv_files": csv_manager.list_csv_files,
    "load_csv": csv_manager.load_csv,
    "get_columns": csv_manager.get_columns,
    "summarize_columns": csv_manager.summarize_columns,
    "describe_column": csv_manager.describe_column,
    "plot_data": csv_manager.plot_data,
    "compute_correlation": csv_manager.compute_correlation,
}

tools_schema = [
    {
        "type": "function",
        "function": {
            "name": "list_csv_files",
            "description": "List available CSV files in the resources/ folder.",
        },
    },
    {
        "type": "function",
        "function": {
            "name": "load_csv",
            "description": "Load a CSV file from the resources/ folder and make it the active dataset.",
            "parameters": {
                "type": "object",
                "properties": {
                    "filename": {
                        "type": "string",
                        "description": "CSV filename in resources/, e.g. 'bike_commute.csv'.",
                    }
                },
                "required": ["filename"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_columns",
            "description": "Get the column names of the currently loaded CSV.",
        },
    },
    {
        "type": "function",
        "function": {
            "name": "summarize_columns",
            "description": "Show basic summary statistics for columns (uses pandas.describe).",
            "parameters": {
                "type": "object",
                "properties": {
                    "columns": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional list of column names. If omitted, summarize all columns.",
                    }
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "describe_column",
            "description": "Show basic summary statistics for a single column (uses pandas.describe).",
            "parameters": {
                "type": "object",
                "properties": {
                    "column": {
                        "type": "string",
                        "description": "Column name to describe.",
                    }
                },
                "required": ["column"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "plot_data",
            "description": "Plot data from the active CSV. If only y is provided, plot y vs row index.",
            "parameters": {
                "type": "object",
                "properties": {
                    "y": {"type": "string", "description": "Column name for y-axis."},
                    "x": {"type": "string", "description": "Optional column name for x-axis."},
                    "plot_type": {
                        "type": "string",
                        "enum": ["scatter", "line"],
                        "description": "Type of plot to create.",
                    },
                },
                "required": ["y"],
            },
        },
    }, 
        # 
      {
        "type": "function",
        "function": {
            "name": "compute_correlation",
            "description": "Compute the Pearson correlation coefficient and p-value between two numeric columns of the loaded CSV. Use this whenever the user asks about correlation, relationship, or how two columns relate.",
            "parameters": {
                "type": "object",
                "properties": {
                    "col1": {"type": "string", "description": "First column name."},
                    "col2": {"type": "string", "description": "Second column name."},
                },
                "required": ["col1", "col2"],
            },
        },
    },    
]
#======== Run_agent=======
def run_agent_cycle(messages, user_text, max_tool_rounds=5):
    """
    Run through one react-agent loop using a simple tool-using agent.
    `messages` parameter will usually just contain a system prompt, 
    and then user text will be appended.  

    The loop has three main steps:

    REASON:
      - Call the model with the conversation so far.
      - The model either replies normally, or asks to call a tool from tool set.

    ACT:
      - If tools are requested, run the Python functions

    OBSERVE:
      - Append each requested tool result back into the LLMs conversation history.
      - On the next iteration, the model reads those tool call results and determines
        whether it has reached the goal.

    Stop condition:
      - If the model returns an assistant message with no tool calls, this is the 
        final answer for this react cycle, this implies that reasoning alone without 
        tool calls was enough.  
      - max_tool_rounds is a safety cap to prevent infinite loops.
    """
    messages.append({"role": "user", "content": user_text})

    def observe_tool_result(tool_call_id, result):
        """
        Return a tool's return value as a message that can be appended to the
        LLMs conversation history. The model will read this tool output on the next
        REASON step.
        """
        content = json.dumps(result, default=str) if not isinstance(result, str) else result
        tool_message = {"role": "tool",
                        "tool_call_id": tool_call_id,
                        "content": content,}
        return tool_message

    for loop_idx in range(max_tool_rounds):
        # REASON: call the model
        # Here it will make use of any previous tool outputs it appended ("observed")
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=messages,
            tools=tools_schema,
        )

        msg = response.choices[0].message

        # Append the assistant message to the conversation history.
        # Use a plain dict so `messages` stays simple and inspectable.
        assistant_entry = {"role": "assistant", "content": msg.content}
        if msg.tool_calls:
            assistant_entry["tool_calls"] = [tc.model_dump() for tc in msg.tool_calls]
        messages.append(assistant_entry)

        # No tool calls means the model is answering directly.
        if not msg.tool_calls:
            return msg.content 

        # ACT + OBSERVE: run each tool call, then append its result.
        # Note there may be multiple tool calls
        for tool_call in msg.tool_calls:
            name = tool_call.function.name
            tool_args = json.loads(tool_call.function.arguments or "{}")

            print(f"ACT: {name}({tool_args})")

            fn = node_tools.get(name)
            if fn is None:
                result = {"error": f"Tool '{name}' not found."}
            else:
                try:
                    result = fn(**tool_args) if tool_args else fn()
                except Exception as e:
                    print(f"Tool error in {name}: {type(e).__name__}: {e}")
                    result = {"error": f"Tool '{name}' failed: {type(e).__name__}: {e}"}
                    
            # OBSERVE: append the tool result back into the conversation history.
            messages.append(observe_tool_result(tool_call.id, result))
            
            # After we appending information about all tool outputs, we loop back and REASON again.

    return "I hit the tool-round limit. Try a simpler request."

#========= Run the correlation request: Question 5 
SYSTEM_PROMPT = """You are a data analysis assistant. You have tools to load
CSV files, inspect columns, summarize data, and compute correlations.
Use the tools to answer the user's questions about the data."""

print("\n=== Q5 ===")
messages = [{"role": "system", "content": SYSTEM_PROMPT}]
result = run_agent_cycle(
    messages,
    "Load bike_commute.csv and compute the correlation between avg_traffic_density and avg_speed_kmh."
)
print("\nFinal response:", result)

#Question 6 : The role each represent in the ReAct loop

#system : The initial instructions to the model setting the model agent behavior and tells the model what tool it has 

#user: a message from the human behind thhe model 

#assitant: the model repsonse on the content or final output after running a tool or both 

#tool: The result of running a tool. The model sees this and depending on the context use it to write the final output or call another tool.

#print result
print("\n Question 6 : Full message list")
print(json.dumps(messages, indent=2, default=str))

#=============================Lesson 04: molagents
from smolagents import ToolCallingAgent , CodeAgent , OpenAIServerModel, tool

#Question 7 
@tool
def compute_correlation(col1:str , col2:str) -> dict:
    """Compute correlation between both columns of the csv
    Args:
        col1: Name of first column.
        col2: Name of second column.
    Returns:
        A dict with two column names, the Prearson r value, and the p-value.
    """
    return csv_manager.compute_correlation(col1, col2)
print("\nQuestion 7")
print(compute_correlation.description)

#Question 7: The smolagents created the description automatically by reading the function and the docstring.
#The question 4 JSON schema , the smolagents pull all the code from there.

#The smolagents need from me to make a good description is the arguments and return value , the docstring with a clear one line summary, "Args:" section describing what comes back, it is curcial to write clear, honest documentation.

#========== Question 8 

# Wrap the rest of the CsvManager methods as smolagents tools
@tool
def list_csv_files() -> dict:
    """List available CSV files in resources/.

    Returns:
        A dict with a "files" list, or a message if none are found.
    """
    return csv_manager.list_csv_files()


@tool
def load_csv(filename: str) -> dict:
    """Load a CSV file from resources/ and make it the active dataset.

    Args:
        filename: CSV filename in resources/ (e.g. 'bike_commute.csv').

    Returns:
        A dict with a status message and column names, or an error dict.
    """
    return csv_manager.load_csv(filename)


@tool
def get_columns() -> list[str] | dict:
    """Return column names for the currently loaded CSV.

    Returns:
        A list of column names, or an error dict if no CSV is loaded.
    """
    return csv_manager.get_columns()


@tool
def summarize_columns(columns: list[str] | None = None) -> dict:
    """Return summary stats for selected columns (or all columns).

    Args:
        columns: Column names to summarize. If None, summarizes all columns.

    Returns:
        A dict of summary statistics (from pandas.describe), or an error dict.
    """
    return csv_manager.summarize_columns(columns)


@tool
def describe_column(column: str) -> dict:
    """Describe a single column (basic stats).

    Args:
        column: The name of the column to describe.

    Returns:
        A dict of basic stats for the column, or an error dict.
    """
    return csv_manager.describe_column(column)


@tool
def plot_data(y: str, x: str | None = None, plot_type: str = "line") -> str | dict:
    """Plot from the active CSV.

    Args:
        y: Column name to plot on the y-axis.
        x: Column name to plot on the x-axis. If None, uses row index.
        plot_type: 'line' or 'scatter'. Scatter requires both x and y.

    Returns:
        A short success message string, or an error dict/string.
    """
    return csv_manager.plot_data(y=y, x=x, plot_type=plot_type)


TOOLS = [
    list_csv_files,
    load_csv,
    get_columns,
    summarize_columns,
    describe_column,
    plot_data,
    compute_correlation,
]

# Set up the model
api_key = os.getenv("OPENAI_API_KEY")
model = OpenAIServerModel(api_key=api_key, model_id="gpt-4o-mini")

# ToolCallingAgent
TOOL_PROMPT = (
    "You are a small data assistant to help analyze CSVs in resources/. "
    "Use the available tools. Keep answers short."
)
tool_agent = ToolCallingAgent(tools=TOOLS, model=model, instructions=TOOL_PROMPT)

# CodeAgent
CODE_INSTRUCTIONS = """
You are a helpful CSV analysis assistant.

You can do two kinds of actions:
1) Call the provided tools.
2) Write and execute Python code when tools are not enough.

Rules:
- Prefer tools for simple tasks.
- IMPORTANT: If the user requests plot styling (color, marker, title text, labels, grid, etc.)
  that the plot_data tool cannot control, DO NOT call plot_data.
  Instead, write matplotlib code directly so the plot matches the request.
- Be honest: only claim you did something if the code or tool actually did it.
- Assume the active dataset lives in csv_manager.df after a CSV is loaded.
"""
code_agent = CodeAgent(
    tools=TOOLS,
    model=model,
    instructions=CODE_INSTRUCTIONS,
    additional_authorized_imports=["pandas", "matplotlib.pyplot", "numpy"],
    max_steps=8,
)

# Run the same prompt through both
prompt = "Load bike_commute.csv. Plot avg_heart_rate vs duration_min as a scatter plot with green dots."

print("\nQuestion 8 : ToolCallingAgent ===")
response_tool = tool_agent.run(prompt)
print("\nToolCallingAgent response:", response_tool)

print("\nQuestion 8: CodeAgent ===")
response_code = code_agent.run(prompt, additional_args={"csv_manager": csv_manager})
print("\nCodeAgent response:", response_code)

# Q8 comment:
# The ToolCallingAgent loaded the CSV and called plot_data to make
# the scatter plot, but it did not change the dots to green the
#  plot_data tool has no color parameter, so the model had no way to
#  control color. The agent often still claimed it made green dots,
#  even though it didn't. That's a hallucination.
#  The CodeAgent loaded the CSV with load_csv and then wrote its own matplotlib code instead of calling plot data,
#  matplotlib code instead of calling plot_data because its system
#  prompt told it to write code when styling is requested. That gave
#  it the freedom to actually produce green dots.
#
# 2. ToolCallingAgent is better when the task fits cleanly inside the
#    tools you've written predictable, safe, and easier to debug.
#    CodeAgent is better when the task needs flexibility the tools
#    don't provide, because the agent can fill the gap by writing code.
#    The tradeoff: CodeAgent runs arbitrary code, which is more
#    powerful but riskier.



#==== Final Reflection ==============
#Overall ToolCallingAgent is better choice for tasked with a well description set of rules , description , and behavior. Such as a customer-support bot that can check order status, issue refunds,
#and look up shipping information. Three operations with its own known function call.

#Now with CoedAgent the biggest risk is that the agent/model can generate and run actionable python code. Even with the guardrails , sandbox interpreter, the model can produce subtle bugs, unsantized the environmennt as well as corrupted the data.
#It can be tricked into writing code the developer didnt ask for ( prompt injection). 
#With a ToolCallingAgent, every possible action has been written and tested by a human


