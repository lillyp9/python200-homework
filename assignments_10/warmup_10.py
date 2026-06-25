#========LLM as Transform ========

#=====Question 1 
#Parse the string "Jan 5th, 2024" into an ISO date format like "2024-01-05".
"""Code since its date parsing """
#Classify a customer support ticket -- "my card was charged twice" -- into one of: billing, technical, or general.
"""LLM since it requires thinking , reading comprehension which a regex can not do."""
#Calculate the average of a list of numbers.
"""Code since its calculting the average """
#Extract the company name from a freeform job title like "Sr. Data Eng @ Acme Corp (contract)".
"""LLM since the input is irregular and needs language understanding  """
#Determine whether a product review is more than 100 words long.
"""Code since is counting words """

#======= Question 2 
#system = "Summarize this product review in a few sentences."

#In a comment block, explain what problem this creates downstream in a pipeline, and rewrite the prompt so it produces output that is easy to parse and store reliably.
#The problem is that saying to summarize the product review in few sentences which makes it unpredictable, can be various length and structure everytime which makes it harder to parse and store in the pipeline.
"""Rewritten prompt:
system = ("Summarize this product review in exactly one sentence."
"Reply with valid JSON only , using this format i am giving you:
'{"summary": "Your one sentence here}')
"""

#===== Question 3 

#Your dataset has 50,000 records and you need to run a classification call for each one using gpt-4o-mini. In a comment block, answer:

    #If each call takes 1 second on average, how long would sequential processing take?
""" It be 50,000 x 1 sec = 50,000 seconds which is equal to 13.9 hours."""
    #What is one practical strategy to handle this more efficiently at scale, without changing models?
"""Use OpenAi batch API to process requests at reduce cost or use concurrency to run multiple calls at once. """

#======== Azure OpenAI============

#======= Question 1 
"""The two reason to use Azure OpenAI is becuase of :
Data residency annd compliiance whohc the requets /info stays inside the Azure infranstructure then having it in the piblic OpenAI server.
Anoher reasin woould be Unified billing/support which costs appear on the same Azure bill and support is avaible through Microsoft . """

#======= Question 2 
""" Three of the Azure specific client parameters is:
Azure_endpoint: The URL of your organaization Azure OpenAI resource
API_Version: the version of Azure OpenAI API you're calling 
API_key: The Azure specific API key for your resources """

#====== Question 3 
#n a comment block, answer: when using AzureOpenAI, the model parameter in chat.completions.create() does not take a value like "gpt-4o-mini". What does it take instead, and where do you find the right value to use?
""" Instead of gpt-40-mini , the model parameter takes the deployment name , a named depolyment created by your party. You will find it in Azuure AI Foundry under deployment section of Azure OpenAI
"""