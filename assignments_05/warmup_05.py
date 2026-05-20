import os 
from dotenv import load_dotenv
from openai import OpenAI

#load the .env file
load_dotenv()
#initialize the OpenAI client
client = OpenAI()

#--------- Chat completion API -------

#API Q1
response = client.chat.completions.create(
    model = "gpt-4o-mini",
    messages=[
        {"role": "user", "content": "What is one thing that makes Python a good language for beginners?"}
    ]
)

print("Question 1 response:", response.choices[0].message.content)
print("Question 1 model:", response.model)
print("Question 1 token usage:", response.usage.total_tokens)

#API Q2
prompt = "Suggest a creative name for a data engineering consultancy."
temperatures = [0, 0.7, 1.5]

#create a loop
for temp in temperatures:
    response = client.chat.completions.create(
        model = "gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature = temp
    )
    print(f"Question 2 temperature={temp}: {response.choices[0].message.content}")
#What i nptice is with temperature = 0 , the ouput is the same , the modle just it everytime. but when it does temperature = 0.7 
# it shows various ouputs, and with temperature = 1.5 , its picks random. For safe testing , I would choose temperature = 0 , since its the same word over and over again for consistent reproduceible output.

# #API Q3
response = client.chat.completions.create(
    model = "gpt-4o-mini",
        messages=[{"role": "user", "content": "Give me a one-sentence fun fact about pandas (the animal, not the library)."}],
    n=3,
    temperature=1.0
)
print(" Question 3 completions:")
#Iterate over response.choices and print each one.
for i, choice in enumerate(response.choices, start=1):
    print(f"{i}. {choice.message.content}")

#API Q4
response = client.chat.completions.create(
    model = "gpt-4o-mini",
    messages = [{"role": "user", "content": "Explain how neural networks work."}],
    max_tokens = 15
)
print("Question 4 response:",response.choices[0].message.content)
print("Question 4 finish reason:", response.choices[0].finish_reason)

#The response cut off is because of the token limit we set which is 15.
#The finish reason is the "length" which the model stop generated because of the max token limit that is set.
#You use  max_tokens to control cost to keep the response short and straight to the pijnt but it leaves unfinished response.

#========= System Q1 
#=============== first personality: patient tutor ================
messages_tutor = [
    {"role": "system", "content": "You are a patient, encouraging Python tutor. You always explain things simply and end with a word of encouragement."},
    {"role": "user", "content": "I don't understand what a list comprehension is."}
]

#model 
response = client.chat.completions.create(
    model = "gpt-4o-mini",
    messages = messages_tutor
)
print("System Q1 - Patient tutor:", response.choices[0].message.content)

#Second personalit : boxer
messages_boxer = [
    {"role": "system", "content": "You are a patient, encouraging Python tutor. You always explain things simply and end with a word of encouragement."},
    {"role": "user", "content": "I don't understand what a list comprehension is."}
]
#model
response = client.chat.completions.create(
    model = "gpt-4o-mini",
    messages = messages_boxer
)
print("System Q1 - Boxer:", response.choices[0].message.content)

# Even with the same prompt , the model responded differently beacuse of the  different peronsality.
# The tutor version, the model explained in a educational way and ended with encouragement.
#while the bxoer version, the mode responded in a greesive way and ended with cheering you on as a boxer would do.

# ====== System Q2

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "My name is Jordan and I'm learning Python."},
    {"role": "assistant", "content": "Nice to meet you, Jordan! Python is a great choice. What would you like to work on?"},
    {"role": "user", "content": "Can you remind me what my name is?"}
]

#model 
response = client.chat.completions.create(
    model = "gpt-4o-mini",
    messages = messages
)
print("System Q2:", response.choices[0].message.content)

#The model knows Jordan  since we feeded it the conversation history. 
#The model has no memory between calls , every request is independent, so 
#it relies on the conversation history we provide or information. So if we rem ove the name or anything that hints the name , the model would have no idea.

#===== Prompt Egineering ======
#prompt Q1

reviews = [
    "The onboarding process was smooth and the team was welcoming.",
    "The software crashes constantly and support never responds.",
    "Great price, but the documentation is nearly impossible to follow."
]
#for loop to iterate over reviews and print the model sentiment review for each one.
for i, review in enumerate(reviews, start=1):
    response = client.chat.completions.create(
        model = "gpt-4o-mini",
        messages = [
            {"role": "user", "content": f"Classify the sentiment of this review from positive, negative, or mixed. Answer in one word.\n\nReview:{review}"}
        ]
    )
    print(f"\nQ1 Review {i}: {response.choices[0].message.content}")

#======== Prompt Q2
#example given to model
one_shot_example = """Example:
Review: "Fast shipping but the item arrived damaged."
Sentiment: mixed
"""

print("\nQ2 results one-shot example:")
#loop
for i, review in enumerate(reviews, start=1):
    prompt = f""" Classify the sentiment of this review from positive, negative, or mixed. 
    {one_shot_example}
    Review:{review}
    Sentiment:"""
    #model response
    response = client.chat.completions.create(
        model = "gpt-4o-mini",
        messages = [
            {"role": "user", "content": prompt}]
    )
    print(f"Review {i}: {response.choices[0].message.content}")

#Adding the one shot example improved the model making it more cxonsistent,
#In Q1 the model has the first letter captilized  but in Q2 the model copies the exact formatand single word.
#Its training the model on what the answer should be likeand to give out the output in the format we want. 

#===== Prompt Q3
#need 3 examples 
#ex1
few_shot_example = """Example 1:
Review: "The product is amazing!"
Sentiment: positive

Example 2:
Review: "Completely fell apart after one use. I want a refund."
Sentiment: negative

Example 3:
Review:" The product is okay, but the customer service was terrible."
Sentiment: mixed
"""

print("\nQ3results few-shot :")
#loop
for i, review in enumerate(reviews, start=1):
    prompt = f""" Classify the sentiment of the following review as positive,negative, or mixed.
{few_shot_example}
Review: {review}
Sentiment: """
    #model response 
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages = [{"role": "user", "content": prompt}]
    )
    print(f"Review {i}: {response.choices[0].message.content}")

#Q3 : Comparing all 3 approaches 
#With the zero shot approach, the output was fast however inconsistent.
#Its a very basic prompt and the model already has the basic knowledge to answer.
#The one-shot approach, has one example which helps the model understand the format, and its conistent with the ouput.
# The few-shot approach, add 3 examples  covering different cases, This help the task whne the model
# get confused when the task is complex. Great use for whne you need high accuracy. 


#============ Local Models with Ollama =============   
# model currently usedd  with same prompt 
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages = [
        {"role": "user", "content": "Explain what a large language model is in two sentences."}
    ]
)
print("OpenAI response:", response.choices[0].message.content)

#=== Ollama Q1 Output from terminal 
"""Okay, the user asked me to explain what a large language model is in two sentences. Let me start 
by breaking down the key points.

First, a large language model is a type of AI model that can understand and generate text. I 
should mention that it's trained on vast amounts of text data to learn patterns and improve 
performance. Then, I need to highlight its ability to understand and create human-like text. 
Wait, but I need to make sure it's only two sentences. Let me check again. Yes, two sentences. 
Make sure it's concise and covers both the purpose and the main features. Also, check for any 
redundancy. Alright, that should do it.
...done thinking.

A large language model is an AI system designed to understand and generate human-like text, 
trained on vast datasets to learn patterns and improve its performance over time. It can 
comprehend complex languages and create coherent, contextually relevant content, making it a 
powerful tool for tasks like writing, translation, or answering questions."""

#What the ollama model do in the terminal is that it was thinking through how to respond to the prompt.
#It was baiscally talking to itself before responding. While the model that was used throught the assignment gave a more direct reponse right away.
#Running it locally the adavantage is that there is no API cost and the data doesnt leave the computer espeically sensitive information.
#The disadavntage to it , is that the model response quality is low , it talks to itself before responding.
#simce its a smaller model it doenst have the same level of understanding , as bigger models.
#However bigger models need strong GPU and lots of RAM.


