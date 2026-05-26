from dotenv import load_dotenv
import os

if load_dotenv():
    print("API key loaded successfully.")
else:
    print("Warning: could not load API key. Check your .env file.")

#========= RAG Concepts =======

#Scenario A: A legal team wants an assistant that can answer questions about their internal policy library — hundreds of PDFs that are updated every quarter.
#The documents change to often for fine-tuning to keep up , it would take time and they are too big to fit in just one prompt.
#RAG lets the system pull only the important part of the documents to answer the question.

#Scenario B: A startup wants their model to write product copy in a very specific brand voice — a dry, minimalist style that does not appear much online. They have 3,000 examples their in-house writers produced over the years.
#Fine-tuning would be a good option since the style is consistent and 3,000 example is enough to train data. The fine-tuning puts the style and tone into the model. 

#Scenario C: A data analyst needs to ask an LLM questions about a single two-page report she just received. She does not need this to work for any other document.
#Prompt engineering wuld be a good option siince the document is small enough to pasta riight into the model prompt. Theres not need for fine-tuning(no training data) or RAG( documents in library).


#========Concept Question 2 
#Having a hallucination is more harmful then a simple "I dont know" because the indiividual can trust the answer blindly and without checking if its true or not , it coukd run with it. If the model said "Im not sure" then the individua knows that they NEED to verify. 
#For example what if the model tells a civil engineer that a certain calculation is correct for a building design and the egineer does not chekc but run with it . Not only could it cause finiiacl loss , the egineer in question would be held libabale for the damages caused byt the buidiing either collapsing or injuries people 
#would have obtain just from being there whne the buidling collapses. If the model said "Heres the calculation, I am not sure its correct."
#Then they would have to verify and it would prevent damages.

#Now the tone the model uses which is being confidence , obviosuly that lets us think its accurate when its not.
#So conetext does matter and double chceking.


#========Concept Question 3
#steps = [
    #"1. Extract text from source documents",
    #"Read the raw text from the PDF, web pages, or other source 

    #"2. Split text into chunks",
    # Break long text/ paragraphs into smaller pieces so each one fits in a prompt.

    #3."Convert text chunks into embeddings",
    #Turn each chunk into a numerical representation with its meaning.

    #"4. Receive the user's query",
    # The individual ask questions or request into the app.

    #"5. Embed the user's query",
    #Turn the question into a numerical representation as the chunks.

    #"6. Retrieve the most relevant chunks",
    #Compare the query vector against the the chunk vectors and pick the closest match.
      
    #"7.Inject retrieved chunks into the prompt",
    #Build a prompt that contains both question and retrieved the context.

    #"8. Generate a response from the LLM",
    #Send the prompt to the model whixh the answered is based o the retrieved context.

#]

import string 

def simple_keyword_retrieval(query, documents, verbose=True):
    """Keyword retrieval using token overlap scoring."""
    stopwords = {
        "a", "an", "the", "and", "or", "in", "on", "of", "for", "to", "is",
        "are", "was", "were", "by", "with", "at", "from", "that", "this",
        "as", "be", "it", "its", "their", "they", "we", "you", "our"
    }
    translator = str.maketrans("", "", string.punctuation)

    query_words = {
        w.translate(translator)
        for w in query.lower().split()
        if w not in stopwords
    }
    if verbose:
        print(f"\nQuery tokens (filtered): {sorted(query_words)}")

    scores = []
    for name, content in documents.items():
        content_words = {
            w.translate(translator)
            for w in content.lower().split()
            if w not in stopwords
        }
        overlap = query_words & content_words
        score = len(overlap)
        scores.append((score, name, content))
        if verbose:
            print(f"[{name}] overlap={score} -> {sorted(overlap)}")

    scores.sort(reverse=True)
    best = next(((name, content) for score, name, content in scores if score > 0), None)
    if best:
        if verbose:
            print(f"\nSelected best match: {best[0]}")
        return [best]
    else:
        if verbose:
            print("\nNo overlapping keywords found.")
        return [("None found", "No relevant content.")]


query = "What are your hours on the weekend?"

documents = {
    "menu.txt": "We serve espresso, lattes, cappuccinos, and cold brew. Pastries include croissants and muffins baked fresh daily. Oat milk and almond milk are available.",
    "hours.txt": "We are open Monday through Friday from 7am to 7pm. On weekends we open at 8am and close at 5pm. We are closed on Thanksgiving and Christmas Day.",
    "hiring.txt": "We are currently hiring baristas and shift supervisors. Send your resume to jobs@groundworkcoffee.com.",
    "loyalty.txt": "Join our loyalty program to earn one point per dollar spent. Redeem 100 points for a free drink of your choice.",
}

#===== Keyword Question 1 :
print("n=== Keyword Question 1 ===")
result = simple_keyword_retrieval(query, documents, verbose=True)
print(f"Selected: {result[0][0]}")

#Comment Question 1 : loyalty.txt was selected because the query word "your"

#===== Keyword Question 2
print("\n Keyword Q2")
query = "Do you have anything without caffeine?"
result = simple_keyword_retrieval(query, documents, verbose=True)
print(f"Selected: {result[0][0]}")

#The funcitions returned "None Found" beacause there is no documents that contain the word coffee or without . 
#The keword RAG got it wrong. The righanswer would be in menu.txt, simce thats where it ll mention milk, almond milk and even if it 
#doesnt contain caffeine at leats it relevanat to the search. 

#======= Keyword Question 3 
# It would choose the loyalty.txt since rewards is just another word for loyalty . But simce RAG might not see the connection. 
#Im not surprised if it said None found .

print("\n=== Keyword Q3 ===")
query = "How do I sign up for rewards?"
result = simple_keyword_retrieval(query, documents, verbose=True)
print(f"Selected: {result[0][0]}")

# My prediction was correct , it did noot make the connection. This is an exampe of failure of keyword search.

#====== Semantic RAG Concepts=========

#Semantic Question 1 : 
#A vector embedding is list of number the represent the meaning o a piece of text. Text with the similar meaniing gets the similar numbers. 
#so there is a comparison mathematically.

#Two text chunks have cosine similarity scores of 0.85 and 0.30 with a given query. Which chunk is more relevant, and what does that number tell you about the relationship between the texts?
#Cosine similarity goes from -1 to 1 or 0 to 1 in practice for text. So 0.85 means the two texts are very close in the meaning while 0.30 means they are less related.

#Semantic search can find relevant chunks without matching exact words because 
#embedding capture the meaning, not what it actually means. For example, "Loyalty" and "rewards" end up close in vector space because they appear in similar context in the training data even if they dont share the same letter.

#Semantic Q2 comparison table:
#
# | Feature                    | Keyword RAG                       | Semantic RAG                          
# |----------------------------|-----------------------------------|
# | What is compared?          | Exact word overlap                | Meaning , similarity between vectors      
# | What is retrieved?         | Full document                     | Certain chunks of text               
# | Can it handle synonyms?    | No                                | Yes                                   
# | Storage format             | Plain text dictionary             | Vector data or section             
# | Relevance score            | Number of overlapping keywords    | Cosine similarity between vectors  


#======= Llama-index ======
from pathlib import Path
from pypdf import PdfReader
from llama_index.core import Document, VectorStoreIndex, Settings
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI as LlamaOpenAI


pdf_path = "/home/lilly/python-200/lessons/06_AI_augmentation/resources/brightleaf_pdfs"

#Configure llamaIndex to us ethe OpenAI for both embeddiing and generating 

# Configure LlamaIndex to use OpenAI for both embedding and generation
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")
Settings.llm = LlamaOpenAI(model="gpt-4o-mini")

print("\nLoading Brightleaf PDFs...")
#documents = SimpleDirectoryReader(pdf_path).load_data() 
#Note SimpleDirectory was returning binary garbage.
documents = []

for pdf_file in Path(pdf_path).glob("*.pdf"):
    reader = PdfReader(str(pdf_file))
    text = "\n".join(page.extract_text() or "" for page in reader.pages)
    if text.strip():
        documents.append(Document(text=text, metadata={"file_name": pdf_file.name}))
        print(f"  Loaded {pdf_file.name}: {len(text)} chars")
    else:
        print(f"  ⚠️  {pdf_file.name} had no extractable text")

#load doc 
print(f"Loaded {len(documents)} documents.")
# vector index 
print("Building vector index (this calls the embeddings API)")
index = VectorStoreIndex.from_documents(documents)
print("Vector Index ready.")

#==========LLAMAINDEX Question 1 ====

questions = [
    "What employee benefits does BrightLeaf offer?",
    "What are BrightLeaf's security policies?",
]

query_engine = index.as_query_engine(similarity_top_k=3)

for q in questions:
    print(f"\n--- Question: {q} ---")
    response = query_engine.query(q)
    print(f"\nAnswer: {response}\n")
    print("Retrieved chunks:")
    #loop to retrieve 
    for i, node in enumerate(response.source_nodes, start=1):
        #For each of the 3 retrieved source nodes:
        print(f"[{i}] score={node.score:.4f}")
        #first 150 characters of the chunk text
        print(f"text: {node.text[:150]}")

# After creating a for function to get the file path , switching to pypdf for PDF reading  and able to have the model 
# say words instead of random binary. I was able to see that both queries return chunks that are relevant this time. Such as benefit query that pulls chunks about health insurnace, PTO, etc. a
# And the security query pulled the chunks about data handling and access rules. The model tone was confident , as it gave specific details from the document. 

