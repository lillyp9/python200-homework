import os
from pathlib import Path
from pypdf import PdfReader
from dotenv import load_dotenv

from llama_index.core import Document, VectorStoreIndex, Settings, SimpleDirectoryReader
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI as LlamaOpenAI

#============= Groundwprl Coffee Q&A Assistant
#load the API
load_dotenv()
if os.getenv("OPENAI_API_KEY"):
    print("API key loaded successfully.")
else:
    raise SystemExit("API key not found. Check your .env file.")
#assert statement to verify and stop if the conditions dont meet.
docs_dir = Path("/home/lilly/python-200/lessons/06_AI_augmentation/resources/groundwork_docs")
assert docs_dir.exists(), f"Document directory not found: {docs_dir}"
print(f"Documents directory found: {docs_dir}")

# === Step 2: Load the Documents ===
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")
Settings.llm = LlamaOpenAI(model="gpt-4o-mini")

print("\nLoading documents...")
documents = SimpleDirectoryReader(str(docs_dir)).load_data()
#How many documen does it have  and what are the file names?

print(f"Loaded {len(documents)} documents:")
for d in documents:
    print(f"{d.metadata.get('file_name', 'unknown')}")

#======= Step 3 : Build the Index and Query Engine ===
index = VectorStoreIndex.from_documents(documents)
print("Building vector index")
query_engine = index.as_query_engine(similarity_top_k=3)
print("Index built. Ready to answer any questions")

#===== Step 4: Query the Assitant 
questions = [
    "What are Groundwork's hours on weekends?",
    "Do you offer any dairy-free milk options?",
    "How does the loyalty program work?",
    "How did Groundwork Coffee get started?",
    "Do you offer catering or wholesale orders?",
]
print("\n" + "=" * 60)
print("Step 4: Five Quieries")
print("=" * 60)

#for statement 
for q in questions:
    print(f"Question: {q}")
    response = query_engine.query(q)
    print(f"Answer: {response}\n")
    top = response.source_nodes[0]
    print(f"Top source: {top.metadata.get('file_name', 'unknown')}")
    print(f"Score: {top.score:.4f}")
    print(f"Text: {top.text[:200]}")

#Reflection:
# The assistant acted very confident when giving the output , the answers matched what the documents say, from the company history to the limited drinks.
#From the time and hours, what holidays be closed as well as the price. As well as the loyalty program.
#However the retrieval score range from 0.32 to 0.76 . The 0.32 is from faq.txt which doesnt surprised me since its frequent questiions asked which means maybe other questi9ons are being asked or left out , thats why the retrieval score is less related.

#==== Step 5 : Find a Failure 
print("\n" + "=" * 60)
print("Step 5: Failure case")
print("=" * 60)

hard_question = "What is Groundwork Coffee's annual revenue and do they think of expanding?"

print(f"\nQuestion: {hard_question}")
response = query_engine.query(hard_question)
print(f"Answer: {response}")
print("All retrieved chunks:")

for i, node in enumerate(response.source_nodes, start=1):
    print(f"  [{i}] {node.metadata.get('file_name', 'unknown')}")
    print(f"Score: {node.score:.4f}")
    print(f"Text:  {node.text[:200]}")

#Reflection
#I asked for the company annual revenue and are they thinking of expanding, i wanted to see how the model would respond.
#I was not suprised that it didnt say I dont know since there isnt any script or prompt indicating  the rules or behavioor the model should display.
#What happened was that it gave the history of how long its been established and how it all started. Then it said what they sell as well as the menu of drinks and the price of it.
#The score is from 0.4 to 0.6 , the system tried to piece it together with what it had , if iit was 0.3 and lower is less related.

#The model sounded confident which is concerning since if anyone didnt read or even give it a second glance , they wouldnt know this is the wrong infromaiton till its given to someone else and they see its not what the question asked fot.
#To improve the system I would need to have a confidence score , setup a prompt , and give examples on the context as well as rules and behvori expectation.
#If the score is below 0.5 then the sysytem should refuse to answer or simply say i do not know, and tell the individual to double chekc indicating you arent sure.


#============ Step 6: Reflection =====
#The LLama setup framework from loadiing the document to building the index and query engine saved me from doing the maunal work from scratch.
#handling chunking, calling the embeddings API, storing vectors, and calculating the cosine similarity , the framwokr did all the heavy load of doing all that with just a few liines of code.
#The tradeoff is that if something breaks , i woudl have to look in closer to deug it , but for big projects it is worh it . 

#Another case to use build a LLama pieline and use RAG is a med spa where it has supplies , client information, insurance , finacial options, etc.
#It would be a nightmare for the doc , nurse even front desk having to go through all the paperwork of different things.
#A RAG assitant could help answer the questons such as " What client is schedlue foor this and that , are we partner with this isurnace to accept it , or how much supplies is there and of what?."
#Its much faster then having to do it manually or not having the notes up to date in  the system.
#Its better to have your own assitant then a public LLM where senesitive information could be used to trained their model which is a security risk.

#Even when retieving the right chunks , the model coud still mess up when generating the response , it might misread the chunks, over-summarize and lose important details, or blend the retrieved information with its oown training data, and just make up whattever response.
#RAG does fix the retrieval problem but the model is still an OpenAI. Thats why having a huammn review on top of the RAG even if it looks like its working.

#Extensiion A:  Side by Side Comparison (Moderate)
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
    
#Load the doc as plain text for keyword search 
keyword_docs = {f.name: f.read_text() for f in docs_dir.glob("*.txt")}

print("\n" + "=" * 60)
print("Extension A: Keyword RAG vs Semantic RAG")
print("=" * 60)

for q in questions:
    print(f"Question: {q}")

    #keyword RAG
    keyword_result = simple_keyword_retrieval(q, keyword_docs)
    keyword_doc, keyword_content = keyword_result[0]
    print(f"[KEYWORD] Selected: {keyword_doc}")
    print(f"Snippet: {keyword_content[:150]}")

    #Semantic RAG (LlamaIndex)
    semantic_response = query_engine.query(q)
    top = semantic_response.source_nodes[0]
    print(f"[SEMANTIC] Top source: {top.metadata.get('file_name', 'unknown')}")
    print(f"Answer: {semantic_response}")

#Extension A comment :
#Keyword RAG works on questions where query and documents share the similar words , It fails when a word that
#is similar to the other words but the machine ca not make the connections. 
#Even if there is a similar keyword on the right document , it returns  the whole document. The Semantic RAG is better to giive the real answer.