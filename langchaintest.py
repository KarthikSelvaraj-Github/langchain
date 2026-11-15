from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
import numpy as np
from langchain_community.vectorstores import FAISS

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
import os


# Create a sample file to load
text = """LangChain is a framework for building AI apps.
It helps connect language models with your own data.
RAG means Retrieval-Augmented Generation."""

##with open("sample.txt", "w") as f:
 #   f.write(text)

#print("sample.txt file created!")

loader = TextLoader("sample.txt")
documents = loader.load()

#print("Number of documents:", len(documents))
#print("Content:")
#print(documents[0].page_content)

loader = PyPDFLoader("LangSmith.pdf")
pages = loader.load()

#print("Total Pages:", len(pages))

#for page in pages:
   # print("Page Number:", page.metadata.get("page_number", "Unknown"))
   # print(page.page_content)


#splitter
long_text = """Python is a programming language. It is easy to learn.
Streamlit is used to build web apps. LangChain helps build AI apps.
RAG combines search with AI models. Embeddings turn text into numbers.
Vector stores save those numbers. Retrievers find the best matches."""

# print("Total characters:", len(long_text))


""" splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,      # each chunk is about 80 characters
    chunk_overlap=10    # 10 characters repeat between chunks
) """

""" chunks = splitter.split_text(long_text)

print("Number of chunks:", len(chunks))
for i, chunk in enumerate(chunks):
    print(f"\nChunk {i+1}:")
    print(chunk) """



""" splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,      # each chunk is about 80 characters
    chunk_overlap=10    # 10 characters repeat between chunks
)
documents = PyPDFLoader("LangSmith.pdf").load()
chunks = splitter.split_documents(documents)
print(chunks) """

""" # A small, free, popular embedding model
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

print("Embedding model is ready!") """

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
#embeddings
""" load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")


# A small, cheap, high-quality OpenAI embedding model
openai_embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key = api_key)

print("OpenAI embedding model is ready!")

text = "I love programming in Python."

vector = openai_embeddings.embed_query(text)

print("How many numbers in this vector?", len(vector))
print("First 5 numbers:", vector[:5])


def similarity(a, b):
    a, b = np.array(a), np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

v1 = openai_embeddings.embed_query("I like dogs")
v2 = openai_embeddings.embed_query("I love puppies")
v3 = openai_embeddings.embed_query("The Pomeranian breed is small")
v4 = openai_embeddings.embed_query("The horse runs fast")

print("dogs vs puppies (similar):", round(similarity(v1, v2), 2))
print("dogs vs Pomeranian (different):", round(similarity(v1, v3), 2))
print("dogs vs horse (different):", round(similarity(v1, v4), 2))
 """


texts = [
    "Python is a programming language.",
    "Streamlit is used to build web apps.",
    "Dogs are friendly animals.",
    "Cats like to sleep a lot.",
    "Tigers like to be dangerous.",
    "Microsoft recent outage affected many users."
]
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
#print(embeddings)
vector_store = FAISS.from_texts(texts, embeddings)

print("All texts are stored in the vector store!")
print("Total texts stored:", len(vector_store.index_to_docstore_id))
print("Vector dimension:", vector_store.index.ntotal)
print(vector_store)

query = "any recent outage happened?"

results = vector_store.similarity_search(query, k=1)   # k=2 means top 2 matches

print("Top matches for:", query)
for r in results:
    print("-", r.page_content)

#vector_store.save_local("my_store")

vector_store = FAISS.load_local("my_store", embeddings, allow_dangerous_deserialization=True)

retriever = vector_store.as_retriever(search_kwargs={"k": 1})

question = "What is used to build web apps?"

results = retriever.invoke(question)

print("Question:", question)
print("\nRelevant information found:")
for r in results:
    print("-", r.page_content)


# 1. Prompt: a template with a blank {topic} to fill in
prompt = ChatPromptTemplate.from_template("Explain {topic} in one simple sentence.")

# 2. Model: the OpenAI chat model
model = ChatOpenAI(model="gpt-4o-mini")

# 3. Output parser: turn the model's reply into plain text
output_parser = StrOutputParser()

print("All three pieces are ready!")

chain = prompt | model | output_parser

answer = chain.invoke({"topic": "Python"})

print(answer)


tip_prompt = ChatPromptTemplate.from_template(
    "Give one short, friendly tip for a beginner learning {skill}."
)

tip_chain = tip_prompt | model | output_parser

print(tip_chain.invoke({"skill": "Python"}))



prompt = ChatPromptTemplate.from_template("Explain {topic} in one simple sentence.")
model = ChatOpenAI(model="gpt-4o-mini")
chain = prompt | model | StrOutputParser()

# This run is automatically recorded in LangSmith.
print(chain.invoke({"topic": "LangSmith"}))

llm = ChatOpenAI(model="gpt-4o-mini")
prompts = ["Give one short, friendly tip for a beginner learning Python", "What is used to build web apps?", "Explain LangSmith in one simple sentence."]
for p in prompts:
 print(llm.invoke(p).content)
