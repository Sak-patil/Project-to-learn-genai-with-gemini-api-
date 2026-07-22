#Going ot use the Chroma db to store embedding vectors locally 
#chunking and embedding 
#here we are using both semantic and lexical search 
import os
from dotenv import load_dotenv
load_dotenv()

with open("practice_file.md", "r", encoding="utf-8") as file:
    text = file.read()

#Semantic chunking 
text_semantic = """
AI is a branch of computer science.
It tries to simulate human intelligence.
Machine Learning is a subset of AI.
My name is sakshi .I am student 
"""
from langchain_experimental.text_splitter import SemanticChunker
from langchain_voyageai import VoyageAIEmbeddings  #we are going ot use the VOYAGE AI embedding model provider 

embeddings = VoyageAIEmbeddings(
    model="voyage-3",
    api_key=os.getenv("VOYAGE_API_KEY")
)

splitter = SemanticChunker(embeddings)

chunks = splitter.split_text(text)
# for i, chunk in enumerate(chunks, 1):
#     print(f"Chunks ={i}")
#     print(chunk)
chunks = [c.strip() for c in chunks if c.strip()]
#To see the embedding vectors and its dimensions
vectors = embeddings.embed_documents(chunks)
# for i, vector in enumerate(vectors):
#     print(f"Chunk {i + 1}")
#     print("Vector:", vector)
#     print("Vector dimensions:", len(vector))
#     print("-" * 50)

#Lexical search
tokenized_chunks = [
    chunk.lower().split()
    for chunk in chunks
]
from rank_bm25 import BM25Okapi
bm25 = BM25Okapi(tokenized_chunks)



#Vector Dastabase Chromadb
import chromadb
 #Create/open a persistent ChromaDB database
client = chromadb.PersistentClient(
    path="./chroma_db"   #means the database file get stoed in local folder we have made in this directory (chroma_db)
)
#Create or get a collection
collection = client.get_or_create_collection(
    name="my_documents"
)

# 3. Create unique IDs
ids = [f"chunk_{i}" for i in range(len(chunks))]

metadatas = [
    {"source": "practice_file.md"}
    for i in range(len(chunks))
]

# 4. Add chunks and their corresponding vectors
collection.add(
    ids=ids,
    documents=chunks,
    embeddings=vectors,
    metadatas=metadatas
)
# print("After:", collection.count())
print("Data added successfully!")

#Now converting the user request into the embedding vector but we have to use the same embedding model as we have used for chunking of files 
print("Ask question")
query = input(">")
#for sematic we have to convert the query into embedding vector to compare
query_vector = embeddings.embed_query(query)
results = collection.query(
    query_embeddings=[query_vector],
    n_results=2 #this means give 3 most similar chunks 
)

#print(results)
retrieved_chunks = results["documents"][0]

#for lexical we have to tokanize the query 
tokenized_query = query.lower().split()
scores = bm25.get_scores(tokenized_query)
top_chunks = bm25.get_top_n(
    tokenized_query,
    chunks,
    n=2
)

#now we got the both result from semantic and leximal but have to send it with scores means both have find the same result have more score and then send it to gemini 
#for that we are going to use RRF function to give score 
def reciprocal_rank_fusion(semantic_ranked, lexical_ranked, k=60):
    scores = {}
    for rank, chunk in enumerate(semantic_ranked):
        scores[chunk] = scores.get(chunk, 0) + 1 / (k + rank + 1)
    for rank, chunk in enumerate(lexical_ranked):
        scores[chunk] = scores.get(chunk, 0) + 1 / (k + rank + 1)
    # sort by combined score, highest first
    return sorted(scores.keys(), key=lambda c: scores[c], reverse=True)

all_chunks = reciprocal_rank_fusion(retrieved_chunks, top_chunks)[:3]  # take top 3 fused
context = "\n\n".join(all_chunks)
print(context)

prompt = f"""
Answer the question using the context below.

Context:
{context}

Question:
{query}

If the answer cannot be found in the context, say:
"I don't know based on the provided context."
"""
from google import genai

client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)

response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents=prompt
)

print(response.text)