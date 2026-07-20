# #we have to use the differnet chunking techniques according to the pdf content 
# #simple fixed size chunking
#remove the # tag to see the separate function 
import os
from dotenv import load_dotenv
load_dotenv()

with open("practice_file.md", "r", encoding="utf-8") as file:
    text = file.read()

# def fixed_size_chunk(text, chunk_size=100):

#     chunks = []

#     for i in range(0, len(text), chunk_size):

#         chunk = text[i: i+ chunk_size]

#         chunks.append(chunk)

#     return chunks

# chunks = fixed_size_chunk(text, 40)

# for i, chunk in enumerate(chunks, 1):  #here 1 means start of indexing as the the index in list starts from 0 but to print that we need to start it from the 1 
#     print(f"Chunk {i}")
#     print(chunk)
#     print("-" * 3)

#Structured based chunking 
import re
def structured_chunk(text):

    pattern = r'(?=# )'    #r means raw string ,(?=...) This is called a Positive Lookahead."Look ahead to see if this pattern exists, but don't consume it."

    chunks = re.split(pattern, text)

    return [chunk.strip() for chunk in chunks if chunk.strip()]  #removing front and end spaces 

chunks = structured_chunk(text)

for chunk in chunks:
    print(chunk)
    print("-" * 30)

#Semantic chunking 
text_semantic = """
AI is a branch of computer science.
It tries to simulate human intelligence.
Machine Learning is a subset of AI.
My name is sakshi .I am student 
"""
from langchain_experimental.text_splitter import SemanticChunker
from langchain_google_genai import GoogleGenerativeAIEmbeddings

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-2",
    google_api_key=os.environ.get("GEMINI_API_KEY")  #for embedding we using the gemini 
)

splitter = SemanticChunker(embeddings)

chunks = splitter.split_text(text_semantic)
for i, chunk in enumerate(chunks, 1):
    print(f"Chunks ={i}")
    print(chunk)