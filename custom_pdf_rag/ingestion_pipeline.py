import sys
sys.path.insert(0, "vendor/")

from dotenv import load_dotenv
load_dotenv()

import pymupdf
import copy
import os
from langchain_text_splitters import MarkdownTextSplitter
import pymupdf4llm
pymupdf4llm.use_layout(False)

from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

# Finding all pdf files
pdf_files = [f"docs/{i}" for i in os.listdir("docs/")]
md_files = []

# Converting PDF files into md files
for _pdf in pdf_files:
    print(f"Reading {_pdf.split(r'/')[1]}.")
    doc = pymupdf.open(_pdf)
    doc_headers = pymupdf4llm.IdentifyHeaders(doc, max_levels=3)
    doc_md = pymupdf4llm.to_markdown(doc, hdr_info=doc_headers)
    print(f"Converted to md file.")
    md_files.append(copy.deepcopy(doc_md))
    print("="*50)

# Using langchain text splitter
print("Creating chunks.")
splitter = MarkdownTextSplitter(chunk_size=800, chunk_overlap=150)
# chunking
chunks = splitter.create_documents(md_files)
print(f"Converted {len(md_files)} into {len(chunks)} chunks!")
print("="*50)

print("Storing into vector db.")

embeddings_model = OpenAIEmbeddings(
    model="text-embedding-3-small"
)

vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings_model,
    persist_directory="db/pdf",
    collection_metadata={"hnsw:space": "cosine"}
)

print("Stored in vector db successfully!")
print("="*50)