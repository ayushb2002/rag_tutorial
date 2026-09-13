# Retrieval Augmented Generation
Learning Project - Getting up to speed on how RAGs work

## Overview

Below are the steps involved in a typical ingestion pipeline:
1. Loading various types of documents using langchain.
2. Applying different chunking strategies to create sensible chunks based on various splitting methods.
3. Storing chunks in vector storage using OpenAI Embedding models.

Below are the steps involved in a typical retrieval pipeline:
1. Loading the data from vector storage.
2. Using various methods to extract the best-matched chunks based on the query for LLM.
3. Providing the query and the context fetched from storage to LLM model and generating answer. 

## Multi Model RAG Pipeline

A full project built using the following dependencies -
- Poppler
- Tesseract
- Libmagic

The project takes a PDF file as input, breaks it into chunks based on the various type of content stored in the pdf such as tables, images or raw texts. The chunks store the text as a summary of text by using LLM model to convert the large texts into short summary. The retrieval pipeline loads the context from the vector storage based on the similarity between query and content and then both the query and relevant context are sent to LLM model and the generated output is returned  for the user.