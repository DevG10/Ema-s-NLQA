# LLM Query Agent

A conversational AI agent capable of answering questions based on lecture notes and other resources using advanced NLP techniques and pre-trained language models.
[Link to the approach](https://docs.google.com/document/d/1U0yRFxvu7T2k9gsM1WRPiXP1HqKt1g9_2qx8U64sjGg/edit?usp=sharing)
## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Data Collection and Preprocessing](#data-collection-and-preprocessing)
- [Creating and Storing Embeddings](#creating-and-storing-embeddings)
- [Building the Query Agent](#building-the-query-agent)
- [Future Work](#future-work)

## Introduction
This project aims to create an AI agent that can answer questions from a specific context (lecture notes, web resources). The agent utilizes NLP techniques and pre-trained language models to provide accurate and meaningful responses.

## Features
- Data Collection from PDFs and websites
- Data Preprocessing using NLP techniques
- Embedding creation and storage using FAISS
- Query processing and response generation using Google's Gemini Pro LLM
- Follow-up question handling using Langchain's Conversation Buffer Memory
- User interface built with Streamlit

## Installation
### Prerequisites
- Python 3.9+
- Pip

### Steps
```bash
# Clone the repository
git clone https://github.com/DevG10/Ema-s-NLQ.git

# Navigate to the project directory
cd Ema-s-NLQ

# Install the dependencies
pip install -r requirements.txt
````

## Usage
### Run the main.py file
```bash
cd src
python -m main.py
````
#### Note: Before running the main file ensure that you have first created a .env file and pasted your gemini key as GOOGLE_API_KEY = 'your_api_key' inside your .env file
## Data Collection and Preprocessing
### Data Sources
* Lecture Notes
* Websites from well-known institutes
### Tools and Techniques
* BeautifulSoup for web scraping
* PyMuPDF for PDF extraction
* NLTK for preprocessing (stopwords removal, lemmatization, etc.)

## Creating and Storing Embeddings
- Embeddings convert text into numerical vectors, allowing for semantic search and retrieval.
- To create and store the embeddings, I used FAISS which is open-source and fast too as it is made using C++
- The embeddings were calculated using sentence transformer's all-MiniLM-L6-v2 pre-trained model

## Building the Query Agent
### Query Processing
- The query given by the user is first converted to the embeddings using the same model which was previously used to create the embeddings of the lecture.
- This converted query is then searched using vector similarity search and the matching docs are returned
### Handling Follow-Up Questions
- To enable follow-up questions, Langchain's Conversation Buffer Memory is used to keep track of the context and questions asked.
  
## Future Work
- Potential improvements and additional features:
- Enhancing the accuracy of responses
- Adding more data sources
- Improving the user interface
