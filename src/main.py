import os
from langchain.prompts import (PromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate, MessagesPlaceholder)
from langchain.chains import ConversationChain
from dotenv import load_dotenv
import streamlit as st
from streamlit_chat import message
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.memory import ConversationBufferWindowMemory
from langchain_google_genai import ChatGoogleGenerativeAI
import google.generativeai as genai

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=api_key)

st.set_page_config(layout="wide", page_title="Simple QA Agent")

if 'responses' not in st.session_state:
    st.session_state['responses'] = ['How can I help you?']
if 'users_question' not in st.session_state:
    st.session_state['users_question'] = []
if 'citations' not in st.session_state:
    st.session_state['citations'] = []

@st.cache_resource(show_spinner=False)
def load_model():
    llm = ChatGoogleGenerativeAI(model='gemini-1.5-pro-latest', temperature=0.3)
    return llm

system_msg_template = SystemMessagePromptTemplate.from_template(template="""
Answer the question given from the provided context and make sure to provide the answers which are in the context don't try to make your own answers. If the answer is not in the context then simply say I don't know about that as it is not in the given context.
""")

human_msg_template = HumanMessagePromptTemplate.from_template(template="{input}")
prompt_template = ChatPromptTemplate.from_messages([system_msg_template, MessagesPlaceholder(variable_name="history"), human_msg_template])

if 'buffer_memory' not in st.session_state:
    st.session_state['buffer_memory'] = ConversationBufferWindowMemory(k=5, return_messages=True)

st.session_state.conversation_chain = ConversationChain(memory=st.session_state['buffer_memory'], prompt=prompt_template, llm=load_model(), verbose=False)

conversation = st.session_state.conversation_chain

response_container = st.container()
text_container = st.container()

def find_match(question):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    question_embeddings = embeddings.embed_query(question)
    vector_data = FAISS.load_local('Data/faiss_index', embeddings, allow_dangerous_deserialization=True)
    docs = vector_data.similarity_search_by_vector(question_embeddings)
    context = "\n".join([doc.page_content for doc in docs])
    citations = [
        {"source": doc.metadata.get("source", "Unknown source"), "content": doc.page_content[:20]}
        for doc in docs
    ]
    return context, citations

with text_container:
    query = st.text_input("Ask a question", key='input')
    if query:
        with st.spinner("Processing..."):
            context, citations = find_match(query)
            response = conversation.predict(input=f"Context: \n{context}\nQuestion: {query}")
            st.session_state['users_question'].append(query)
            st.session_state['responses'].append(response)
            st.session_state['citations'].append(citations)

# Display the conversation
with response_container:
    if 'responses' in st.session_state and st.session_state['responses']:
        for i in range(len(st.session_state['responses'])):
            if i < len(st.session_state['users_question']):
                message(st.session_state['users_question'][i], is_user=True, key=str(i) + '_user')
            message(st.session_state['responses'][i], key=str(i))
            if i < len(st.session_state['citations']):
                st.markdown("**Citations:**")
                for citation in st.session_state['citations'][i]:
                    st.markdown(f"- {citation['source']}: {citation['content']}")









