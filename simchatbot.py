'''
UTF - 8
A RAG - LLM application that uses Llama 2 and Gradio
Built by Varad Puntambekar 
05 March 2025 - 3 Wharf Mews, OX26DJ Oxford (AirBnb Rasika Graduation)
Copied from 
https://github.com/varadpuntambekar/PDF-RAG-with-Llama2-and-Gradio/tree/master
'''
#import libraries
import yaml
import torch
import gradio as gr
from PIL import Image
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.document_loaders import PyPDFLoader, WebBaseLoader, TextLoader
from langchain_huggingface import HuggingFacePipeline
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from typing import List, Dict
from transformers import AutoModelForCausalLM , AutoTokenizer, pipeline
from langchain.prompts import ChatPromptTemplate



# There have been some changes to the libraries of how to store memory of chats. Perhaps I can find it out later

#First iteration is always building a skeleton.

class document_handler(object):
    def __init__(self, data_dir = "./data"):
        self.data_dir = data_dir
        #Splitting document for code script
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size = 500,
            chunk_overlap = 100,
            separators=["\n\n", "\n", "def", "class"]
        )

        self.embedding_model = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2")

    def load_split_doc (self):
        pass

    def create_vectorstore (self):
        pass

class LLMHandler(object):
    def __init__(self):
        pass

    def rag_pipeline(self):
        pass
class RAGChain(object):
    def __init__(self):
        pass

    def initialize(self):
        pass

    def process_query(self,query):
        pass

class UI(object): #can change later on to see how it works.
    def __init__(self):
        pass
    
    def respond (self):
        pass

    def launch_app (self):
        pass
    







class simchatbot(object):
    def __init__(self, filepath: str, embedding_model: str, llm_model: str, query: str):
        
        #conversation_history_manager
        self.conversation_history = List[Dict[str, str]] = []
        #doc_loader
        loader = TextLoader(filepath)
        documents = loader.load()

        #text_splitter
        text_splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap = 100, 
                                                       separators= ["\n", "\n\n", "def", "class"]) #specific for reading code scripts
        splits = text_splitter.split_documents(documents)
        #embed_docs (tiktoken or any other embedding technique)
        embeddings = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2")
        #retriever
        vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
        retriever = vectorstore.as_retriever(search_kwargs = {"k" : 3})

        #prompt template
        template = """
        You are an expert Discrete Event Simulation, Cervical Cancer and Health System Researcher. The people asking the questions are policymakers and clinicians
        who have no professional background in the above fields. Answer the questions in a way that is easier for people
        from different professional backgrounds to answer

        Context : {context}

        Question : {question}

        Your response should refer to parts of the code and explain to the user what this part of the code does and how it is placed
        in the context of the entire codebase and how it affects the implementation and interpretation of model findings.
        If the answer to the question is not available in the given codebase, then give responses similar to I don't know the answer to this,
        or this document cannot satisfactorily answer your query.
        """
        prompt = ChatPromptTemplate(template)
     

        #LLM model
        model_name = "meta-llama/Meta-Llama-3.2-1B"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype = torch.float16, device_map = "auto"
        )
        
        hf_pipeline = pipeline(
            task = "text-generation",
            model = model,
            tokenizer = tokenizer,
            max_new_tokens = 512,
            temperature = 0.7,
            repetition_penalty = 1.1
        )

        llm_model = HuggingFacePipeline(pipeline = hf_pipeline)
        #Chain
        rag_chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | llm_model
            | StrOutputParser()

        )

        #Store_conversation_history

        #User_query_input.
        
        pass

class ui_interface(object):
    '''
    A gradio object that creates the UI for the Chatbot
    '''












class code_chatbot (object):
    def __init__(self, config_path =  "../config.yaml"):
        pass