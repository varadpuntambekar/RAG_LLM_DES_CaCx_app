'''
Building a basic chatbot to understand how to call ollama as the first building block of the RAG App
'''
from langchain_ollama import OllamaLLM
from langchain_ollama import OllamaEmbeddings
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import os
import pprint
import pymupdf
from PIL import Image
import io

embedding_model = OllamaEmbeddings(model = 'nomic-embed-text')

directory_path = "data/"

def load_docs (directory_path):
    '''
    Loads all PDF files from a Folder (directory) provided to the function
    Each doc is a Langchain_Document type object that has metadata and page_content
    Function loads one page at a time and converts it into one document with all the pages.
    '''
    print(f"===Loading docs from folder {directory_path}===")
    
    docs = []
    #loading one file at a time
    for file in os.listdir(directory_path): 
        loader = PyPDFLoader(os.path.join(directory_path, file), mode="single",)
        #loading one page at a time (hence there are two loops
        one_doc = loader.load()
        docs.extend(one_doc)
        
          
            
    print(f"=== Loaded {len(docs)} documents ===")
    return docs # a list[Documents] type object that has a list of all documents in that folder

#loaded_docs = load_docs(directory_path)

def load_pages_from_docs(directory_path):
    '''
    Loads all pdf files from the directory, Splits every doc into pages and attaches pages as a metadata to those files
    Important to trace where this chunk came from and then display it as well
    '''

    print(f"======Loading docs from folder {directory_path}============")
    docs = []
    for file in os.listdir(directory_path):
        loader = PyPDFLoader(os.path.join(directory_path, file), mode = 'page')
        #loading one page at a time I hope
        one_page = loader.load()
        docs.extend(one_page)

    print(f"==Loaded {len(docs)}=== documents")
    #print(docs[0].metadata)
    
    return docs

def split_docs_with_metadata(document_list):
    '''
    Accepts a list of documents to be split. This split makes chunking and embedding more efficient
    Creates a list of documents type object with every chunk having the same metadata as the parent document
    '''
    #initializing the text splitter, here I use a premade text splitter specified by langchain
    
    text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 500,
    chunk_overlap = 100,
    length_function = len)

    chunked_docs = [] #list of chunks

    for doc in document_list:
        print(f"Splitting Doc {doc.metadata.get('source')}")
        doc_chunks = text_splitter.split_documents([doc]) #doc_chunks is a list of chunks which retains the metadata of the parent
        for chunk in doc_chunks:
            chunked_docs.append(chunk)

    print("===All Docs Split===")
    print(f"===Total Chunks = {len(chunked_docs)}")
    return chunked_docs

#chunk_list = split_docs_with_metadata(loaded_docs)

def load_or_create_vectorstore (document_list, embedding_model, persist_directory, collection_name):
    os.makedirs(persist_directory, exist_ok= True)

    try:
        vectorstore = Chroma(
            embedding_model,
            persist_directory= persist_directory,
            collection_name= collection_name,
            
        )
        if vectorstore._collection.count() ==0:
            print("Existing vector store is empty, Creating a new vectorstore")
            raise FileNotFoundError
        
        print(f"Loading existing vectorstore {vectorstore._collection_name} with {vectorstore._collection.count()} embeddings")
        return vectorstore
    
    except (FileNotFoundError, ValueError):
        print("Creating a new vectorstore")
        vectorstore = Chroma.from_documents(
            document_list,
            embedding_model,
            persist_directory= persist_directory,
            collection_name=collection_name
        )

        print(f"Created new vectorstore with name {vectorstore._collection_name} with {vectorstore._collection.count()} embeddings")
        return vectorstore

def create_vectorstore(document_list, embedding_model, persist_directory, collection_name):
    '''
    Creates a ChromaDB vectorstore. Embeds the loaded docs and into a vectorstore
    This vectorstore will then need not be created again. 
    Creating the vectorstore takes the most amount of time, 
    '''
    
    print("Vectorstore created")
    vectorstore = Chroma(
        collection_name= collection_name,
        embedding_function=embedding_model,
        persist_directory=persist_directory,
    )
    print (f"Total docs to process = {len(document_list)}")
    counter = 0
    for doc in document_list:
        print(f"Embedding {doc.metadata.get('source')} file right now")
        print(f"Embedding doc {counter} out of  {len(document_list)} total docs docs")
        vectorstore.add_documents([doc])
        counter += 1
    print(f"Created vectorstore named {vectorstore._collection_name} with {vectorstore._collection.count()} embeddings ")

#vector_store = create_vectorstore(
#     chunk_list, embedding_model, 'chroma_db_2.0', 'chunk_store' 
# )

def load_vectorstore(persist_directory, collection_name, embedding_model):
    '''
    Loads an existing vectorstore from a Chroma DB file so that I don't have to create the vectorstore again and again which takes soo much time.
    '''
    vectorstore = Chroma(
        collection_name=collection_name,
        persist_directory=persist_directory,
        embedding_function=embedding_model
    )
    print(f"Vectorstore named {vectorstore._collection_name} with {vectorstore._collection.count()} embeddings has be loaded to the vectorstore")
    return vectorstore

def add_file_to_db(documents, vectorstore):
    '''
    Checks if new documents have been added to the directory, if yes then chunk those documents 
    and add those documents to the already created vectorstore.
    '''
    pass




chroma_directory = 'chroma_db_2.0'
chroma_collection = 'chunk_store'
#loaded_vector_db = load_vectorstore (chroma_directory, chroma_collection, embedding_model )

def relevant_chunks(vector_db, query, n_results = 3):
    '''
    Takes in a query, returns a list[Documents] type object of relevant chunks that are closest to the query
    '''
    retriever = vector_db.as_retriever(search_kwargs = {"k": n_results})
    retrieved_docs = retriever.invoke(query)
    for result in retrieved_docs:
        print(result.metadata['source'])

    return retrieved_docs

#Now finally generating a response from an LLM
def chat():
    '''
    Takes in a query, returns relevant chunks, adds them to the context and generates a response
    '''
    human_message = input("Which document would you like to query (RAG) ")
    get_chunks = relevant_chunks(human_message)
    context = "\n\n".join([chunk.page_content for chunk in get_chunks])

    
    


    print(f"The retrieved document is {get_chunks[0].metadata}.  \n The context is as follows \n {context}")
    template = '''
        You are a question answer answering assistant. Use the following pieces of data from the retrieved context only to answer the question
        If you don't know the answer to the question then answer by saying I don't know or answering this question is out of my scope
        Use three sentences maximum and keep your explanations simple and concise as if explaining a beginner.

        "Context": {context}
        "Question":{human_message}
        "Answer":

        '''
    prompt = ChatPromptTemplate.from_template(template)

    llm_model = OllamaLLM(model="llama3.2:1b")
    
    rag_chain = (
            {"context": lambda x: context, "question":RunnablePassthrough()}
            |prompt
            |llm_model
            |StrOutputParser()
    )
    while True:
        ask_doc = input(f"What would you like to ask to the doc {get_chunks[0].metadata['source']}? ")
        if ask_doc.lower() == 'exit':
            print("It was nice chatting with you, Have a great Day")
            break
        else:
            result = rag_chain.invoke(human_message)
            print("AI Says: ", result)
            context += f"\nUser:{human_message}\n AI: {result}  "


#Display chunk in document
def chunk_in_doc():
    '''
    When we retrieve a chunk from the list of documents, This function matches the page number from the chunk metadata
    with the metadata of the document list and then highlights the information in the document
    '''
    # docs = load_pages_from_docs(directory_path)

    # print(docs[0].metadata)
    
    # page_chunks = split_docs_with_metadata(document_list=docs)
    
    chroma_page_directory = 'chroma_db_page_2.0'
    chroma_page_collection = 'chunk_page_storea_2.0'

    loaded_page_vector_db = load_vectorstore(persist_directory= chroma_page_directory, collection_name= chroma_page_collection
                                             , embedding_model=embedding_model)

    
    
    query = input("What is your query from the RAG System?")
    get_chunk = relevant_chunks(loaded_page_vector_db, query)

    print(get_chunk[0].metadata)

    #    

    # page_vectorstore = create_vectorstore(page_chunks, embedding_model=embedding_model, persist_directory=chroma_page_directory, 
    #                                       collection_name=chroma_page_collection)
    
    # print("======Page Embedding Complete========")
    
    # return page_vectorstore

    relevant_page_source = get_chunk[0].metadata['source']
    relevant_page_num = get_chunk[0].metadata['page']

    relevant_page_upload = pymupdf.open(relevant_page_source)
    relevant_page = relevant_page_upload.load_page(relevant_page_num)

    highlight_instances = relevant_page.search_for(get_chunk[0].page_content, quads = False) #returns a list of rectangle objects that contain the text in the get chunk page content
    relevant_page.add_highlight_annot(highlight_instances)
    
    pix = relevant_page.get_pixmap(annots = True)
    img_data = pix.tobytes('png')
    img = Image.open(io.BytesIO(img_data))
    img.show()





# for doc in loaded_docs:
#     chunks = split_text(doc.page_content)
#     print("===Splitting Doc===")

#human_message = (input("What would you like to ask me? "))
#generate_response(human_message)




#docs[0].metadata
#print(len(docs[0].page_content))

#print(len(docs))

#Splitting the documents into chunks

# text_splitter = RecursiveCharacterTextSplitter(
#     chunk_size = 2000,
#     chunk_overlap = 500,
#     separators=[ "def", "class"]
# )

# chunk_txt_base = []
# counter = 0

#for doc in docs:
 #   splits = text_splitter.split_text(doc.page_content)
  #  chunk_txt_base.append(splits)
   # counter += 1

# print(counter)
# print(len(chunk_txt_base))

# all_chunks = [chunk for document in chunk_txt_base for chunk in document]

#Embedding the document

# vector_store = Chroma.from_texts(all_chunks, embedding_model)

# print("Embedding done")

# template = '''
# Speak like a normal person
# Give answers only based on the codebase (chunks) that has
# been provided as context 

# Here is the conversation history: {context}

# Question: {question}

# Answer:

# '''

# prompt = ChatPromptTemplate.from_template(template)




# model = OllamaLLM(model="llama3.2:1b")

# chain = prompt | model

# def chat ():
#     context = ""
#     print("My name is Batolebaz, small LLM app built by Varad who's learning to build RAG applications, I have been modified to speak like"
# "Shakespeare himself. Let's try this out, Type exit to quit\n")

#     while True:



#         query = str(input("What would you like to ask me? "))

#         retrieval = vector_store.similarity_search(query)
#         print(f"Chunk most similar to the query: {retrieval}" )

#         context += f"Chunks retrieved: {retrieval}"

#         if query.lower() == 'exit':
#             print("It was nice chatting with you, Have a great Day")
#             break
#         else:
#             result = chain.invoke({"context": "", "question": query})
#             print("Shakespeare says: ", result)
#             context += f"\nUser:{query}\n AI: {result}  "


# #Adding and embedding documents

if __name__ == "__main__":
    chunk_in_doc()     


