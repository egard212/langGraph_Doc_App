from langchain_core.tools import tool
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredWordDocumentLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma

import os

""" -------------------- Vectorstore for our input document -------------------- """

class DocumentVectorStore:
    def __init__(self, doc_path):
        self.doc_path = doc_path
        self.__text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=80,
            length_function=len,
            is_separator_regex=False,
        )
        self.__embeddings = GPT4AllEmbeddings(
            model_name="all-MiniLM-L6-v2.gguf2.f16.gguf",
            gpt4all_kwargs={'allow_download': 'True'}
        )
        self.retriever = self.__create_vectorstore()
    
    def __create_vectorstore(self):
        # Ensure the file exists
        if os.path.isfile(self.doc_path) == False:
            print("Unable to Find File")
            return None
            
        # Check the file extension
        root, ext = os.path.splitext(self.doc_path)
        match ext:
            case ".pdf":
                doc_loader = PyPDFLoader(file_path=self.doc_path)           
            case ".docx":
                doc_loader = UnstructuredWordDocumentLoader(file_path=self.doc_path)
            case _:
                print("Unsupported File Extension")
                return None
        
        # Load in the document
        try:
            docs = doc_loader.load()
        except:
            print("Unable to Open File")
            return None
        
        # Create our vectorstore
        all_splits = self.__text_splitter.split_documents(docs)
        vectorstore = Chroma.from_documents(
            documents=all_splits,
            collection_name="rag-chroma",
            embedding=self.__embeddings
        )

        return vectorstore.as_retriever()




""" Creates Tools for our Document Format Analyzer """




""" Create RAG Pipeline for our Document Content Analyzer """
