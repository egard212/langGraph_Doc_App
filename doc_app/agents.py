from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

""" -------------------- Agent Router -> Node who decides which Agent a User's Query should go to -------------------- """
def get_question_router():

    router_llm = ChatOllama(model="llama3:8b-instruct-q8_0", format="json", temperature=0)

    router_prompt = PromptTemplate(
        template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are an expert at routing a 
        user question to one of three agents, a json analyzer, a document format assesser, and a document content specialist. 
        Use the json analyzer when the question contains a json file, use the document format assesser when the question  
        contains querys about the document formatting (like font, title, etc.), and use the document content specialist for questions 
        regarding the actual content of the paper. You do not need to be stringent with the keywords in the question related to these topics. 
        Give one of three choices 'json_analyzer', 'document_format_analyzer', or 'document_content_specialist' based on the question. 
        Return a JSON with a single key 'node' and no premable or explanation. 
        Question to route: {question} <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
        input_variables=["question"],
    )

    question_router = router_prompt | router_llm | JsonOutputParser()
    return question_router

""" -------------------- Content Specialist -> RAG Agent for querys about the document's content -------------------- """
def get_rag_chain():

    rag_llm = ChatOllama(model="llama3:8b-instruct-q8_0")

    # Prompt
    prompt = PromptTemplate(
        template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are an assistant for question-answering tasks. 
        Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. 
        Use three sentences maximum and keep the answer concise <|eot_id|><|start_header_id|>user<|end_header_id|>
        Question: {question} 
        Context: {context} 
        Answer: <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
        input_variables=["question", "document"],
    )

    # Chain
    rag_chain = prompt | rag_llm | StrOutputParser()
    return rag_chain

""" -------------------- Document Analysis Router -> Routes a question to either the code executor or Image analyzer  -------------------- """
def get_analysis_router():
    router_llm = ChatOllama(model="llama3:8b-instruct-q8_0", format="json", temperature=0)

    doc_router_prompt = PromptTemplate(
        template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are an expert at routing a 
        user question to one of two agents, a code writer/executor and an image analysis agent. 
        Use the code executor when the question contains informtion related to formatting (like font or document formatting) or
        information that would be easy to verify by searching through the document (like if it contains a specific paragraph). Use the image
        analyzer when the question contains information related to images (like does it contain an image, signature, or plot).
        You do not need to be stringent with the keywords in the question related to these topics. 
        Give one of two choices 'code executor' or 'image analysis' based on the question. 
        Return a JSON with a single key 'node' and no premable or explanation. 
        Question to route: {question} <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
        input_variables=["question"],
    )

    doc_question_router = doc_router_prompt | router_llm | JsonOutputParser()
    return doc_question_router

""" -------------------- Document Format Analyzer-> Answers questions about word document formatting -------------------- """
