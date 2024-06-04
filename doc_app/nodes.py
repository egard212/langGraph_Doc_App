from .tools import DocumentVectorStore
from .agents import *
from .state import GraphState

class Nodes():
    """
    Each Node the state machine will traverse through
    """
    def __init__(self, doc_path: str):
        self.rag = DocumentVectorStore(doc_path)

    def user_question_router(self, state: GraphState) -> str:
        """
        Routes question to either RAG, json, or doc analysis router.

        Args:
            state (dict): The current graph state

        Returns:
            str: Next node to call
        """
        # Retrieve User's Question
        print("--- INITIAL ROUTE NODE ---")
        question = state['question']

        # Let Router decide next node 
        question_router = get_question_router()
        next_node = question_router.invoke({"question": question})

        print(f"--- ROUTING QUESTION TO {next_node['node'].upper()} NODE ---")
        return next_node['node']
        
    def rag_content_retrieval(self, state: GraphState):
        """
        Retrieves requested context from the Vector Store and generates answer

        Args:
            state (dict): The current graph state
        
        Returns:
            state (dict): Updates current state's answer
        """
        # Get our Rag Chain and VectorStore retriever
        print("--- RAG NODE ---")
        rag_chain = get_rag_chain()
        retriever = self.rag.retriever

        # Retrieve relevant docs from the vectorstore
        docs = retriever.invoke(state.question)

        # Generate the answer from the docs
        generation = rag_chain.invoke({"context": docs, "question": state.question})

        return {"answer": generation}

    def json_steps_itr(self, state: GraphState):
        """
        Iterates through all steps in the provided json file, sending each step to the document
        analysis router

        Args:
            state (dict): The current graph state
        
        Returns:
            state (dict): Updates current state's question

        """
        pass
    
    def doc_analysis_router(self, state: GraphState) -> str:
        """
        Routes question to either the image analysis agent or the code executor.

        Args:
            state (dict): The current graph state

        Returns:
            str: Next node to call
        """
        pass
    
    def code_plan_and_exec(self, state: GraphState):
        """
        Given a query, multiple agents (a planner, code writer, and code executor)
        attempt to answer it via a group chat discussion.

        Args:
            state (dict): The current graph state
        
        Returns:
            state (dict): Updates current state's answer
        """
        pass

    def convert_doc_to_image(self,state: GraphState):
        """
        Converts the document (either a word or pdf document) to an image
        that our LLM (Llava) can analyze the content of
        """
        pass

    def image_analysis(self, state: GraphState):
        """
        Given a query, multiple agents (a planner, code writer, and code executor)
        attempt to answer it via a group chat discussion.

        Args:
            state (dict): The current graph state
        
        Returns:
            state (dict): Updates current state's answer
        """
        pass

    def give_answer(self, state: GraphState):
        """
        Provides the user with the answer to their query
        (Last Node in our Graph)

        Args:
            state (dict): The current graph state
        """
        pass
