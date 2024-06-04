from langgraph.graph import StateGraph

from .state import GraphState
from .nodes import Nodes


class WorkFlow:
    """
    The State Machine for the workflow. 
    """
    def __init__(self, nodes: Nodes):
        self.nodes = nodes       
        workflow = StateGraph(GraphState)

        # Add all nodes to the workflow
        workflow.add_node("user_question_router", self.nodes.user_question_router)
        workflow.add_node("rag_content_retrieval", self.nodes.rag_content_retrieval)
        workflow.add_node("json_steps_itr", self.nodes.json_steps_itr)
        workflow.add_node("doc_analysis_router", self.nodes.doc_analysis_router)
        workflow.add_node("code_plan_and_exec", self.nodes.code_plan_and_exec)
        workflow.add_node("convert_doc_to_image", self.nodes.convert_doc_to_image)
        workflow.add_node("image_analysis", self.nodes.image_analysis)
        workflow.add_node("give_answer", self.nodes.give_answer)

        # Set the entry point for the workflow
        workflow.set_entry_point("user_question_router")

        # Add conditional edges to the workflow
        workflow.add_conditional_edges(
            "user_question_router",
            self.nodes.user_question_router,
            {
                "json_analyzer" : "json_steps_itr",
                "document_format_analyzer" : "doc_analysis_router",
                "document_content_specialist" : "rag_content_retrieval"
            }
        )
        workflow.add_conditional_edges(
            "doc_analysis_router",
            self.nodes.doc_analysis_router,
            {
                "code executor" : "code_plan_and_exec",
                "image analysis" : "convert_doc_to_image"
            }
        )

        # Add remaining edges
        workflow.add_edge("rag_content_retrieval", "give_answer")
        workflow.add_edge("image_analysis", "give_answer")
        workflow.add_edge("code_plan_and_exec", "give_answer")

        workflow.add_edge("json_steps_itr", "doc_analysis_router")
        workflow.add_edge("convert_doc_to_image", "image_analysis")

        # Finish Workflow
        self.app = workflow.compile()


