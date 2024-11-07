from typing import Annotated, Literal, Sequence
from typing_extensions import TypedDict

from langgraph.graph import END, StateGraph, START
from langgraph.prebuilt import ToolNode

from langchain import hub
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph.message import add_messages
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

from pydantic import BaseModel, Field


from langgraph.prebuilt import tools_condition

from langgraph.checkpoint.memory import MemorySaver

import os
import getpass



def _set_env(var: str):
    if not os.environ.get(var):
        os.environ[var]=getpass.getpass(f"{var}:")

### OpenAI api with qwen
_set_env("OPENAI_API_KEY")
expt_llm = "qwen-plus"
base="https://dashscope.aliyuncs.com/compatible-mode/v1"


###retriever
import retrieval
from langchain.tools.retriever import create_retriever_tool

retriever = retrieval.vectorstore.as_retriever()

if len(retriever.invoke("lilianweng")) == 0:
    print("---Init chroma db---")
    retrieval.init()
    



retriever_tool = create_retriever_tool(
    retriever,
    "retrieve_blog_posts",
    "Search and return information about Lilian Weng blog posts on LLM agents, prompt engineering, and adversarial attacks on LLMs.",
)

tools = [retriever_tool]

###graph

class AgentState(TypedDict):
    # The add_messages function defines how an update should be processed
    # Default is to replace. add_messages says "append"
    messages: Annotated[Sequence[BaseMessage], add_messages]

# class grade(BaseModel):
#     """Binary score for relevance check."""
#     binary_score: str = Field(description="Relevance score 'yes' or 'no'")

# class GradeChain:
#     prompt = PromptTemplate(
#         template="""You are a grader assessing relevance of a retrieved document to a user question. \n 
#         Here is the retrieved document: \n\n {context} \n\n
#         Here is the user question: {question} \n
#         If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n
#         Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question.""",
#         input_variables=["context", "question"],
#     )

#     def __init__(self, expt_llm, base):
#         ## transfer as stream
#         self.llm= ChatOpenAI(temperature=0, model=expt_llm, base_url=base, streaming=True)
#         self.structured_llm_claude = self.llm.with_structured_output(grade)
#          # Chain
#         self.chain = GradeChain.prompt | self.structured_llm_claude


### Edges


def grade_documents(state) -> Literal["generate", "rewrite"]:
    """
    Determines whether the retrieved documents are relevant to the question.

    Args:
        state (messages): The current state

    Returns:
        str: A decision for whether the documents are relevant or not
    """

    print("---CHECK RELEVANCE---")

    # Data model
    class grade(BaseModel):
        """Binary score for relevance check."""

        binary_score: str = Field(description="Relevance score 'yes' or 'no'")

    # LLM
    model = ChatOpenAI(temperature=0, model=expt_llm, base_url=base, streaming=True)

    # LLM with tool and validation
    llm_with_tool = model.with_structured_output(grade)

    # Prompt
    prompt = PromptTemplate(
        template="""You are a grader assessing relevance of a retrieved document to a user question. \n 
        Here is the retrieved document: \n\n {context} \n\n
        Here is the user question: {question} \n
        If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n
        Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question.""",
        input_variables=["context", "question"],
    )

    # Chain
    chain = prompt | llm_with_tool

    messages = state["messages"]
    last_message = messages[-1]

    question = messages[0].content
    docs = last_message.content

    scored_result = chain.invoke({"question": question, "context": docs})

    score = scored_result.binary_score

    if score == "yes":
        print("---DECISION: DOCS RELEVANT---")
        return "generate"

    else:
        print("---DECISION: DOCS NOT RELEVANT---")
        print(score)
        return "rewrite"


### Nodes

# Max tries
max_iterations = 2

def agent(state):
    """
    Invokes the agent model to generate a response based on the current state. Given
    the question, it will decide to retrieve using the retriever tool, or simply end.

    Args:
        state (messages): The current state

    Returns:
        dict: The updated state with the agent response appended to messages
    """

    print("---CALL AGENT---")
    messages = state["messages"]
    model = ChatOpenAI(temperature=0, streaming=True, model=expt_llm, base_url=base)
    model = model.bind_tools(tools)
    response = model.invoke(messages)
    # We return a list, because this will get added to the existing list
    return {"messages": [response]}


def rewrite(state):
    """
    Transform the query to produce a better question.

    Args:
        state (messages): The current state

    Returns:
        dict: The updated state with re-phrased question
    """

    print("---TRANSFORM QUERY---")
    messages = state["messages"]
    question = messages[0].content

    msg = [
        HumanMessage(
            content=f""" \n 
    Look at the input and try to reason about the underlying semantic intent / meaning. \n 
    Here is the initial question:
    \n ------- \n
    {question} 
    \n ------- \n
    Formulate an improved question: """,
        )
    ]

    # Grader
    model = ChatOpenAI(temperature=0, model=expt_llm, base_url=base, streaming=True)
    response = model.invoke(msg)
    return {"messages": [response]}


def generate(state):
    """
    Generate answer

    Args:
        state (messages): The current state

    Returns:
         dict: The updated state with re-phrased question
    """
    print("---GENERATE---")
    messages = state["messages"]
    question = messages[0].content
    last_message = messages[-1]

    docs = last_message.content

    # Prompt
    prompt = hub.pull("rlm/rag-prompt")

    # LLM
    llm = ChatOpenAI(model=expt_llm, base_url=base, temperature=0, streaming=True)

    # Post-processing
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # Chain
    rag_chain = prompt | llm | StrOutputParser()

    # Run
    response = rag_chain.invoke({"context": docs, "question": question})
    return {"messages": [response]}


# Define a new graph
workflow = StateGraph(AgentState)

# Define the nodes we will cycle between
workflow.add_node("agent", agent)  # agent
retrieve = ToolNode([retriever_tool])
workflow.add_node("retrieve", retrieve)  # retrieval
workflow.add_node("rewrite", rewrite)  # Re-writing the question
workflow.add_node(
    "generate", generate
)  # Generating a response after we know the documents are relevant
# Call agent node to decide to retrieve or not
workflow.add_edge(START, "agent")

# Decide whether to retrieve
workflow.add_conditional_edges(
    "agent",
    # Assess agent decision
    tools_condition,
    {
        # Translate the condition outputs to nodes in our graph
        "tools": "retrieve",
        END: END,
    },
)

# Edges taken after the `action` node is called.
workflow.add_conditional_edges(
    "retrieve",
    # Assess agent decision
    grade_documents,
)
workflow.add_edge("generate", END)
workflow.add_edge("rewrite", "agent")

# memory
memory=MemorySaver()
# Compile
graph = workflow.compile(checkpointer=memory)


if __name__ == "__main__":
    # import pprint

    # inputs = {
    #     "messages": [
    #         ("user", "What does Lilian Weng say about the types of agent memory?"),
    #     ]
    # }
    # for output in graph.stream(inputs):
    #     for key, value in output.items():
    #         pprint.pprint(f"Output from node '{key}':")
    #         pprint.pprint("---")
    #         pprint.pprint(value, indent=2, width=80, depth=None)
    #     pprint.pprint("\n---\n")
    import gradio as gr
    config ={"configurable":{"thread_id":"1"}}
    def predict(message, history):
        # history_langchain_format = []
        # for msg in history:
        #     if msg['role'] == "user":
        #         history_langchain_format.append(HumanMessage(content=msg['content']))
        #     elif msg['role'] == "assistant":
        #         history_langchain_format.append(AIMessage(content=msg['content']))
        # history_langchain_format.append(HumanMessage(content=message))
        gpt_response=""
        events = graph.stream({"messages":("user",message)},config,stream_mode="values")
        for event in events:
            event["messages"][-1].pretty_print()

        gpt_response=graph.get_state(config).values["messages"][-1].content
        return gpt_response

    gr.ChatInterface(predict, type="messages").launch()
