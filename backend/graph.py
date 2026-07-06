"""
Graph Module — LangGraph state graph with router, tool nodes, and subtool routing.

ARCHITECTURE:
  router → (conditional) → tool_node
  
  For search_agent:
    search_agent → subtool_router → (conditional) → subtool_node → END
                                                  ↘ END (if subtool=none)
  
  For all other tools:
    tool_node → END

KEY DESIGN:
  Each subtool (summarizer, mcq_generator, etc.) can be reached from TWO paths:
    1. Directly from router (primary tool)
    2. From subtool_router (as a sub-agent of search_agent)
  
  To avoid LangGraph's "duplicate edge" error, we use WRAPPER NODES for subtools.
  - "summarizer" is the primary node (router → summarizer → END)
  - "sub_summarizer" is the subtool wrapper (subtool_router → sub_summarizer → END)
  Both use the same ToolNode([summarizer_func]) internally.
"""
import hashlib
import json
import re

from typing import TypedDict, List

from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, BaseMessage
from langchain_core.runnables import RunnableLambda
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode


class AgentState(TypedDict):
    messages: List[BaseMessage]
    next_tool: str
    subtool: str


def classify_query(query: str, llm) -> str:
    """
    Intelligent hybrid classifier to determine the correct agent.
    Combines direct keyword parsing (high confidence) with few-shot LLM classification.
    """
    ql = query.lower()
    
    # 1. High-confidence direct compound keyword matches
    if any(phrase in ql for phrase in ["multiple choice", "practice test with options"]):
        return "mcq_generator"
    if any(phrase in ql for phrase in ["study plan", "revision plan", "exam strategy", "question paper", "test paper", "mock exam", "probable questions", "mock paper"]):
        return "exam_prep_agent"
    if any(phrase in ql for phrase in ["make notes", "key notes", "revision notes", "study notes"]):
        return "notes_maker"
    if any(phrase in ql for phrase in ["how does it work", "explain concept", "what does this mean"]):
        return "concept_explainer"
    if any(phrase in ql for phrase in ["search the web", "latest news", "external research", "google search"]):
        return "search_agent"

    # 2. Few-shot LLM intent classification (dynamic & conversational)
    classification_prompt = (
        f"You are the intent classifier for a multi-agent RAG system.\n"
        f"Analyze the user's query and output ONLY the exact tool name from the list below.\n\n"
        f"Tools and their descriptions:\n"
        f"1. search_agent: For queries requiring external search, internet lookup, latest news, or current affairs (e.g., 'who won the match yesterday?', 'search the web for the latest updates on YOLO').\n"
        f"2. summarizer: For queries requesting a brief summary, general overview, TL;DR, or condensing information (e.g., 'summarize the document', 'give me a brief overview of YOLO').\n"
        f"3. mcq_generator: For queries asking to generate multiple choice questions, quizzes, or option-based tests (e.g., 'test my knowledge with a quiz', 'give me some MCQs on YOLO', 'create a multiple choice test').\n"
        f"4. notes_maker: For queries asking for revision sheets, cheat sheets, structured study notes, outlines, or key takeaways (e.g., 'extract the key points as notes', 'make study notes on YOLO', 'can I have a cheat sheet of this?').\n"
        f"5. exam_prep_agent: For queries asking for mock exams, practice tests (without options/not MCQs), study/revision plans, or probable exam questions (e.g., 'prepare a question paper on YOLO', 'what questions will likely be on my test?', 'help me prepare for my exam').\n"
        f"6. concept_explainer: For queries asking to explain, define, or clarify a specific term, concept, algorithm, or topic (e.g., 'how does YOLO work?', 'define anchor boxes', 'explain object detection in simple terms').\n"
        f"7. chat_agent: For general conversation, greetings, casual talk, or when no other agent fits (e.g., 'hello', 'how are you?', 'thank you').\n\n"
        f"Here are some examples of queries and their correct classification:\n"
        f"- 'test me on YOLO' -> mcq_generator\n"
        f"- 'help me study for my test next week' -> exam_prep_agent\n"
        f"- 'I want to write a mock exam for YOLO' -> exam_prep_agent\n"
        f"- 'what are the main takeaways of YOLO' -> summarizer\n"
        f"- 'give me a cheat sheet of this' -> notes_maker\n"
        f"- 'what is the current status of YOLOv10' -> search_agent\n"
        f"- 'can you define the concept of IOU' -> concept_explainer\n"
        f"- 'prepare a question paper on YOLO' -> exam_prep_agent\n"
        f"- 'nice job, thanks!' -> chat_agent\n\n"
        f"Query: '{query}'\n\n"
        f"Respond with exactly one of these tool names: search_agent, summarizer, mcq_generator, notes_maker, exam_prep_agent, concept_explainer, chat_agent."
    )
    
    try:
        classification_response = llm.invoke(classification_prompt)
        tool_name = getattr(classification_response, "content", str(classification_response)).strip().lower()
        tool_name = re.sub(r'[^a-z_]', '', tool_name.split('\n')[0].strip())
        
        valid_tools = {"search_agent", "summarizer", "mcq_generator", "notes_maker", "exam_prep_agent", "concept_explainer", "chat_agent"}
        if tool_name in valid_tools:
            return tool_name
    except Exception:
        pass

    # 3. Looser keyword fallback if LLM classification fails/times out
    if any(word in ql for word in ["mcq", "mcqs", "quiz", "quizzes"]):
        return "mcq_generator"
    if any(word in ql for word in ["exam", "prep", "test"]):
        return "exam_prep_agent"
    if any(word in ql for word in ["summarize", "summary", "overview", "condense", "brief"]):
        return "summarizer"
    if any(word in ql for word in ["notes"]):
        return "notes_maker"
    if any(word in ql for word in ["explain", "define", "definition", "clarify"]):
        return "concept_explainer"
    if any(word in ql for word in ["search", "internet", "google", "browse", "lookup"]):
        return "search_agent"
        
    return "chat_agent"


def build_graph(tools, llm):
    """Build and compile the LangGraph agent graph."""

    tool_map = {t.name: t for t in tools}

    # Tools that can also be used as subtools from search_agent
    SUBTOOL_NAMES = ["summarizer", "mcq_generator", "notes_maker", "exam_prep_agent", "concept_explainer"]

    def route_agent(state: AgentState) -> AgentState:
        """LLM-based intent classification router using unified classify_query."""
        if not state["messages"]:
            raise ValueError("No messages found in state")

        query_text = state["messages"][-1].content
        tool_name = classify_query(query_text, llm)

        tool_call = {
            "name": tool_name,
            "args": {"query": query_text},
            "id": f"call_{tool_name}_{hashlib.md5(query_text.lower().encode()).hexdigest()[:8]}",
        }
        ai_message = AIMessage(content="", tool_calls=[tool_call])

        return {
            "messages": state["messages"] + [ai_message],
            "next_tool": tool_name,
            "subtool": "none",
        }

    def route_subtool(state: AgentState) -> AgentState:
        """Route search_agent output to a subtool if needed."""
        if state["next_tool"] != "search_agent":
            return {"messages": state["messages"], "subtool": "none"}

        # Find the search_agent ToolMessage
        tool_msgs = [
            m for m in state["messages"]
            if isinstance(m, ToolMessage) and m.tool_call_id.startswith("call_search_agent")
        ]
        if not tool_msgs:
            return {"messages": state["messages"], "subtool": "none"}

        result_content = tool_msgs[-1].content

        # ToolNode serializes dict returns as JSON strings
        result = None
        if isinstance(result_content, str):
            try:
                parsed = json.loads(result_content)
                if isinstance(parsed, dict):
                    result = parsed
            except (json.JSONDecodeError, TypeError, ValueError):
                pass
        elif isinstance(result_content, dict):
            result = result_content

        if result is None:
            return {"messages": state["messages"], "subtool": "none"}

        subtool = result.get("subtool", "none")
        search_content = result.get("content", "")

        if not subtool or subtool == "none" or not search_content:
            return {"messages": state["messages"], "subtool": "none"}

        if subtool not in SUBTOOL_NAMES:
            return {"messages": state["messages"], "subtool": "none"}

        query = state["messages"][0].content
        tool_call = {
            "name": subtool,
            "args": {"query": query, "context": search_content},
            "id": f"call_sub_{subtool}_{hashlib.md5(query.encode()).hexdigest()[:8]}",
        }
        ai_message = AIMessage(content="", tool_calls=[tool_call])

        return {
            "messages": state["messages"] + [ai_message],
            "subtool": subtool,
        }

    # ==================== Build the graph ====================
    graph = StateGraph(AgentState)

    # Router node
    graph.add_node("router", RunnableLambda(route_agent))

    # Primary tool nodes (used when the router selects them directly)
    for tool_func in tools:
        graph.add_node(tool_func.name, ToolNode([tool_func]))

    # Subtool wrapper nodes (used when subtool_router activates them after search_agent)
    # These are separate nodes to avoid LangGraph's "duplicate edge" error
    for sname in SUBTOOL_NAMES:
        if sname in tool_map:
            graph.add_node(f"sub_{sname}", ToolNode([tool_map[sname]]))

    # Subtool router node
    graph.add_node("subtool_router", RunnableLambda(route_subtool))

    # ---- Edges ----

    # Router → tool nodes (conditional)
    graph.add_conditional_edges(
        "router",
        lambda state: state["next_tool"],
        {t.name: t.name for t in tools},
    )

    # All primary tools except search_agent → END
    for tool_func in tools:
        if tool_func.name != "search_agent":
            graph.add_edge(tool_func.name, END)

    # search_agent → subtool_router
    graph.add_edge("search_agent", "subtool_router")

    # subtool_router → sub_* wrapper nodes or END (conditional)
    subtool_edges = {sname: f"sub_{sname}" for sname in SUBTOOL_NAMES}
    subtool_edges["none"] = END

    graph.add_conditional_edges(
        "subtool_router",
        lambda state: state.get("subtool", "none"),
        subtool_edges,
    )

    # All sub_* wrapper nodes → END
    for sname in SUBTOOL_NAMES:
        graph.add_edge(f"sub_{sname}", END)

    graph.set_entry_point("router")
    return graph.compile()
