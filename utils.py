from langchain_openai import ChatOpenAI
from typing_extensions import Literal

from typing_extensions import TypedDict


def generate_comment_prompt(knowledge=None):
    pattern_prompt_no_kg = "-- Using valid SQLite, answer the following questions for the tables provided above."
    pattern_prompt_kg = "-- Using valid SQLite and understading External Knowledge, answer the following questions for the tables provided above."

    knowledge_prompt = "-- External Knowledge: {}".format(knowledge)

    if not knowledge:
        result_prompt = pattern_prompt_no_kg
    else:
        result_prompt = knowledge_prompt + '\n' + pattern_prompt_kg

    return result_prompt


def get_llm():
    # judge_llm = ChatOpenAI(model="/home/jane/weights/Qwen/Qwen2.5-7B-Instruct", 
    # base_url="http://localhost:8087/v1", api_key="token-abc123",
    # temperature=0.01
    # )

    llm = ChatOpenAI(model="/home/jane/weights/LLM-Research/TableGPT2-7B", 
    base_url="http://localhost:8088/v1", api_key="token-abc123",
    temperature=0.01
    )
    # llm = ChatOpenAI(model="tablegpt2-7b", 
    # base_url="http://localhost:18081/v1", api_key="token-abc123",
    # temperature=0.01
    # )

    # llm = ChatOpenAI(model="qwen2.5-7b-instruct", 
    # base_url="http://localhost:18080/v1", api_key="token-abc123",
    # temperature=0.01
    # )
    return llm

def show_image(graph):
    from IPython.display import display, Image
    img  = Image(graph.get_graph().draw_mermaid_png())
    # display(img)
    with open("graph.jpg", 'wb') as f:
        f.write(img.data)

class Router(TypedDict):
    """Worker to route to next. If no workers needed, route to FINISH."""
    # ["generator", "exec_sql", "FINISH"]
    next: Literal["generator", "reverse_verification", "check_syntax", "exec_sql"]

class GeneratorSQL(TypedDict):
    """Check whether the sql is correct. If not, return the details and determine the reason"""
    # ["generator", "exec_sql", "FINISH"]
    final_sql: str
    detailed_thinking_steps: str


class checkSyntax(TypedDict):
    """Check whether the sql is correct. If not, return the details and determine the reason"""
    # ["generator", "exec_sql", "FINISH"]
    has_error: Literal["true", "false"]
    error_details: str

class ReverseVerification(TypedDict):
    """Check whether the sql is correct. If not, return the details and determine the reason"""
    # ["generator", "exec_sql", "FINISH"]
    intent_match: Literal["true", "false"]
    detailed_reasons: str


class Sql2Query(TypedDict):
    """Check whether the sql is correct. If not, return the details and determine the reason"""
    # ["generator", "exec_sql", "FINISH"]
    sql_to_human_query: str


class ExecSQL(TypedDict):
    """Check whether the sql is correct. If not, return the details and determine the reason"""
    # ["generator", "exec_sql", "FINISH"]
    sql_to_human_query: str
