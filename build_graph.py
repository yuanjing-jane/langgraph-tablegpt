from langchain_core.runnables.config import RunnableConfig
from langgraph.graph import MessagesState
from langgraph.types import Command
from typing_extensions import Literal

# from typing_extensions import TypedDict
from langchain_core.messages import HumanMessage, FunctionMessage, SystemMessage, AIMessage
# from gpt_request import generate_schema_prompt, cot_wizard
# from prompt import supervisor_system_prompt
from langgraph.graph import START, END, StateGraph
# from langgraph.prebuilt import create_react_agent

from utils import get_llm, Router
from self_tools import exec_sql, parser_sql
from copy import deepcopy

from prompt import (
    systax_prompt, 
    reverse_verification_prompt, 
    sql2query_prompt, 
    fix_syntax_prompt, 
    fix_reverse_verification_prompt,
    fix_exec_prompt, 
    agent_prompt
    )
from utils import GeneratorSQL, checkSyntax, Sql2Query, ReverseVerification

# # Agent
members = ["generator", "check_syntax", "sql2query", "reverse_verification", "exec_sql"] # , "reason_analysis", "evaluate"
# options = members + ["FINISH"]
llm = get_llm()



def supervisor_node(state: MessagesState)-> Command[Literal["generator", "check_syntax", "sql2query", "reverse_verification", "exec_sql"]]:
    
    messages = [SystemMessage(content=agent_prompt)] + state["messages"]
    
    response = llm.with_structured_output(Router).invoke(messages)
    
    goto = response["next"]
    # if goto == "FINISH":
    #     goto = END

    # print("&"*100, "goto:", response)

    return Command(goto=goto)


def generator_node(state: MessagesState, config: RunnableConfig)-> Command[Literal["check_syntax"]]:

    if state["messages"][-1].name=="first_request":
        messages = state["messages"]
    else:
        messages = state["messages"]
    # print("1"*100, "messages")
    # print(messages)
    # # exit()

    result = llm.with_structured_output(GeneratorSQL).invoke(messages)

    # print(result)
    # exit()
    
    # sql = parser_sql.invoke(result.content)
    sql = result["final_sql"]

    # print(sql)
    # exit()
    node_num = messages[-1].additional_kwargs["node_num"]
    node_num += 1
    additional_kwargs = {"generator_sql": sql, "node_num": node_num}

    content = f"The final sql is: {sql}"


    goto = "check_syntax"
    goto = goto_end(goto, node_num, config)
    # print("="*100, "generator_node", node_num)

    return Command(update={
        "messages": [AIMessage(content=content, additional_kwargs=additional_kwargs, name="generator")]
        }, goto=goto
        )


# check_syntax_agent = create_react_agent(llm, tools=[parser_json])
def check_syntax_node(state: MessagesState, config: RunnableConfig)-> Command[Literal["generator", "sql2query"]]:
    # print("2"*100)

    sql = state["messages"][-1].additional_kwargs["generator_sql"]
    node_num = state["messages"][-1].additional_kwargs["node_num"]
    node_num += 1

    messages = state["messages"] + [HumanMessage(content=systax_prompt)]
    
    result = llm.with_structured_output(checkSyntax).invoke(messages)
    
    new_messages = state["messages"][:2] + [state["messages"][-1]]
    if result["has_error"] == "true":
        goto = "generator"
        fix_syntax_prompt_format = fix_syntax_prompt.format_map({"error_details": result["error_details"]})

        new_messages = new_messages + [HumanMessage(content=fix_syntax_prompt_format, name="check_syntax")]
    else:
        goto = "sql2query"

        schema_prompt = config.get("configurable", {}).get("schema_prompt")

        sql2query_prompt_format = sql2query_prompt.format_map(
            {"table_info": schema_prompt, "sql_query": sql}
        )
        new_messages = new_messages + [HumanMessage(content=sql2query_prompt_format, name="check_syntax")]

    additional_kwargs = {"generator_sql": sql, "node_num": node_num}
    new_messages[-1].additional_kwargs = additional_kwargs

    goto = goto_end(goto, node_num, config)
    # print("="*100, "check_syntax_node", node_num)
    
    return Command(update={
        "messages": new_messages
        }, goto=goto
        )


def sql2query_node(state: MessagesState, config: RunnableConfig)-> Command[Literal["reverse_verification"]]:
    # print("3"*100)
    
    messages = [state["messages"][-1]]
    # print(messages)
    # exit()

    # 做sql to query 意图转换的时候， 不能泄露前面已经给的query
    # print("="*100)
    # print(messages)

    # result = judge_llm.with_structured_output(Sql2Query).invoke(messages)
    # print("judge_llm: ",result)
    result = llm.with_structured_output(Sql2Query).invoke(messages)
    # print("llm: ",result)

    # return
    additional_kwargs = state["messages"][-1].additional_kwargs

    node_num = state["messages"][-1].additional_kwargs["node_num"]
    node_num += 1
    additional_kwargs["node_num"] = node_num

    content = result["sql_to_human_query"]

    # 替换掉上一个 要进行sql2human的prompt, 而不是追加
    state["messages"][-1] = HumanMessage(
        content=content, name="sql2query"
    )
    
    state["messages"][-1].additional_kwargs = additional_kwargs

    goto = "reverse_verification"
    goto = goto_end(goto, node_num, config)
    # print("="*100, "sql2query_node", node_num)
    
    return Command(update={
        "messages": state["messages"]
        }, goto=goto
        )

def reverse_verification_node(state: MessagesState, config: RunnableConfig)-> Command[Literal["generator", "exec_sql"]]:

    # print("3"*100)
    question = config.get("configurable", {}).get("question")
    sql_to_human_query = state["messages"][-1].content
    additional_kwargs = state["messages"][-1].additional_kwargs

    node_num = state["messages"][-1].additional_kwargs["node_num"]
    node_num += 1
    additional_kwargs["node_num"] = node_num

    verification_prompt = reverse_verification_prompt.format_map(
        {"query": question, "sql_to_query": sql_to_human_query})
    
    messages = state["messages"]
    messages[-1] = HumanMessage(content=verification_prompt, name="reverse_verification")

    # print(messages)
    # exit()


    result = llm.with_structured_output(ReverseVerification).invoke(messages)

    # print(result)
    # exit()
    if result["intent_match"] == "true":
        goto = "exec_sql"

        state["messages"][-1] = HumanMessage(content=str(result), name="reverse_verification")
    else:
        goto = "generator"
        fix_reverse_verification_prompt_format = fix_reverse_verification_prompt.format_map(
            {"query": question, "sql_to_query": sql_to_human_query, "detailed_reasons": result["detailed_reasons"]}
        )
        state["messages"][-1] = HumanMessage(content=fix_reverse_verification_prompt_format, name="reverse_verification")

    
    state["messages"][-1].additional_kwargs = additional_kwargs

    goto = goto_end(goto, node_num, config)
    # print("="*100, "reverse_verification_node", node_num)
    return Command(update={
        "messages": state["messages"]
        }, goto=goto
        )


def exec_node(state: MessagesState, config: RunnableConfig)-> Command[Literal["generator", "__end__"]]:

    # print("4"*100, "exec_node")
    # print(state)

    # exit()

    additional_kwargs = state["messages"][-1].additional_kwargs

    node_num = state["messages"][-1].additional_kwargs["node_num"]
    node_num += 1
    additional_kwargs["node_num"] = node_num

    sql = additional_kwargs["generator_sql"]

    bool_res, exec_res_str, exec_res = exec_sql.invoke(sql, config)
    

    if bool_res:  #  and len(exec_res) != 0
        # 执行成功， 且 结果不为空
        additional_kwargs["finish_sql"] = sql
        goto = END  # 直接返回end
        content = deepcopy(exec_res_str)

    else:
        # 判断循环了几次， 如果循环次数太多,仍没有解决问题，也还是返回end
        goto = "generator"
        content = fix_exec_prompt.format_map({
            "error_message": exec_res_str
            })
        
    state["messages"][-1] = HumanMessage(
            content=content, name="exec", additional_kwargs=additional_kwargs)

    # print(state)
    goto = goto_end(goto, node_num, config)
    # print("="*100, "exec_node", node_num)
    return Command(
        update = {
            "messages": state["messages"]
            }, goto=goto
    )

def goto_end(goto, node_num, config: RunnableConfig):
    recursion_limit = config["recursion_limit"]
    if node_num >= recursion_limit - 1:
        goto = END
    else:
        goto = goto
    return goto


def build_graph():
    builder = StateGraph(MessagesState)
    builder.add_edge(START, "generator")
    # builder.add_node("supervisor", supervisor_node)
    builder.add_node("generator", generator_node)
    builder.add_node("check_syntax", check_syntax_node)
    builder.add_node("sql2query", sql2query_node)
    builder.add_node("reverse_verification", reverse_verification_node)
    builder.add_node("exec_sql", exec_node)
    

    graph = builder.compile()
    return graph
