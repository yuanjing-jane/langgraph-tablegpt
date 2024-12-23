from langchain_openai import ChatOpenAI
from gpt_request import generate_combined_prompts_one
from langchain_core.tools import tool
from typing import List, Annotated
from langchain_core.runnables.config import RunnableConfig
from langgraph.graph import MessagesState
from langgraph.types import Command
from typing_extensions import Literal

# from typing_extensions import TypedDict
from langchain_core.messages import HumanMessage, FunctionMessage, SystemMessage, AIMessage
from gpt_request import generate_schema_prompt, cot_wizard
from prompt import supervisor_system_prompt
from langgraph.graph import START, END, StateGraph
from langgraph.prebuilt import create_react_agent

from utils import generate_comment_prompt, get_llm, Router
from self_tools import exec_sql, parser_sql
from utils import show_image
from gpt_request import decouple_question_schema, generate_sql_file

import os
import json
from tqdm import tqdm
from joblib import Parallel, delayed
from langgraph.errors import GraphRecursionError
from build_graph import build_graph
from evaluation import evaluation_main
from prompt import sql_generator_system_prompt, sql_generator_user_prompt

"""
# Agent
members = ["generator", "exec_sql"] # , "reason_analysis", "evaluate"
options = members + ["FINISH"]
llm = get_llm()


def supervisor_node(state: MessagesState)-> Command[Literal[*members, "__end__"]]:
    
    # messages = [{"role": "system", "content": supervisor_system_prompt}] + state["messages"]
    messages = [SystemMessage(content=supervisor_system_prompt)] + state["messages"]
 
    response = llm.with_structured_output(Router).invoke(messages)
    
    goto = response["next"]
    if goto == "FINISH":
        goto = END

    # print("&"*100, "goto:", response)

    return Command(goto=goto)

# graph

def generator_node(state: MessagesState, config: RunnableConfig)-> Command[Literal["supervisor"]]:    
    db_path = config.get("configurable", {}).get("db_path")
    schema_prompt = generate_schema_prompt(db_path, num_rows=None)
    comment_prompt = generate_comment_prompt(knowledge=None)

    combined_prompts = schema_prompt + '\n\n' + comment_prompt + cot_wizard()

    messages = [{"role": "system", "content": combined_prompts}] + state["messages"]

    result = llm.invoke(messages)

    return Command(update={
        "messages": [AIMessage(content=result.content, name="generator")]
        }, goto="supervisor"
        )

# exec_agent = create_react_agent(llm, tools=[exec_sql])

def exec_node(state: MessagesState, config: RunnableConfig)-> Command[Literal["supervisor"]]:

    # print("3"*100, "exec_node")

    # print("3"*100, "state")
    # print(state)

    text = state["messages"][-1].content
    # sql = parser_sql(text)
    sql = parser_sql.invoke(text)

    bool_res, exec_res_str = exec_sql.invoke(sql, config)
    # result = exec_agent.invoke(state)

    # print("3"*100, "exec_res_str")
    # print(exec_res_str)

    # exit()

    # result = exec_agent.invoke(state)

    # print(dir(result["messages"][-1]))
    # exit()
    if bool_res:
        additional_kwargs={"finish_sql": sql}
    else:
        additional_kwargs={}
    res = Command(
        update = {
            "messages": [HumanMessage(content=exec_res_str, name="exec", additional_kwargs=additional_kwargs)
        
        ]}, goto="supervisor"
    )
    return res

def build_graph():
    builder = StateGraph(MessagesState)
    builder.add_edge(START, "supervisor")
    builder.add_node("supervisor", supervisor_node)
    builder.add_node("generator", generator_node)
    # builder.add_node("paser_sql", paser_sql_node)
    builder.add_node("exec_sql", exec_node)
    # builder.add_node("evaluate", evaluate_node)
    # builder.add_node("reason_analysis", reason_analysis_node)
    graph = builder.compile()
    return graph
"""


def sort_results(list_of_dicts):
    return sorted(list_of_dicts, key=lambda x: x['sql_idx'])


def get_output_name(args):

    if args.chain_of_thought == 'True':
        output_name = os.path.join(args.data_output_path, f'{args.eval_data_name}_{args.mode}_cot.json')
    else:
        output_name =  os.path.join(args.data_output_path, f'{args.eval_data_name}_{args.mode}_{args.is_use_knowledge}.json')
        unparser_name = os.path.join(args.data_output_path,f'{args.eval_data_name}_{args.mode}_{args.is_use_knowledge}_unparser.json')

    return output_name

def generate_main(eval_data, args):

    question_list, db_path_list, knowledge_list = decouple_question_schema(
        datasets=eval_data, db_root_path=args.db_root_path, args=args)
    assert len(question_list) == len(db_path_list) == len(knowledge_list)

    sql_result = generate_parallel(db_path_list, question_list, knowledge_list, num_cpus=args.num_cpus, meta_time_out=args.meta_time_out)
    sort_sql_result = sort_results(sql_result)

    response_list = []
    for i in tqdm(range(len(question_list)), desc="genertate sql"):
        question = question_list[i]
        db_path = db_path_list[i]
        sql = sql_result[i]["finish_sql"]  #get_single_sql(db_path, question, i)
        
        db_id = db_path.split('/')[-1].split('.sqlite')[0]
        sql = sql + '\t----- bird -----\t' + db_id # to avoid unpredicted \t appearing in codex results
        response_list.append(sql)

    output_name = get_output_name(args) #"output/response_list.json"
    generate_sql_file(sql_lst=response_list, output_path=output_name)
    
    print(f'output: {output_name}')
    # 返回推理数据保存路径
    return output_name

def get_single_sql(db_path, question, knowledge, idx):

    config = {
        "configurable": {
            "db_path": db_path, "question": question, "knowledge": knowledge, 
            "node_num": 0},
        "run_id": idx, "recursion_limit": 24
        }
    inputs, config = get_input_messages(config)

    try:
        res = graph.invoke(inputs, config, stream_mode="values")
        # print("--------: res")
        # print(res)
        additional_kwargs = res["messages"][-1].additional_kwargs
        finish_sql = additional_kwargs["generator_sql"]
    except GraphRecursionError as e:
        finish_sql = str(e)
        print("GraphRecursionError --------: ", str(e))
        # exit()
    except Exception as e:
        finish_sql = str(e)
        print("Exception --------: ", str(e))
        # exit()

    result = {"sql_idx": idx, "finish_sql": finish_sql}
    return result


def generate_parallel(db_path_ls, question_ls, knowledge_list, num_cpus=16, meta_time_out=30.0):
    
    if num_cpus > 1:
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

    results = Parallel(n_jobs=num_cpus, backend='multiprocessing')(
        delayed(get_single_sql)(
            db_path_ls[i], question_ls[i], knowledge_list[i], i) for i in tqdm(range(len(question_ls)), desc="generate"))
    
    return results

def get_input_messages(config):
    sql_generator_system_prompt_format, sql_generator_user_prompt_format, schema_prompt = sql_generator_prompt(config)

    config["configurable"]["schema_prompt"] = schema_prompt
    additional_kwargs = {"node_num": 0}
    inputs = {"messages": [
        SystemMessage(content=sql_generator_system_prompt_format, name="system"),
        HumanMessage(content=sql_generator_user_prompt_format, name="first_request", 
                     additional_kwargs=additional_kwargs
                     )]}
    return inputs, config


def sql_generator_prompt(config: RunnableConfig):
    db_path = config.get("configurable", {}).get("db_path")
    question = config.get("configurable", {}).get("question")
    knowledge = config.get("configurable", {}).get("knowledge")
    schema_prompt = generate_schema_prompt(db_path, num_rows=None)
    sql_generator_system_prompt_format = sql_generator_system_prompt.format_map(
        {"table_info": schema_prompt, "knowledge": knowledge}
        )
    sql_generator_user_prompt_format = sql_generator_user_prompt.format_map(
        {"query": question}
        )
    return sql_generator_system_prompt_format, sql_generator_user_prompt_format, schema_prompt

def stream_run():
    base_dir = "/home/jane/LLM/evaluation/table_related_benchmarks/evalset"

    # data_name = "spider_data"
    # data_type = "dev"
    # db_name = "concert_singer"

    data_name = "bird_data"
    data_type = "dev"
    db_name = "california_schools"

    db_path = f"{base_dir}/{data_name}/{data_type}_database/{db_name}/{db_name}.sqlite"
    question = "What is the highest eligible free rate for K-12 students in the schools in Alameda County?"
    knowledge = "Eligible free rate for K-12 = `Free Meal Count (K-12)` / `Enrollment (K-12)`"

    config = {"configurable": {"db_path": db_path, "question": question, "knowledge": knowledge, "node_num": 0},
               "run_id": 0, "recursion_limit": 24}
    inputs, config = get_input_messages(config)

    global graph
    graph = build_graph()
    # show_image(graph)
    # exit()

    for chunk in graph.stream(
        inputs, config, stream_mode="values"
    ):
        chunk["messages"][-1].pretty_print()
    # res = graph.invoke(inputs, config) #, stream_mode="values"
    # print("--------: res")
    # print(res)

    # result = get_single_sql(db_path, question, 1)
    # print(result)


def graph_main(eval_datas, args):
    global graph
    graph = build_graph()
    # show_image(graph)
    # exit()

    # eval_datas = eval_datas[:10]
    output_name = generate_main(eval_datas, args)

    return output_name

if __name__ == "__main__":
    # graph_main()

    stream_run()




    # from concurrent.futures import ThreadPoolExecutor
    # import os

    # # 创建一个不指定 max_workers 的 ThreadPoolExecutor
    # executor = ThreadPoolExecutor()

    # # 查看默认的最大工作线程数
    # print(f"系统 CPU 核心数: {os.cpu_count()}")
    # print(f"默认 max_workers: {executor._max_workers}")

