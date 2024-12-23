import json
import argparse
from evaluation import evaluation_main
# from gpt_request import generate_main
from main import graph_main

def main(args):
    base_path = "/home/jane/LLM/evaluation/table_related_benchmarks/evalset/"
    if args.eval_data_name == "bird" and args.mode == "dev":
        args.db_root_path = base_path + "bird_data/dev_database"
        args.eval_data_path = base_path +"bird_data/dev.json"
        args.ground_truth_path = base_path +"bird_data/dev.sql"
        # args.mode = "dev"
        # args.use_knowledge = "True"
        
    if args.eval_data_name == "spider" and args.mode == "test":
        args.db_root_path = base_path +"spider_data/test_database"
        args.eval_data_path = base_path +"spider_data/test.json"
        args.ground_truth_path = base_path + "spider_data/test_gold.sql"
        # args.mode = "test"
        # args.use_knowledge = "False"

    if args.eval_data_name == "spider" and args.mode == "dev":
        args.db_root_path = base_path + "spider_data/dev_database"
        args.eval_data_path = base_path + "spider_data/dev.json"
        args.ground_truth_path = base_path + "spider_data/dev_gold.sql"
        # args.mode = "dev"
        # # args.use_knowledge = "False"
        # #  

    if args.is_use_knowledge:
        args.use_knowledge = "True"
    else:
        args.use_knowledge = "False"
    eval_datas = json.load(open(args.eval_data_path, 'r'))
    # eval_datas = eval_datas[:10]

    # from evaluation import get_ids_datas
    # eval_datas = get_ids_datas(eval_datas)

    predicted_sql_path = graph_main(eval_datas, args)
    # predicted_sql_path = "/home/jane/LLM/langgraph/output/tablegpt_7b/tt/spider_dev_False.json"
    evaluation_main(args, eval_datas, predicted_sql_path)
 



if __name__ == '__main__':

    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--eval_type', type=str, choices=["ex"], default="ex")
    args_parser.add_argument('--eval_data_name', type=str, choices=["bird", "spider"], default="bird")
    args_parser.add_argument('--mode', type=str, choices=["dev", "test"], default="dev")
    args_parser.add_argument('--is_use_knowledge', default=False, action="store_true")
    args_parser.add_argument('--data_output_path', type=str, default="output/tablegpt_7b/workflow_3")
    # args_parser.add_argument('--data_output_path', type=str, default="output/qwen_7b/workflow_2")
    args_parser.add_argument('--chain_of_thought', type=str, default="False")
    args_parser.add_argument('--model_path', type=str) # , required=True
    args_parser.add_argument('--gpus_num', type=int, default=1)
    args_parser.add_argument('--num_cpus', type=int, default=16)
    args_parser.add_argument('--meta_time_out', type=float, default=30.0)
    args_parser.add_argument('--use_encoder', default=False, action="store_true")
    args_parser.add_argument('--use_gpt_api', default=False, action="store_true")
    
    args = args_parser.parse_args()
    main(args)


    # CUDA_VISIBLE_DEVICES=6 python table_related_benchmarks/run_text2sql_eval.py --model_path /data4/sft_output/qwen2.5-7b-ins-0929/checkpoint-3200 --eval_data_name spider