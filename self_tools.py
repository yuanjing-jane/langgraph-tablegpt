import sys
import json
from langchain_core.runnables.config import RunnableConfig
from langchain_core.tools import tool
from func_timeout import func_timeout, FunctionTimedOut

@tool
def exec_sql(sql: str, config: RunnableConfig):
    """
    Use this to execute sql code
    """
    db_path = config.get("configurable", {}).get("db_path")

    def execute_sql(db_path, sql):
        import sqlite3
        conn = sqlite3.connect(db_path)
        # Connect to the database
        cursor = conn.cursor()
        cursor.execute(sql)
        res = cursor.fetchall()
        return res
    
    meta_time_out = 30.0
    try:
        res = func_timeout(meta_time_out, execute_sql,
                                  args=(db_path, sql))
    except KeyboardInterrupt:
        import sys
        sys.exit(0)
    except FunctionTimedOut as e:
        return False, f"Failed to execute. Error: {str(e)}", []
    except Exception as e:
        return False, f"Failed to execute. Error: {str(e)}", []

    # if len(res) == 0:
    #     return True, f"The sql is executed successfully, but the result is: {res}", res
    # else:
    #     return True, f"The sql is executed successfully", res
    return True, f"The sql is executed successfully", res

@tool
def parser_sql(text):
    """
    """
    import re
    text = text.strip()
    sql_query_1 = re.findall(r'```sql(.*?)```', text, re.DOTALL)
    sql_query_2 = re.findall(r'```(.*?)```', text, re.DOTALL)
    sql_query_3 = re.findall(r'"""(.*?)"""', text, re.DOTALL)
    
    if sql_query_1:
        extracted_sql = sql_query_1[-1].strip()
    elif sql_query_2:
        extracted_sql = sql_query_2[-1].strip()
    elif sql_query_3:
        extracted_sql = sql_query_3[-1].strip()
    else:
        top_word = text.split(" ")[0]
        if not top_word.lower().startswith("select"):
            extracted_sql = "SELECT " + text
        else:
            extracted_sql = text
    extracted_sql_ls = extracted_sql.split("\n")
    extracted_sql_ls = [s for s in extracted_sql_ls if not s.lower().startswith("-- ") ]
    extracted_sql = "\n".join(extracted_sql_ls)
    return extracted_sql


@tool
def parser_json(text):
    """
    """
    import re
    text = text.strip()
    json_1 = re.findall(r'```json(.*?)```', text, re.DOTALL)
    json_2 = re.findall(r'```(.*?)```', text, re.DOTALL)
    if json_1:
        json_res = json_1[-1].strip()
    elif json_2:
        json_res = json_2[-1].strip()
    else:
        return text
    # trans_to_dict
    json_res = json.loads(json_res)
    return json_res
