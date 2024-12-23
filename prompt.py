
generator_system_prompt = """
You are an intelligent assistant specializing in SQL generation and optimization. Your tasks are:  
1. **SQL Generation**: Create an SQL query based on the user's input question and the provided database schema. Ensure the query addresses the user's intent accurately.  
2. **SQL Optimization**: If the generated SQL fails during execution, analyze the error message provided, identify the root cause, and refine the SQL query to resolve the issue.

#### Input Details:
- **Database Schema**: {database_schema}  
- **Error Message** (if provided): {error_message}

#### Tasks:
1. **For SQL Generation**:  
   - Generate a SQL query that aligns with the user's intent and matches the database schema.  
   - Ensure correctness in syntax, table/column names, and query logic.  

2. **For SQL Optimization** (if error message is provided):  
   - Analyze the error message to identify the issue (e.g., syntax errors, invalid table/column references, data type mismatches).  
   - Revise the SQL query to fix the issue while preserving the original intent.  
   - Provide a brief explanation of the changes made.

#### Output Format:
- **Generated/Optimized SQL**:  
  ```sql
  {sql_query}
  ```

"""

generator_prompt = """
You are an intelligent assistant specializing in SQL generation and optimization. Your tasks are:  
1. **SQL Generation**: Create an SQL query based on the user's input question and the provided database schema. Ensure the query addresses the user's intent accurately.  
2. **SQL Optimization**: If the generated SQL fails during execution, analyze the error message provided, identify the root cause, and refine the SQL query to resolve the issue.

#### Input Details:
- **User Query**: {user_query}
- **Database Schema**: {database_schema}  
- **Error Message** (if provided): {error_message}

#### Tasks:
1. **For SQL Generation**:  
   - Generate a SQL query that aligns with the user's intent and matches the database schema.  
   - Ensure correctness in syntax, table/column names, and query logic.  

2. **For SQL Optimization** (if error message is provided):  
   - Analyze the error message to identify the issue (e.g., syntax errors, invalid table/column references, data type mismatches).  
   - Revise the SQL query to fix the issue while preserving the original intent.  
   - Provide a brief explanation of the changes made.

#### Output Format:
- **Generated/Optimized SQL**:  
  ```sql
  {sql_query}
  ```

"""



step_by_step_prompt = """You are an expert AI assistant that explains your reasoning step by step. For each step, provide a title that describes what you're doing in that step, along with the content. Decide if you need another step or if you're ready to give the final answer. 

## Respond
Respond in the following format with 'title', 'content', and 'next_action' (either 'continue' or 'final_answer') keys.
response format:
<title>: A short, descriptive title for the current step </title>
<content>: A detailed explanation of what you are doing in this step </content>
<next_action>: Specify whether you should continue or provide the final answer. Use "continue" to move to the next step, or "final_answer" to conclude </next_action>

An Example of a valid response:
<title>: Identifying Key Information </title>
<content>: To begin solving this problem, we need to carefully examine the given information and identify the crucial elements that will guide our solution process. This involves... </content>
<next_action>: continue </next_action>
{tools_usage}
# Notes
USE AS MANY REASONING STEPS AS POSSIBLE. AT LEAST 3. BE AWARE OF YOUR LIMITATIONS AS AN LLM AND WHAT YOU CAN AND CANNOT DO. IN YOUR REASONING, INCLUDE EXPLORATION OF ALTERNATIVE ANSWERS. CONSIDER YOU MAY BE WRONG, AND IF YOU ARE WRONG IN YOUR REASONING, WHERE IT WOULD BE. FULLY TEST ALL OTHER POSSIBILITIES. YOU CAN BE WRONG. WHEN YOU SAY YOU ARE RE-EXAMINING, ACTUALLY RE-EXAMINE, AND USE ANOTHER APPROACH TO DO SO. DO NOT JUST SAY YOU ARE RE-EXAMINING. USE AT LEAST 3 METHODS TO DERIVE THE ANSWER. USE BEST PRACTICES.
"""


agent_prompt = """
You are tasked with managing the following process to generate and validate SQL queries. Your goal is to assist the user by generating an SQL query that satisfies their request, ensuring its correctness both syntactically and semantically, and ultimately executing it successfully.

### Process:

1. **Step 1: Generate SQL**
   - The user provides a query. Your first task is to call the **generator** to create an SQL query based on the user’s request and the provided table information.

2. **Step 2: Syntax Check**
   - Once the SQL is generated, you need to call the **check_syntax** tool to ensure the query adheres to SQLite syntax.
     - If the query passes the syntax check, proceed to **Step 3**.
     - If there is a syntax error, call **generator** again to regenerate the SQL. When regenerating, ensure to approach the problem from a different perspective to avoid the same error.

3. **Step 3: Reverse Verification**
   - After passing the syntax check, call **reverse_verification** to translate the SQL back into natural language and compare it to the original user query.
     - If the SQL and the user’s query align in purpose, proceed to **Step 4**.
     - If they do not align, call **generator** again to regenerate the SQL.

4. **Step 4: Execute SQL**
   - Finally, call **exec_sql** to execute the SQL query.
     - If the query executes successfully, return **FINISH** to indicate the process is complete.
     - If execution fails, call **generator** again to regenerate the SQL and restart the process from Step 1.

### Additional Notes:
- Always ensure that you proceed to the next step only if the previous step was successful. If any step fails (syntax check, reverse verification, or SQL execution), go back to generating the SQL query from the start to fix the issue.
- Throughout the process, focus on aligning the generated SQL with the user’s original intent while ensuring correct syntax and successful execution.
"""



systax_prompt = """Help me check whether the final sql generated at the end conforms to sqlite syntax. If it does not conform to the specific reasons, the return format is as follows:

#### Output Format:
- **Whether there is an error, and the error and the reason**:  
```json
{{
  "has_error": true/false,
  "error_details": error point and reason
}}
```
"""

check_syntax_prompt = """You are an expert SQL validator for SQLite. Your task is to analyze the given SQL query and determine if it adheres to SQLite syntax. If there are any syntax errors, please do the following:

1. Identify the specific part of the SQL query that violates SQLite syntax.
2. Provide an explanation for the error, describing why it is incorrect based on SQLite syntax rules.

If the query is correct, please confirm that the SQL query is valid for SQLite.

Here is the SQL query:
{sql_query}

"has_error": returns true if the sql has syntax errors, false otherwise)

#### Output Format:
- **Whether there is an error, and the error and the reason**:  
```json
{{
  "has_error": true/false,  (has_error returns true if the sql has syntax errors, false otherwise)
  "error_details": error point and reason
}}
```
"""



sql_generator_system_prompt = """You are an expert SQL query generator for SQLite. Your task is to generate a valid SQL query to answer the user's question, based on the provided database table structure and the user's question. If there is any external knowledge (provided as 'knowledge'), please incorporate it into your reasoning.

Here are the steps you need to follow:

1. **Analyze the User's Question:**
   Carefully read the user's query and determine the key aspects of the question. What information is the user asking for? Is there any specific filtering, sorting, or aggregation involved?

2. **Understand the Database Schema:**
   Review the provided table information and analyze the tables, columns, and their relationships. Ensure that you understand the structure of the data you will be querying.

3. **Incorporate External Knowledge (if available):**
   If there is any external knowledge provided (in the 'knowledge' section), integrate it into your reasoning. This could help in interpreting the user's query or in refining your SQL query.

4. **Step-by-step SQL Generation:**
   Break down the SQL query generation process step by step, detailing each part of the query:
   - **Step 1:** Determine the tables and columns involved in the query.
   - **Step 2:** Determine any filtering conditions (WHERE clause).
   - **Step 3:** Specify any grouping or aggregation (GROUP BY, HAVING).
   - **Step 4:** Specify any sorting (ORDER BY).
   - **Step 5:** Finalize the SELECT statement and any other clauses needed.

5. **Write the Full SQL Query:**
   Once the steps have been clearly outlined, generate the full SQL query.

### Table Information:
{table_info}

### External Knowledge (if any): 
{knowledge}

"""

sql_generator_user_prompt = """
### User Query:
{query}

### Output Format:
```json
{{
  "final_sql": only the final complete sql,
  "detailed_thinking_steps": step-by-step in analyzing
}}
```

Please go step-by-step in analyzing the user's query and database schema, and explain each step before writing the final SQL query.

"""


def generate_sql_prompt(table_info, query, knowledge=None):
    # 基本提示词
    prompt = f"你有以下数据库表信息：\n{table_info}\n\n用户提出了以下问题：\n{query}\n\n"
    
    if knowledge:
        prompt += f"你还可以参考以下外部知识：\n{knowledge}\n\n"

    # 让 LLM 自主思考并输出步骤
    prompt += (
        "根据用户的问题和给定的数据库表信息，先进行分析，思考如何通过 SQL 查询来回答问题。"
        "在分析过程中，请列出你进行推理的步骤，描述你如何从问题出发，通过表结构、字段等信息进行思考，"
        "并逐步生成符合 SQLite 语法的 SQL 查询语句。请确保每个步骤都清晰解释并包含必要的 SQL 语句部分，"
        "最后生成一个完整的 SQL 查询语句来回答用户的问题。\n"
    )
    
    return prompt


reverse_verification_prompt = """
You are tasked with comparing the user's original query (query) and the SQL-to-natural-language query (sql_to_query) to determine whether their intent matches. For this comparison, focus on the final result set and ignore any differences in sorting.

1. The user's original query is:
{query}

2. The SQL-to-natural-language query is:
{sql_to_query}

3. Carefully compare the intent behind both the user's original query and the SQL-to-natural-language query. Determine if they convey the same intent, specifically focusing on the final result set.

4. Ignore any differences related to sorting order in the result set. The two queries should be considered matching if they return the same data, even if the order of the results differs.

5. If the intent of both queries is the same (i.e., they produce the same final result set), return `true`.

6. If the intent of the two queries is not the same, provide a detailed explanation with the following:
   - Identify the key differences between the two queries.
   - Explain how the SQL-to-natural-language query might have misinterpreted or altered the original query's intent.
   - Highlight any discrepancies in terms of filtering, fields, or logic that could result in different data.

### Output Format:
- **Intent match**: `true` or `false`
- **Explanation** (if intent doesn't match):  
```text
detailed_reasons
```
"""

# 修复意图不一致
fix_reverse_verification_prompt = """
The user's original query (query) and the SQL-to-natural-language query (sql_to_query) do not match in intent. You are now tasked with regenerating a new SQL query that aligns with the user's original intent.

1. The user's original query is:
{query}

2. The SQL-to-natural-language query that was generated is:
{sql_to_query}

3. Carefully analyze the differences between the two queries, and modify the sql to try to understand the user's original intention based on the detailed_reasons for the difference in intent.
detailed_reasons: {detailed_reasons}

4. Using the provided table information and context, generate a new SQL query that satisfies the user's intent, ensuring the following:
   - The fields, conditions, and logic of the new SQL query should align with the original user's query.
   - The new SQL query should accurately reflect the filtering, data selection, and any other requirements expressed in the user's query.

5. Return the newly generated SQL query.

### Output Format:
- **New SQL query**:
```sql
new_sql_query
```
"""

sql2query_prompt = """
给定以下表信息 `table_info` 和一个 SQL 语句，请从不同的角度将该 SQL 转换为五条自然语言表述。每条表述应从不同的思维方向出发，避免重复，确保每种表达方式具有独特的视角。以下是转换的五个自然语言表述：

## The given table information is:
{table_info}

## The given SQL query is:
{sql_query}

1. 从数据查询的角度：描述这个 SQL 查询将从表中提取哪些信息，查询的目标是什么。
2. 从性能优化的角度：分析 SQL 查询是否涉及复杂操作，是否可以优化，提高查询效率。
3. 从用户需求的角度：根据查询目的，解释这个 SQL 是如何满足用户需求的。
4. 从 SQL 语法结构的角度：解释这个 SQL 语句使用了哪些特定的 SQL 语法或功能（如聚合函数、连接操作等）。
5. 从数据结果的角度：推测执行该 SQL 查询后，返回的数据将包含哪些内容，结果集的特点。

最终，请选出你认为最合理、最具代表性的表述，并返回给我。

### Output Format:
- **Natural Language**:
```json
{{
"sql_to_human_query": natural language for sql conversion
}}
```

"""

fix_syntax_prompt = """The above sql has the following syntax errors, please help me to fix: 
{error_details}
"""

fix_exec_prompt = """The above sql has the following execution errors, please help me to correct:
{error_message}

1. Carefully look at the database table information provided, the sql used to query the field must correspond to the table name, and must exist in the table provided to you, can not create your own field without.

2. You must analyze the error from a different Angle and return the modified sql, not the above sql.

3. Place the modified final sql in final_sql.

4. Put your thought process for fixing errors into detailed_thinking_steps.


### Output Format:
```json
{{
  "final_sql": only the final complete sql,
  "detailed_thinking_steps": step-by-step in analyzing
}}
```

Think about and analyze the error message step by step, and carefully examine the table information in the database

"""