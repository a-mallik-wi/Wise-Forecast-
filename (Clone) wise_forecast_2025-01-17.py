# Databricks notebook source
pip install langchain-community databricks-sql-connector databricks_langchain openai databricks-sqlalchemy~=1.0

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

from langchain.agents import create_sql_agent
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.sql_database import SQLDatabase
from langchain import OpenAI
from databricks_langchain import ChatDatabricks

db = SQLDatabase.from_databricks(catalog="forecast", schema="schema1", engine_args={"pool_pre_ping": True})
llm = ChatDatabricks(
    endpoint="databricks-meta-llama-3-1-70b-instruct",
    temperature=0.1,
    max_tokens=250,
)

toolkit = SQLDatabaseToolkit(db=db, llm=llm)
agent = create_sql_agent(llm=llm, toolkit=toolkit, verbose=True)



# COMMAND ----------

agent.run("briefly describe the tables in the database")


# COMMAND ----------

# MAGIC %md
# MAGIC # **Get Schemas of the relevant tables**

# COMMAND ----------

from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
# from langchain.llms import OpenAI
from langchain.sql_database import SQLDatabase
from pydantic import BaseModel, Field
from databricks_langchain import ChatDatabricks

db = SQLDatabase.from_databricks(catalog="forecast", schema="schema1", engine_args={"pool_pre_ping": True})
llm = ChatDatabricks(
    endpoint="databricks-meta-llama-3-1-70b-instruct",
    temperature=0.1,
    max_tokens=250,
)

# Define the Pydantic model for the table
class Table(BaseModel):
    """Table in SQL database."""
    name: str = Field(description="Name of the table in SQL database.")

# Assuming db.get_usable_table_names() gives a list of table names
# For demonstration purposes, let's mock it:
table_names = "\n".join(db.get_usable_table_names())  # Example table names

system_message = f"""Return the names of ALL the SQL tables that MIGHT be relevant to the user question. 
The tables are:

{table_names}

Remember to include ALL POTENTIALLY RELEVANT tables."""

# Create an extraction chain with Pydantic
# The PromptTemplate will be used to handle the question about table relevance
prompt_template = PromptTemplate(
    input_variables=["input"],
    template=system_message + "\nUser question: {input}\nRelevant tables:"
)

# Initialize the chain with the model and prompt
chain = LLMChain(prompt=prompt_template, llm=llm)




# COMMAND ----------

extract_catalog_schema_prompt = PromptTemplate(
    input_variables=["input"],
    template="""
    You are a helpful assistant that extracts catalog and schema names from a database query.
    The user has provided the following query: "{input}"

    Please extract the catalog and schema mentioned in the query.
    If either the catalog or schema is not provided, return 'Not provided' for the missing one.
    Format the response like: Catalog: <catalog_name>, Schema: <schema_name>.
    """
)

def process_user_query(user_query):
    catalog_schema_chain = LLMChain(prompt=extract_catalog_schema_prompt, llm=llm)
    catalog_schema_response = catalog_schema_chain.invoke({"input": user_query})
    # Extract catalog and schema from the response
    catalog, schema = 'Not provided', 'Not provided'
    if 'Catalog:' in catalog_schema_response['text'] and 'Schema:' in catalog_schema_response['text']:
        catalog = catalog_schema_response['text'].split('Catalog:')[1].split(',')[0].strip()
        schema = catalog_schema_response['text'].split('Schema:')[1].strip()
    return catalog.replace('.', ''), schema.replace('.', '')


# COMMAND ----------

user_query = "connect to the forecast catalog and schema1 schema"
catalog,  schema= process_user_query(user_query)
print("Extracted Catalog:", catalog)
print("Extracted Schema:", schema)

# COMMAND ----------

from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.sql_database import SQLDatabase
from pydantic import BaseModel, Field
from databricks_langchain import ChatDatabricks
from sqlalchemy import create_engine, inspect
catalog,  schema= process_user_query(user_query)

def get_available_tables(catalog, schema):
    try:
        tables = spark.sql(f"SHOW TABLES IN {catalog}.{schema}").collect()
        table_names = [row.tableName for row in tables if row.tableName != 'small_sales' and row.tableName != '_sqldf']
        return table_names
    except Exception as e:
        print(f"Error retrieving tables from {catalog}.{schema}: {e}")
        return []


def get_columns_from_db(catalog, schema, tables):
    schema_col = {}
    for table in tables:
        try:
            cols = spark.table(f"{catalog}.{schema}.{table}").columns  
            schema_col[table] = cols
        except Exception as e:
            print(f"Error retrieving columns for {catalog}.{schema}.{table}: {e}")
            schema_col[table] = [] 
    return schema_col

  
table_names = get_available_tables(catalog, schema)

columns_schema = get_columns_from_db(catalog, schema, table_names)




# COMMAND ----------

llm = ChatDatabricks(
    endpoint="databricks-meta-llama-3-1-70b-instruct",
    temperature=0.1,
    max_tokens=800,
)

prompt_template = PromptTemplate(
    input_variables=["input", "tables"],
    template="""
You are a helpful assistant tasked with suggesting the most relevant tables from a database based on the user’s query.

The user has asked: "{input}"

The available tables in the database are: {tables}

The columns for each table are provided in the dictionary: {columns_schema}

Based on the user's query, please:

List the names of the most relevant tables that can help answer the query, separated by commas.
For each relevant table, provide a list of the most important columns. These columns should contain sufficient data to build an accurate forecasting model, considering the user's request. Focus on the columns that would contribute the most to making predictions and training a forecasting model. Ensure the columns selected are relevant to the user's query and can be used for accurate forecasting.
Additionally, feel free to explore other catalog data and suggest other relevant tables from other available sample catalog in databricks that could aid in answering the user's question. 
Ensure that the suggested tables are **only from the available tables or any other tables in the current catalog** listed above, and exclude any other external tables not listed in the current catalog.
"""
)

# Create an LLMChain with the model and the prompt template
chain = LLMChain(prompt=prompt_template, llm=llm)

# COMMAND ----------


user_query = "Suggest data transformations useful for forecasts"

# Run the chain to get the relevant tables for the user's query
relevant_tables = chain.invoke({"input": user_query, "tables": table_names, "columns_schema": columns_schema})

print("Suggested Relevant Tables:")
print(relevant_tables['text'])

# COMMAND ----------

# MAGIC %sql
# MAGIC show catalogs

# COMMAND ----------

import re

input2 = relevant_tables['text']
prompt2 = PromptTemplate(
    input_variables=["input2"],
    template="""
    Based on the suggestion in {input2}
    Please provide some recommendations for useful data transformations that I can apply to this data.The data transformation should also suggest transpose or melt operation wherever necessary. The data transformation should take into account the You can suggest aggregations, filtering, joins, or other transformations to extract meaningful insights from the data. Suggest most relevant transformation for the user's query. list all transformations in proper sequence as needed. Please mention SQL queries for the same. Create unique queries and not alternatives. 
    No need to normalize data. Just give relevant mergings and trandformations.
    """
)
chain2 = LLMChain(prompt=prompt2, llm=llm)

generated_output = chain2.run(input2=input2)

print(generated_output)




# COMMAND ----------

sql_pattern = r"(?s)(SELECT .*?;|INSERT INTO .*?;|UPDATE .*?;|DELETE .*?;|CREATE TABLE .*?;|DROP TABLE .*?;)"


sql_queries = re.findall(sql_pattern, generated_output, re.IGNORECASE | re.DOTALL)

for query in sql_queries:
    print("Query: ", query, "\n")

# COMMAND ----------

import re

expanded_queries = []

for query in sql_queries:
    prompt3 = PromptTemplate(
        input_variables=["query"],
        template="""
        Based on the query below, generate a fully expanded SQL statement that can run **without any modifications**:
        
        **Query:**
        {query}

        **Important Rules:**
        - Do **not** use shortcuts like (d_1, d_2, ..., d_196)..
        - Ensure correct SQL syntax that runs in Databricks.
        - Return **only** the SQL query. Do **not** include explanations.
        """
    )

    chain3 = LLMChain(prompt=prompt3, llm=llm)
    output = chain3.run(query=query) 
    print(output) 

    expanded_query = re.findall(sql_pattern, output, re.IGNORECASE | re.DOTALL)

    expanded_queries.extend(expanded_query)  

# COMMAND ----------

for i, query in enumerate(expanded_queries, 1):
    print(f"Query {i}:\n{query}\n")


# COMMAND ----------

for i, query in enumerate(expanded_queries, 1):
    try:
        result = db.run(query)  
        print(f"Query {i}:\n{query}\nResult: {result}\n")
    except Exception as e:
        print(f"Error executing Query {i}:\n{query}\nError: {e}\n")

# COMMAND ----------

user_query = "suggest data transformation for tables used in query Forecast revenue from California stores for the next month?"

relevant_transformation = chain2.invoke({
    "input": user_query,
    "input2": input2,
    "columns_schema": columns_schema,
    "tables": table_names
})
print(relevant_transformation['text'])

# COMMAND ----------

# MAGIC %md
# MAGIC # **Trial with genie API**

# COMMAND ----------

from databricks_langchain.genie import GenieAgent

# add your genie space id here
genie_space_id =  "genie_test1"
genie_agent = GenieAgent(genie_space_id, "Genie", description="This Genie space has access to forecast databases")

# COMMAND ----------

input_data = {"messages": ["show calendar data"]}
output = genie_agent.invoke(input_data)
# output = genie_agent.ainvoke('select * from calendar')
output


# COMMAND ----------

from databricks.sdk.service.dashboards import GenieAPI


# COMMAND ----------

ge = GenieAPI.start_conversation_and_wait(genie_space_id)

# COMMAND ----------

GenieAPI.start_conversation("genie_test1")

# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------


