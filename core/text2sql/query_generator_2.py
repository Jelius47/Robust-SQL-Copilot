import sys
sys.dont_write_bytecode =True
from io import BytesIO
from datetime import datetime
import os
import logging
import asyncio
import instructor
import nest_asyncio
import pandas as pd
from typing import List
from openai import OpenAI
from pydantic import BaseModel,Field
from difflib import get_close_matches
from openai import OpenAI as instructor_OpenAI
from core.text2sql.sql_connectors import SQLConnector
from core.text2sql.vectorestores import QdrantVectorStore
from core.text2sql.add_context import AddTableContext
from core.text2sql.text_splitter import Schema2Chunks

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

nest_asyncio.apply()

class SQLqueryFormat(BaseModel):
    requirements: str = Field(description="Describe the user requirements in details")
    step_by_step_plan: str = Field(description="Write step by step plan with the relevant tables to generate accurate SQL query to provide all the information listed in the checklist or fix the error in the sql query")
    query_type: str = Field(description="The type of the sql query it could be either intermediate_query, final_query or explanation",examples=["intermediate_query","final_query","explanation"])
    list_of_intermediate_query: List[str] = Field(description="One or more intermediate queries",examples=["SELECT DISTINCT category FROM table_schema.table_name;"])
    final_query: str = Field(description="The accurate query to answer user question or empty in case of irrelevant question",examples=["Select * from table_schema.table_name;",""])
    explanation:str = Field(description="If the provided schema are not sufficient to answer user question")

class SQLColumnValue(BaseModel):
    column : str = Field(description="The column name from the SQL query")
    value : List[str] = Field(description="The values from the user question which should be used in where clause")

class ColumnAndValue(BaseModel):
    column_and_values : List[SQLColumnValue] = Field(description="The columns and the values associated with them. Ignore Date related columns and values, foucs only on categorical columns.")

example_syntax = {
    "MySQL": "SELECT column_name FROM database_name.table_name WHERE condition;\n",
    "PostgreSQL": 'SELECT column_name FROM "schema_name"."table_name" WHERE condition;\nSELECT DISTINCT column_name FROM "schema_name"."table_name";\n. Never use this syntax :"schema_name.table_name" ',
    "Snowflake": 'SELECT column_name FROM schema_name.table_name WHERE condition;\n',
    "SQL Server": 'SELECT column_name FROM schema_name.table_name WHERE condition;\n'
}

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[logging.StreamHandler()]
)

class Text2SQL(QdrantVectorStore, SQLConnector,AddTableContext,Schema2Chunks):

    def __init__(self,model_name,api_key,db_type,host,port,username,password,database,db_location=None, db_url="http://qdrant:6333",dense_model="sentence-transformers/all-MiniLM-L6-v2", sparse_model="prithivida/Splade_PP_en_v1", hybrid=True,override_existing_index=False,max_attempts=5,add_additional_context=True) -> None:
        
        self.api_key = os.getenv("OPENAI_API_KEY") or api_key

        if not self.api_key:

            raise ValueError("Please Set Your OPENAI_API_KEY.")

        # Initialize QdrantVectorStore

        self.collection_name = f"{self._deterministic_uuid(content=f"{host,port,username,password,database}")}_collection"

        if not db_url:

            db_location = self._deterministic_uuid(content=f"{host,port,username,password,database}")

            QdrantVectorStore.__init__(self,db_location=db_location,collection_name=self.collection_name, dense_model=dense_model, sparse_model=sparse_model, hybird=hybrid)
        
        else:

            QdrantVectorStore.__init__(self, url=db_url,collection_name=self.collection_name, dense_model=dense_model, sparse_model=sparse_model, hybird=hybrid)
        
        # Initialize SQLConnector
        SQLConnector.__init__(self,db_type,host,port,username,password,database)

        AddTableContext.__init__(self,model_name)

        Schema2Chunks.__init__(self,model_name)

        self.model_name = model_name

        self.max_attempts = max_attempts

        self.override_existing_index=override_existing_index

        self.add_additional_context = add_additional_context

        self.instructor_client = instructor.from_openai(instructor_OpenAI(api_key=self.api_key))

        self.db_type,self.host,self.port,self.username,self.password,self.database = db_type,host,port,username,password,database

        self.__connect_to_db()

    def __connect_to_db(self):
        try:
           # Use getattr to dynamically call the correct method
            logging.info(f"Connecting to The Database.....!")

            if self.db_type == "PostgreSQL":
                func = SQLConnector.connect_to_postgresql
            elif self.db_type == "Snowflake":
                func = SQLConnector.connect_to_snowflake
            elif self.db_type == "SQL Server":
                func = SQLConnector.connect_to_sql_server
            elif self.db_type == "MySQL":
                func = SQLConnector.connect_to_mysql
            else:
                raise ValueError(f"Unsupported database type: {self.db_type}")

            # Call the connection function
            try:
                func(self, self.host, self.port, self.username, self.password, self.database)
            except Exception as conn_error:
                logging.error(f"Database connection failed: {str(conn_error)}")
                return None

        # Check if connection was successful        
            # Critical check: Verify schema_description was populated
            if self.schema_description is None:
                logging.error("CRITICAL: schema_description is None after database connection!")
                logging.error("Your connection method is not setting self.schema_description properly")
                logging.error("Check your connection method implementation")
                return None
            
            if hasattr(self.schema_description, 'empty') and self.schema_description.empty:
                logging.error("CRITICAL: schema_description DataFrame is empty!")
                logging.error("No tables found in the database or insufficient permissions")
                return None
            
            logging.info(f"Schema description loaded successfully: {self.schema_description.shape}")
            logging.info(f"Tables found: {self.schema_description['table_name'].nunique() if 'table_name' in self.schema_description.columns else 'unknown'}")
            
            # Check vector database conditions
            try:
                collection_exists = self.client_qdrant.collection_exists(self.collection_name)
                collection_count = self.client_qdrant.count(self.collection_name).count if collection_exists else 0
            except Exception as qdrant_error:
                logging.error(f"Error checking Qdrant collection: {str(qdrant_error)}")
                return None
            
            if not collection_exists or collection_count == 0 or self.override_existing_index:
                filtred_data = self.schema_description
                
                # Extract table relationships with error handling
                try:
                    common_cols = self.extract_table_relationships(filtred_data)
                    logging.info(f"Common columns extracted: {len(common_cols) if common_cols else 0}")
                except Exception as rel_error:
                    logging.error(f"Error extracting table relationships: {str(rel_error)}")
                    common_cols = []
                
                if self.add_additional_context:
                    documents = []
                    ids = []
                    metadata = []
                    
                    try:
                        data_points = asyncio.run(self.process_all_schema(filtred_data, common_cols))
                        
                        # Handle the case where process_all_schema returns a list instead of dict
                        if isinstance(data_points, list):
                            if not data_points:
                                logging.warning("process_all_schema returned empty list")
                                return None
                            else:
                                logging.error("process_all_schema returned list instead of dict")
                                return None
                        
                        if not isinstance(data_points, dict):
                            logging.error(f"process_all_schema returned {type(data_points)}, expected dict")
                            return None
                        
                        if not data_points:
                            logging.warning("process_all_schema returned empty dict")
                            return None
                        
                        # Process data points
                        for key in data_points.keys():
                            try:
                                if 'chunks' not in data_points[key]:
                                    logging.warning(f"No 'chunks' found in data_points[{key}]")
                                    continue
                                
                                if 'ids' not in data_points[key]:
                                    logging.warning(f"No 'ids' found in data_points[{key}]")
                                    continue
                                
                                documents.extend(data_points[key]['chunks'])
                                metadata.extend([{
                                    "table_id": data_points[key]['ids'][0] if data_points[key]['ids'] else f"table_{key}",
                                    "text_data": data_points[key].get('text_data', ''),
                                    "relationships": data_points[key].get('relationships', []),
                                    "common_columns": data_points[key].get('common_columns', [])
                                }] * len(data_points[key]['chunks']))
                                
                            except Exception as process_error:
                                logging.error(f"Error processing data_points[{key}]: {str(process_error)}")
                                continue
                        
                        if not documents:
                            logging.warning("No documents generated from schema processing")
                            return None
                        
                        logging.info(f"Prepared {len(documents)} documents for VectorDB")
                        
                    except Exception as schema_error:
                        logging.error(f"Error in schema processing: {str(schema_error)}")
                        import traceback
                        traceback.print_exc()
                        return None
                    
                    logging.info(f"Adding Schema details to VectorDB.....!")
                    
                else:
                    # Alternative path
                    try:
                        ids = []
                        if not hasattr(self, 'schema_data_to_train') or self.schema_data_to_train is None:
                            logging.error("schema_data_to_train is not set")
                            return None
                        
                        schemas = self.schema_data_to_train.to_dict(orient="records")
                        documents, metadata = self.split_text(schemas, common_cols)
                        logging.info(f"Adding Schema details to VectorDB.....!")
                        
                    except Exception as split_error:
                        logging.error(f"Error in split_text processing: {str(split_error)}")
                        return None
                
                # Add documents to vector database
                try:
                    return self.add_documents_to_schema_details(documents, ids, metadata)
                except Exception as add_error:
                    logging.error(f"Error adding documents to VectorDB: {str(add_error)}")
                    return None
            
            else:
                logging.info("Using existing vector database collection")
                return True
                
        except Exception as e:
            logging.error(f"Unexpected error in __connect_to_db: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
    
    def TextAgent(self,messages,format):
        format = self.instructor_client.chat.completions.create(
            model=self.model_name,
            response_model=format,
            messages=messages,
            temperature=0,
            max_retries=self.max_attempts
        )
        return format.model_dump()

    def execute_inertmediate_query(self,user_question,sub_query):

        intermediate_results = "Below are the outputs of intermediate queries.\n\n"
        
        try:

            df = self.run_sql_query(sub_query)

            if df.shape[0]>=50:

                try:

                    messages=[
                    {"role":"system","content":"You are an helful assistant"},
                    {"role": "user", "content": self.prepare_user_prompt_to_get_column_and_value(user_question,sub_query)}
                    ]

                    col_val = self.TextAgent(messages,ColumnAndValue)
                    
                    df = self.reorder_dataframe(df,col_val)
                    
                except Exception as e:
                    logger.error(f"Error Occured while sorting the df: {str(e)}")

            intermediate_results+="Intermediate Query : "+sub_query + "\n"

            intermediate_results+="Output :\n"+df.iloc[:50,:].to_markdown()

            intermediate_results+="\n\n******************************************\n\n"

        except Exception as e:
            intermediate_results = f"I am getting the following error while executing the given SQL queries: {e} Please give me the correct query."
        return intermediate_results

    def prepare_user_prompt_to_get_column_and_value(self,query,sql_query):

        user_prompt_col_val = """You are given with user question and an intermediate sql query. you have to extract the column name from the SQL query and value from the user question which is associated with the given SQL query."""

        user_prompt_col_val+=f"\n\nUser Question: {query}"

        user_prompt_col_val+=f"\n\nIntermediate Query: {sql_query}"

        return user_prompt_col_val

    def reorder_dataframe(self,df, column_and_values):

        print("************************ Re-ordering ***********************************")
        
        final_df = []
        
        for i in column_and_values['column_and_values']:
            
            column = i['column']
            
            if column in df.columns:
                
                closest_matchs = []
                
                for value in i['value']:

                    standardized_value = value.replace("_", " ").title()
                    
                    closest_match = get_close_matches(standardized_value, df[column], n=1, cutoff=0.6)
                    
                    if len(closest_match):
                        
                        closest_matchs.append(closest_match[0])
                
                if len(closest_matchs):
                    
                    filtred_df = df[df[column].isin(closest_matchs)].iloc[:5,:]
                    
                    final_df.append(filtred_df)
        if len(final_df):
            reordered_df = pd.concat(final_df+[df]).reset_index(drop=True)
            return reordered_df.iloc[:len(df),:]
        return df


    def get_sql_query_type(self,query):
        """
        Determines the type of a SQL query.
        
        Parameters:
        query (str): The SQL query to analyze.
        
        Returns:
        str: The type of the SQL query (SELECT, INSERT, UPDATE, DELETE, CREATE, DROP, or UNKNOWN).
        """
        query = query.strip().lower()
        
        if query.startswith("select") or query.startswith("with") or query.startswith("(select"):
            return "SELECT"
        elif query.startswith("insert"):
            return "INSERT"
        elif query.startswith("update"):
            return "UPDATE"
        elif query.startswith("delete"):
            return "DELETE"
        elif query.startswith("create"):
            return "CREATE"
        elif query.startswith("drop"):
            return "DROP"
        else:
            return "UNKNOWN"