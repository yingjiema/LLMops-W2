# External Libraries
import pandas as pd
from sqlalchemy import create_engine
from typing import List
from pydantic import BaseModel, Field

# llama_index Imports
from llama_index import ServiceContext
from llama_index.node_parser.simple import SimpleNodeParser
from llama_index.langchain_helpers.text_splitter import TokenTextSplitter
from llama_index.llms import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding

from llama_index import VectorStoreIndex
from llama_index.vector_stores import ChromaVectorStore
from llama_index.storage.storage_context import StorageContext
import chromadb
from llama_index.readers.wikipedia import WikipediaReader
from llama_index.tools import FunctionTool
from llama_index.vector_stores.types import (
    VectorStoreInfo,
    MetadataInfo,
    ExactMatchFilter,
    MetadataFilters,
)
from llama_index.retrievers import VectorIndexRetriever
from llama_index.query_engine import RetrieverQueryEngine
from llama_index.agent import OpenAIAgent
from llama_index import SQLDatabase
from llama_index.indices.struct_store.sql_query import NLSQLTableQueryEngine
from llama_index.tools.query_engine import QueryEngineTool

import chainlit as cl

class AutoRetrieveModel(BaseModel):
    query: str = Field(..., description="natural language query string")
    filter_key_list: List[str] = Field(
        ..., description="List of metadata filter field names"
    )
    filter_value_list: List[str] = Field(
        ...,
        description=(
            "List of metadata filter field values (corresponding to names specified in filter_key_list)"
        )
    )

@cl.author_rename
def rename(orig_author: str):
    rename_dict = {"RetrievalQA": "Consulting The Kens"}
    return rename_dict.get(orig_author, orig_author)

@cl.on_chat_start
async def init():
    msg = cl.Message(content=f"Building Index...")
    await msg.send()

    embed_model = OpenAIEmbedding(embed_batch_size=10) ### YOUR CODE HERE
    chunk_size = 2048 ### YOUR CODE HERE
    llm = OpenAI(
        temperature=0, 
        model="gpt-3.5-turbo", 
        streaming=True
    )

    service_context = ServiceContext.from_defaults(
        llm=llm, 
        chunk_size=chunk_size, 
        embed_model=embed_model
    )

    text_splitter = TokenTextSplitter(
        chunk_size=chunk_size
    )

    node_parser = SimpleNodeParser(
        text_splitter=text_splitter
    )

    # ### BarbenHeimer Wikipedia Retrieval Tool w/ `QueryEngine`!
    # #### ChromaDB

    chroma_client = chromadb.Client()
    chroma_collection = chroma_client.get_or_create_collection("wikipedia_barbie_opp")

    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    wiki_vector_index = VectorStoreIndex([], storage_context=storage_context, service_context=service_context)

    movie_list = ["Barbie (film)", "Oppenheimer (film)"]
    wiki_docs = WikipediaReader().load_data(pages=movie_list, auto_suggest=False)

    # #### Node Construction

    for movie, wiki_doc in zip(movie_list, wiki_docs):
        nodes = node_parser.get_nodes_from_documents([wiki_doc])
        for node in nodes:
            node.metadata = {"title" : movie}
        wiki_vector_index.insert_nodes(nodes)

    # #### Auto Retriever Functional Tool
    # First, we need to create our `VectoreStoreInfo` object which will hold all the relevant metadata we need for each component (in this case title metadata).

    top_k = 3

    vector_store_info = VectorStoreInfo(
        content_info="semantic information about movies",
        metadata_info=[MetadataInfo(
            name="title",
            type="str",
            description="title of the movie, one of [Barbie (film), Oppenheimer (film)]",
        )]
    )

    # Now we can build our function that we will use to query the functional endpoint.
    # >The `docstring` is important to the functionality of the application.
    def auto_retrieve_fn(
        query: str, filter_key_list: List[str], filter_value_list: List[str]
    ):
        """Auto retrieval function.
        Performs auto-retrieval from a vector database, and then applies a set of filters.
        """
        query = query or "Query"

        exact_match_filters = [
            ExactMatchFilter(key=k, value=v)
            for k, v in zip(filter_key_list, filter_value_list)
        ]
        retriever = VectorIndexRetriever(
            wiki_vector_index, filters=MetadataFilters(filters=exact_match_filters), top_k=top_k
        )
        query_engine = RetrieverQueryEngine.from_args(retriever)

        response = query_engine.query(query)
        return str(response)

    # Now we need to wrap our system in a tool in order to integrate it into the larger application.
    description = f"""\
    Use this tool to look up semantic information about films.
    The vector database schema is given below:
    {vector_store_info.json()}
    """

    auto_retrieve_tool = FunctionTool.from_defaults(
        fn=auto_retrieve_fn,
        name="AutoRetrieve",
        description=description,
        fn_schema=AutoRetrieveModel,
    )

    # ### BarbenHeimer SQL Tool

    barbie_df = pd.read_csv("barbie_data/barbie.csv")
    oppenheimer_df = pd.read_csv("oppenheimer_data/oppenheimer.csv")

    # #### Create SQLAlchemy engine with SQLite

    engine = create_engine("sqlite+pysqlite:///:memory:")

    # #### Convert `pd.DataFrame` to SQL tables

    barbie_df.to_sql(
        "barbie",
        engine
    )

    oppenheimer_df.to_sql(
        "oppenheimer",
        engine
    )

    # #### Construct a `SQLDatabase` index

    sql_database = SQLDatabase(
        engine, 
        include_tables=["barbie", "oppenheimer"])

    # #### Create the NLSQLTableQueryEngine interface for all added SQL tables

    sql_query_engine = NLSQLTableQueryEngine(
        sql_database=sql_database,
        tables=["barbie", "oppenheimer"]
    )

    # #### Wrap It All Up in a `QueryEngineTool`

    sql_tool = QueryEngineTool.from_defaults(
        query_engine=sql_query_engine,
        name="SQL",
        description=(
            "Useful for translating a natural language query into a SQL query over a table containing: "
            "barbie, containing information related to reviews of the Barbie movie"
            "oppenheimer, containing information related to reviews of the Oppenheimer movie"
        ),
    )

    # ### Combining The Tools Together
    # Now, we can simple add our tools into the `OpenAIAgent`, and off we go!

    barbenheimer_agent = OpenAIAgent.from_tools(
        [sql_tool, auto_retrieve_tool], llm=llm, verbose=True
    )

    msg.content = f"Index built!"
    await msg.send()

    cl.user_session.set("agent", barbenheimer_agent)


@cl.on_message
async def main(message):
    agent = cl.user_session.get("agent")
    res = agent.chat(message)
    answer = str(res)

    await cl.Message(content=answer).send()