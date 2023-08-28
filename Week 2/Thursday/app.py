from llama_index import SimpleDirectoryReader, ServiceContext
from llama_index.llms import OpenAI
from llama_index.evaluation import DatasetGenerator
from llama_index import VectorStoreIndex

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import answer_relevancy, faithfulness

import chainlit as cl
import random

@cl.cache
def to_cache():
    documents = SimpleDirectoryReader(
        input_files=["hitchhikers.pdf"]
    ).load_data()

    random.seed(42)
    random.shuffle(documents)

    gpt_35_context = ServiceContext.from_defaults(
        llm=OpenAI(model="gpt-3.5-turbo", temperature=0.3)
    )

    question_gen_query = (
        "You are a Teacher/ Professor. Your task is to setup "
        "a quiz/examination. Using the provided context from a "
        "report on climate change and the oceans, formulate "
        "a single question that captures an important fact from the "
        "context. Restrict the question to the context information provided."
    )

    dataset_generator = DatasetGenerator.from_documents(
        documents[:50],
        question_gen_query=question_gen_query,
        service_context=gpt_35_context,
    )

    questions = dataset_generator.generate_questions_from_nodes(num=40)

    with open("train_questions.txt", "w") as f:
        for question in questions:
            f.write(question + "\n")

    dataset_generator = DatasetGenerator.from_documents(
        documents[
            50:
        ],  # since we generated ~1 question for 40 documents, we can skip the first 40
        question_gen_query=question_gen_query,
        service_context=gpt_35_context,
    )

    questions = dataset_generator.generate_questions_from_nodes(num=40)
    with open("eval_questions.txt", "w") as f:
        for question in questions:
            f.write(question + "\n")


    # limit the context window to 2048 tokens so that refine is used
    gpt_35_context = ServiceContext.from_defaults(
        llm=OpenAI(model="gpt-3.5-turbo", temperature=0.3), context_window=2048
    )

    index = VectorStoreIndex.from_documents(documents, service_context=gpt_35_context)

    query_engine = index.as_query_engine(similarity_top_k=2)
    return query_engine

query_engine = to_cache()

@cl.author_rename
def rename(orig_author: str):
    rename_dict = {"RetrievalQA": "Consulting The Kens"}
    return rename_dict.get(orig_author, orig_author)

@cl.on_chat_start
async def init():
    msg = cl.Message(content=f"Generating training and evaluation questions...")
    await msg.send()

    questions = []
    with open("eval_questions.txt", "r") as f:
        for line in f:
            questions.append(line.strip())

    msg.content = f"Generated {len(questions)} evaluation questions!"
    await msg.send()

    contexts = []
    answers = []

    for question in questions:
        response = query_engine.query(question)
        contexts.append([x.node.get_content() for x in response.source_nodes])
        answers.append(str(response))

    ds = Dataset.from_dict(
        {
            "question": questions,
            "answer": answers,
            "contexts": contexts,
        }
    )

    result = evaluate(ds, [answer_relevancy, faithfulness])

    msg.content = f"Evaluation results: {str(result)}"
    await msg.send()

    cl.user_session.set("query_engine", query_engine)


@cl.on_message
async def main(message):
    query_engine = cl.user_session.get("query_engine")
    res = query_engine.query(message)
    answer = str(res)

    await cl.Message(content=answer).send()