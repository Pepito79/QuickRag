from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser


def get_answer_gemini(
    model: str,
    query: str,
    docs: list[Document],
):
    """
    Generate an answer with gemini llm by giving as context a list of documents
    Args:
        model (str): the gemini model
        query (str): the query
        doc(list[Document]): list of document retrieved

    Returns:
        str: the answer of the query by using the documents as a context
    """

    template = """Use the following context to answer the question at the end. 
           You must be respectful and helpful, and answer in the language of the question.
           If you don't know the answer, say that you don't know.

           Context: {context}

           Question: {question}
           """

    template_format = ChatPromptTemplate.from_template(template)
    prompt_run = RunnableLambda(
        lambda args: template_format.format_messages(
            context=args["context"], question=args["question"]
        )
    )
    query_runnable = RunnablePassthrough()
    context_runnable = RunnableLambda(
        lambda _: "\n\n".join(doc.page_content for doc in docs)
    )
    llm = ChatGoogleGenerativeAI(
        model=model,
        temperature=0.6,
    )

    chain = (
        {"context": context_runnable, "question": query_runnable}
        | prompt_run
        | llm
        | StrOutputParser()
    )

    return chain.invoke(query)
