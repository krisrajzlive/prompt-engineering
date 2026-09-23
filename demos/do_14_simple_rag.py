"""A minimal RAG (retrieval-augmented generation) pipeline.

Split docs -> embed -> store in a vector index -> retrieve the most
relevant chunks for a query -> stuff them into the prompt as context ->
let the model answer *from that context* instead of its own training data.
`vectorstore.as_retriever()` returns a `BaseRetriever`, which is itself a
`Runnable` — that's what lets it slot directly into an LCEL chain via
`RunnableParallel`, right alongside the prompt and model (see
demos/do_10_lcel_chains.py for the composition primitives used here).
"""

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough

from core import get_embeddings, get_model

DOCS = [
    Document(page_content="Acme Robotics was founded in 2019 and builds warehouse picking robots."),
    Document(page_content="Acme Robotics' flagship product, the PickBot 3000, has a 99.2% pick accuracy rate."),
    Document(page_content="Acme Robotics is headquartered in Austin, Texas, with a satellite office in Chennai, India."),
    Document(page_content="The PickBot 3000 costs $45,000 per unit and includes a 2-year hardware warranty."),
    Document(page_content="Acme Robotics' main competitor is FetchWorks, which focuses on outdoor delivery robots."),
]

PROMPT = ChatPromptTemplate.from_template(
    "Answer the question using ONLY the context below. If the context doesn't "
    "contain the answer, say you don't know.\n\n"
    "Context:\n{context}\n\n"
    "Question: {question}"
)


def format_docs(docs: list[Document]) -> str:
    return "\n".join(f"- {d.page_content}" for d in docs)


def main() -> None:
    # 1. Build the vector index once from the source documents.
    vectorstore = FAISS.from_documents(DOCS, get_embeddings())
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

    # 2. Inspect retrieval on its own, before it's wired into a chain.
    query = "How accurate is the PickBot 3000?"
    retrieved = retriever.invoke(query)
    print("--- Retrieved chunks for:", query, "---")
    for d in retrieved:
        print(" -", d.page_content)

    # 3. Wire retriever -> prompt -> model -> parser into one RAG chain.
    #    RunnableParallel feeds the same input to both branches: the raw
    #    question passes through unchanged while the retriever+format_docs
    #    branch turns it into context text.
    rag_chain = (
        RunnableParallel(
            context=retriever | format_docs,
            question=RunnablePassthrough(),
        )
        | PROMPT
        | get_model()
        | StrOutputParser()
    )

    print("\n--- RAG answer (question answerable from context) ---")
    print(rag_chain.invoke("How accurate is the PickBot 3000?"))

    print("\n--- RAG answer (question NOT covered by context) ---")
    print(rag_chain.invoke("What programming language does the PickBot 3000 run?"))


if __name__ == "__main__":
    main()
