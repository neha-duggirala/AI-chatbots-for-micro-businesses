from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain_qdrant import QdrantVectorStore
from transformers import pipeline
from dotenv import load_dotenv
load_dotenv()

def format_docs(docs):
    """Utility function to concatenate retrieved document contents."""
    return "\n\n".join(doc.page_content for doc in docs)


def main():
    # 1. Define sample knowledge base
    docs = [
        Document(
            page_content="Qdrant is an open-source vector database written in Rust. It provides a fast and scalable matching engine for high-dimensional vectors."
        ),
        Document(
            page_content="LangChain is a framework designed to simplify the creation of applications using large language models through composable building blocks."
        ),
        Document(
            page_content="Hugging Face provides thousands of open-source models, datasets, and pipelines for natural language processing and machine learning tasks."
        ),
        Document(
            page_content="Retrieval-Augmented Generation (RAG) is a technique that improves LLM responses by fetching authoritative context from an external knowledge base."
        ),
    ]

    # 2. Modern Hugging Face Embeddings integration
    print("Loading embedding model...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # 3. In-Memory Qdrant Vector Store (using langchain-qdrant)
    print("Initializing Qdrant in-memory vector store...")
    vector_store = QdrantVectorStore.from_documents(
        documents=docs,
        embedding=embeddings,
        location=":memory:",
        collection_name="local_rag",
    )
    retriever = vector_store.as_retriever(search_kwargs={"k": 2})

    # 4. Hugging Face LLM Pipeline
    print("Loading local Hugging Face generation pipeline...")
    hf_pipeline = pipeline(
        "text-generation",
        model="gpt2",
        max_new_tokens=100,
        do_sample=True,
        temperature=0.7,
        pad_token_id=50256,
    )
    llm = HuggingFacePipeline(pipeline=hf_pipeline)

    # 5. Define Chat Prompt Template
    system_prompt = (
        "You are a helpful assistant. Use the following context to answer the question.\n"
        "Context:\n{context}\n\n"
        "Question: {question}"
    )
    prompt = ChatPromptTemplate.from_template(system_prompt)

    # 6. Construct Modern LCEL Chain (No legacy helpers or chains)
    # RunnableParallel extracts context (retriever + format_docs) and forwards question in parallel
    rag_chain = (
        RunnableParallel(
            {
                "context": retriever | format_docs,
                "question": RunnablePassthrough(),
            }
        )
        | prompt
        | llm
        | StrOutputParser()
    )

    # 7. Run Query
    query = "What is Qdrant?"
    print(f"\n--- Query: {query} ---")
    response = rag_chain.invoke(query)

    print("\n--- Generated Answer ---")
    print(response)


if __name__ == "__main__":
    main()