from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain_postgres import PGVector

class RAGSearchTool:
    def __init__(self, connection_string, collection_name="rag_docs", model_name="gpt-3.5-turbo"):
        self.vectorstore = PGVector(
            connection=connection_string,
            embedding_function=ChatOpenAI(model_name="text-embedding-3-small").embeddings,
            collection_name=collection_name
        )
        # Retriever for RAG queries
        self.retriever = self.vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=ChatOpenAI(temperature=0, model_name=model_name),
            retriever=self.retriever
        )

    def run(self, query: str):
        """
        Executes RAG search and returns summarized answer.
        """
        result = self.qa_chain.run(query)
        return result
