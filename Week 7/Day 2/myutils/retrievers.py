"""
retrievers.py

Collects all retrievers for convenience

"""

from langchain_community.vectorstores import Qdrant
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI

from langchain_community.retrievers import BM25Retriever

from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
# from langchain_cohere import CohereRerank
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain_openai import OpenAI

from langchain.retrievers.multi_query import MultiQueryRetriever

from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient, models

from langchain.retrievers import EnsembleRetriever

from langchain_core.runnables import RunnablePassthrough
from operator import itemgetter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate


chat_model = ChatOpenAI()


class Retrievers:
    def __init__(self,
                 documents=None,
                 collection_name="JohnWick",
                 pd_collection_name="full_documents",
                 embeddings_model="text-embedding-3-small",
                 cohere_rerank_model="rerank-english-v3.0",
                 mq_chat_model=chat_model,
                 child_chunk_size_for_pd_retriever=200
                ):
        self.documents = documents
        self.embeddings_model = embeddings_model
        self.collection_name = collection_name
        self.pd_collection_name = pd_collection_name
        self.cohere_rerank_model = cohere_rerank_model
        self.mq_chat_model = mq_chat_model
        self.child_chunk_size_for_pd_retriever = child_chunk_size_for_pd_retriever

        self._set_up_embeddings()
        self._set_up_basic_vectorstore()
        return
    
    def _set_up_embeddings(self):
        self.embeddings = OpenAIEmbeddings(model=self.embeddings_model)
        return self
    
    def _set_up_basic_vectorstore(self):
        self.vectorstore = Qdrant.from_documents(
            self.documents,
            self.embeddings,
            location=":memory:",
            collection_name=self.collection_name
        )
        return self
    
    def _set_up_pd_text_splitter_and_vectorstores(self):
        self.child_splitter = \
            RecursiveCharacterTextSplitter(
                chunk_size=self.child_chunk_size_for_pd_retriever
            )

        client = QdrantClient(location=":memory:")
        client.create_collection(
            collection_name=self.pd_collection_name,
            vectors_config=models.VectorParams(size=1536, distance=models.Distance.COSINE)
        )
        self.parent_document_vectorstore = Qdrant(
            collection_name=self.pd_collection_name, 
            embeddings=self.embeddings, 
            client=client
        )
        self.store = InMemoryStore()
        return self

    def get_naive_retriever(self):
        naive_retriever = self.vectorstore.as_retriever(search_kwargs={"k" : 10})
        return naive_retriever
    
    def get_bm25_retriever(self):
        bm25_retriever = BM25Retriever.from_documents(self.documents)
        return bm25_retriever
    
    def get_cohere_contextual_compression_retriever(self):
        compressor = CohereRerank(model=self.cohere_rerank_model)
        self.naive_retriever = self.get_naive_retriever()
        cohere_contextual_compression_retriever = ContextualCompressionRetriever(
            base_compressor=compressor, base_retriever=self.naive_retriever
        )
        return cohere_contextual_compression_retriever
    
    def get_langchain_compression_retriever(self):
        llm = OpenAI(temperature=0)
        compressor = LLMChainExtractor.from_llm(llm)
        self.naive_retriever = self.get_naive_retriever()
        langchain_compression_retriever = ContextualCompressionRetriever(
            base_compressor=compressor, base_retriever=self.naive_retriever
        )
        return langchain_compression_retriever

    def get_multi_query_retriever(self):
        self.naive_retriever = self.get_naive_retriever()
        multi_query_retriever = MultiQueryRetriever.from_llm(
            retriever=self.naive_retriever, llm=self.mq_chat_model
        )
        return multi_query_retriever
    
    def get_parent_document_retriever(self):
        self._set_up_pd_text_splitter_and_vectorstores()
        parent_document_retriever = ParentDocumentRetriever(
            vectorstore = self.parent_document_vectorstore,
            docstore=self.store,
            child_splitter=self.child_splitter
        )

        parent_document_retriever.add_documents(self.documents, ids=None)
        return parent_document_retriever
    
    def get_ensemble_retriever(self):
        naive_retriever = self.get_naive_retriever()
        bm25_retriever = self.get_bm25_retriever()
        # cohere_contextual_compression_retriever = self.get_cohere_contextual_compression_retriever()
        langchain_compression_retriever = self.get_langchain_compression_retriever()
        multi_query_retriever = self.get_multi_query_retriever()
        parent_document_retriever = self.get_parent_document_retriever()

        retriever_list = [
            naive_retriever,
            bm25_retriever,
            langchain_compression_retriever,
            # cohere_contextual_compression_retriever,
            multi_query_retriever,
            parent_document_retriever
        ]

        equal_weighting = [1/len(retriever_list)] * len(retriever_list)

        ensemble_retriever = EnsembleRetriever(
            retrievers=retriever_list, weights=equal_weighting
        )
        return ensemble_retriever
    
    def get_retriever(self, retriever_type):
        look_up_retriever_method = {
            'naive_retriever': self.get_naive_retriever,
            'bm25_retriever': self.get_bm25_retriever,
            # 'cohere_contextual_compression_retriever': self.get_cohere_contextual_compression_retriever,
            'langchain_compression_retriever': self.get_langchain_compression_retriever,
            'multi_query_retriever': self.get_multi_query_retriever,
            'parent_document_retriever': self.get_parent_document_retriever,
            'ensemble_retriever': self.get_ensemble_retriever
        }
        final_retriever = look_up_retriever_method[retriever_type]()
        return final_retriever
