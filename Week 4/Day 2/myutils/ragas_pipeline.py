"""
ragas_pipeline.py

Implements the core pipeline to generate test set for RAGAS.

"""

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ragas.testset.generator import TestsetGenerator

from myutils.rag_pipeline_utils import SimpleTextSplitter, SemanticTextSplitter, VectorStore, AdvancedRetriever


class RagasPipeline:
    def __init__(self, generator_llm_model, critic_llm_model, embedding_model,
                 number_of_qa_pairs, 
                 chunk_size, chunk_overlap, documents,
                 distributions):
        self.generator_llm = ChatOpenAI(model=generator_llm_model)
        self.critic_llm = ChatOpenAI(model=critic_llm_model)
        self.embeddings = OpenAIEmbeddings(model=embedding_model)
        self.number_of_qa_pairs = number_of_qa_pairs

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.documents = documents

        self.distributions = distributions

        self.generator = TestsetGenerator.from_langchain(
            self.generator_llm,
            self.critic_llm,
            self.embeddings
        )
        return
    
    def generate_testset(self):
        text_splitter = SimpleTextSplitter(
            chunk_size=self.chunk_size, 
            chunk_overlap=self.chunk_overlap, 
            documents=self.documents
        )
        ragas_text_splits = text_splitter.split_text().all_splits

        testset = self.generator.generate_with_langchain_docs(
            ragas_text_splits, 
            self.number_of_qa_pairs, 
            self.distributions
        )

        testset_df = testset.to_pandas()
        return testset_df
