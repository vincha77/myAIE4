"""
ragas_pipeline.py

Implements the core pipeline to generate test set for RAGAS.

"""

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ragas.testset.generator import TestsetGenerator
from ragas import evaluate

from datasets import Dataset

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
        ragas_text_splits = text_splitter.split_text()

        testset = self.generator.generate_with_langchain_docs(
            ragas_text_splits, 
            self.number_of_qa_pairs, 
            self.distributions
        )

        testset_df = testset.to_pandas()
        return testset_df

    def ragas_eval_of_rag_pipeline(self, retrieval_chain, ragas_questions, ragas_groundtruths, ragas_metrics):
        """
        Helper function that runs and evaluates different rag pipelines
            based on RAGAS test questions
        """

        # run RAG pipeline on RAGAS synthetic questions
        answers = []
        contexts = []

        for question in ragas_questions:
            response = retrieval_chain.invoke({"question" : question})
            answers.append(response["response"].content)
            contexts.append([context.page_content for context in response["context"]])

        # Save RAG pipeline results to HF Dataset object
        response_dataset = Dataset.from_dict({
            "question" : ragas_questions,
            "answer" : answers,
            "contexts" : contexts,
            "ground_truth" : ragas_groundtruths
        })

        # Run RAGAS Evaluation - using metrics
        results = evaluate(response_dataset, ragas_metrics)

        # save results to df
        results_df = results.to_pandas()

        return results, results_df
