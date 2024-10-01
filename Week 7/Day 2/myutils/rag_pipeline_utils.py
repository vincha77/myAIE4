"""
rag_pipeline_utils.py

This python script implements various classes useful for a RAG pipeline.

Currently I have implemented:

   Text splitting
      SimpleTextSplitter: uses RecursiveTextSplitter
      SemanticTextSplitter: uses SemanticChunker (different threshold types can be used)

   VectorStore
      currently only sets up Qdrant vector store in memory
   
   AdvancedRetriever
      simple retriever is a special case - 
      advanced retriever - currently implemented MultiQueryRetriever

"""

from operator import itemgetter
from typing import List

from langchain_core.runnables import RunnablePassthrough
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore

from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document
from datasets import Dataset

from ragas import evaluate


def load_all_pdfs(list_of_pdf_files: List[str]) -> List[Document]:
    alldocs = []
    for pdffile in list_of_pdf_files:
        thisdoc = PyMuPDFLoader(file_path=pdffile).load()
        print(f'loaded {pdffile} with {len(thisdoc)} pages ')
        alldocs.extend(thisdoc)
    print(f'loaded all files: total number of pages: {len(alldocs)} ')
    return alldocs


class SimpleTextSplitter:
    def __init__(self, 
                 chunk_size, 
                 chunk_overlap, 
                 documents):
       self.chunk_size = chunk_size
       self.chunk_overlap = chunk_overlap
       self.documents = documents
       return
    
    def split_text(self):
       text_splitter = RecursiveCharacterTextSplitter(
          chunk_size=self.chunk_size,
          chunk_overlap=self.chunk_overlap
       )
       all_splits = text_splitter.split_documents(self.documents)
       return all_splits


class SemanticTextSplitter:
    def __init__(self, 
                 llm_embeddings=OpenAIEmbeddings(), 
                 threshold_type="interquartile", 
                 documents=None):
       self.llm_embeddings = llm_embeddings
       self.threshold_type = threshold_type
       self.documents = documents
       return
    
    def split_text(self):
       text_splitter = SemanticChunker(
          embeddings=self.llm_embeddings,
          breakpoint_threshold_type="interquartile"
       )

       print(f'loaded {len(self.documents)} to be split ')
       all_splits = text_splitter.split_documents(self.documents)
       print(f'returning docs split into {len(all_splits)} chunks ')
       return all_splits


class VectorStore:
    def __init__(self,
                 location,
                 name,
                 documents,
                 size,
                 embedding=OpenAIEmbeddings()):
       self.location = location
       self.name = name
       self.size = size
       self.documents = documents
       self.embedding = embedding

       self.qdrant_client = QdrantClient(self.location)
       self.qdrant_client.create_collection(
          collection_name=self.name,
          vectors_config=VectorParams(size=self.size, distance=Distance.COSINE),
       )
       return
    
    def set_up_vectorstore(self):
       self.qdrant_vector_store = QdrantVectorStore(
          client=self.qdrant_client,
          collection_name=self.name,
          embedding=self.embedding
       )

       self.qdrant_vector_store.add_documents(self.documents)
       return self


class AdvancedRetriever:
    def __init__(self, 
                 vectorstore):
        self.vectorstore = vectorstore
        return

    def set_up_simple_retriever(self):
        simple_retriever = self.vectorstore.as_retriever(
            search_type='similarity', 
            search_kwargs={
                'k': 5
            }
        )
        return simple_retriever
    
    def set_up_multi_query_retriever(self, llm):
        retriever = self.set_up_simple_retriever()
        advanced_retriever = MultiQueryRetriever.from_llm(
            retriever=retriever, llm=llm
        )
        return advanced_retriever


def run_and_eval_rag_pipeline(location, collection_name, embed_dim, text_splits, embeddings,
                              prompt, qa_llm, metrics, test_df):
    """
    Helper function that runs and evaluates different rag pipelines
        based on different text_splits presented to the pipeline
    """
    # vector store
    vs = VectorStore(location=location, 
                     name=collection_name, 
                     documents=text_splits,
                     size=embed_dim, 
                     embedding=embeddings)

    qdvs = vs.set_up_vectorstore().qdrant_vector_store

    # retriever
    retriever = AdvancedRetriever(vectorstore=qdvs).set_up_simple_retriever()

    # q&a chain using LCEL
    retrieval_chain = (
        {"context": itemgetter("question") | retriever, "question": itemgetter("question")}
        | RunnablePassthrough.assign(context=itemgetter("context"))
        | {"response": prompt | qa_llm, "context": itemgetter("context")}
    )

    # get questions, and ground-truth
    test_questions = test_df["question"].values.tolist()
    test_groundtruths = test_df["ground_truth"].values.tolist()


    # run RAG pipeline
    answers = []
    contexts = []

    for question in test_questions:
        response = retrieval_chain.invoke({"question" : question})
        answers.append(response["response"].content)
        contexts.append([context.page_content for context in response["context"]])

    # Save RAG pipeline results to HF Dataset object
    response_dataset = Dataset.from_dict({
        "question" : test_questions,
        "answer" : answers,
        "contexts" : contexts,
        "ground_truth" : test_groundtruths
    })

    # Run RAGAS Evaluation - using metrics
    results = evaluate(response_dataset, metrics)

    # save results to df
    results_df = results.to_pandas()

    return results, results_df


def set_up_rag_pipeline(location, collection_name, 
                        embeddings, embed_dim, 
                        prompt, qa_llm, 
                        text_splits,):
    """
    Helper function that sets up a RAG pipeline
    Inputs
        location:           memory or persistent store
        collection_name:    name of collection, string
        embeddings:         object referring to embeddings to be used
        embed_dim:          embedding dimension
        prompt:             prompt used in RAG pipeline
        qa_llm:             LLM used to generate response
        text_splits:        list containing text splits

    
    Returns a retrieval chain
    """
    # vector store
    vs = VectorStore(location=location, 
                     name=collection_name, 
                     documents=text_splits,
                     size=embed_dim, 
                     embedding=embeddings)

    qdvs = vs.set_up_vectorstore().qdrant_vector_store

    # retriever
    retriever = AdvancedRetriever(vectorstore=qdvs).set_up_simple_retriever()

    # q&a chain using LCEL
    retrieval_chain = (
        {"context": itemgetter("question") | retriever, "question": itemgetter("question")}
        | RunnablePassthrough.assign(context=itemgetter("context"))
        | {"response": prompt | qa_llm, "context": itemgetter("context")}
    )

    return retrieval_chain


def test_rag_pipeline(retrieval_chain, list_of_questions):
    """
    Tests RAG pipeline
    Inputs
        retrieval_chain:    retrieval chain
        list_of_questions:  list of questions to use to test RAG pipeline
    Output
        List of RAG-pipeline-generated responses to each question
    """
    all_answers = []
    for i, question in enumerate(list_of_questions):
        response = retrieval_chain.invoke({'question': question})
        answer = response["response"].content
        all_answers.append(answer)
    return all_answers


def get_vibe_check_on_list_of_questions(collection_name,
                                        embeddings, embed_dim,
                                        prompt, llm, text_splits,
                                        list_of_questions):
    """
    HELPER FUNCTION
    set up retrieval chain for each scenario and print out results
    of the q_and_a for any list of questions
    """

    # set up baseline retriever
    retrieval_chain = \
        set_up_rag_pipeline(location=":memory:", collection_name=collection_name,
                            embeddings=embeddings, embed_dim=embed_dim, 
                            prompt=prompt, qa_llm=llm,
                            text_splits=text_splits)
                            
    # run RAG pipeline and get responses
    answers = test_rag_pipeline(retrieval_chain, list_of_questions)

    # create question, answer tuples
    q_and_a = [(x, y) for x, y in zip(list_of_questions, answers)]

    # print out question/answer pairs to review the performance of the pipeline
    for i, item in enumerate(q_and_a):
        print('=================')
        print(f'=====question number: {i} =============')
        print(item[0])
        print(item[1])

    return retrieval_chain, q_and_a
