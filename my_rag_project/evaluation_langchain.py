import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFaceHub
from datasets import Dataset
import ragas
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall

# Define constants
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # Free, lightweight model
COLLECTION_NAME = "company_documents_langchain"
CHROMA_DB_PATH = "../chroma_db_langchain"
DOC_DIR = "../GenAI-Dev-Onboarding-Starter-Kit"

def load_documents():
    """Load and process documents for evaluation"""
    print("Loading documents for evaluation...")
    
    # Define document paths
    doc_files = ["doc1.txt", "doc2.txt"]
    doc_paths = [os.path.join(DOC_DIR, filename) for filename in doc_files]
    
    # Load documents using Langchain TextLoader
    all_docs = []
    for path in doc_paths:
        try:
            loader = TextLoader(os.path.abspath(path), encoding="utf-8")
            loaded_docs = loader.load()
            for doc in loaded_docs:
                doc.metadata["filename"] = os.path.basename(path)
            all_docs.extend(loaded_docs)
            print(f"Loaded document: {os.path.basename(path)}")
        except Exception as e:
            print(f"Error loading document {path}: {e}")
    
    if not all_docs:
        print("No documents loaded. Exiting.")
        return None
    
    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        is_separator_regex=False,
    )
    split_docs = text_splitter.split_documents(all_docs)
    print(f"Split documents into {len(split_docs)} chunks.")
    
    return split_docs

def create_vector_store(documents):
    """Create or load vector store"""
    print("Creating/loading vector store...")
    
    try:
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        print(f"Initialized HuggingFaceEmbeddings with model: {EMBEDDING_MODEL}")
    except Exception as e:
        print(f"Error initializing HuggingFaceEmbeddings: {e}")
        return None

    chroma_db_full_path = os.path.abspath(os.path.join(os.path.dirname(__file__), CHROMA_DB_PATH))
    print(f"Connecting to Chroma vector store at: {chroma_db_full_path}")
    
    try:
        vector_store = Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            collection_name=COLLECTION_NAME,
            persist_directory=chroma_db_full_path
        )
        vector_store.persist()
        print("Successfully connected to Chroma vector store")
        return vector_store
    except Exception as e:
        print(f"Error creating/connecting to Chroma vector store: {e}")
        return None

def test_qa_chain(vector_store):
    """Test the QA chain with sample questions"""
    print("Testing QA chain...")
    
    # Sample questions for testing
    test_questions = [
        "What is TechCorp Solutions?",
        "What are the main products offered by TechCorp?",
        "How many employees does TechCorp have?",
        "What is the annual revenue of TechCorp?",
        "What technologies are used in the AI Analytics Platform?"
    ]
    
    # Create a simple QA chain (without LLM for now, just retrieval)
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    
    results = []
    for question in test_questions:
        try:
            # Get relevant documents
            docs = retriever.get_relevant_documents(question)
            contexts = [doc.page_content for doc in docs]
            
            # For now, we'll just show the retrieved contexts
            # In a full implementation, you'd use an LLM to generate answers
            result = {
                "question": question,
                "contexts": contexts,
                "answer": f"Retrieved {len(contexts)} relevant document chunks. Use an LLM to generate a proper answer."
            }
            results.append(result)
            print(f"Question: {question}")
            print(f"Retrieved {len(contexts)} relevant chunks")
            print("-" * 50)
            
        except Exception as e:
            print(f"Error during QA chain invocation for question '{question}': {e}")
            results.append({
                "question": question,
                "contexts": [],
                "answer": "Error retrieving relevant documents"
            })
    
    return results

def run_ragas_evaluation(qa_results):
    """Run Ragas evaluation on the QA results"""
    print("Running Ragas evaluation...")
    
    try:
        # Prepare data for Ragas
        questions = [result["question"] for result in qa_results]
        contexts = [result["contexts"] for result in qa_results]
        answers = [result["answer"] for result in qa_results]
        
        # Create dataset
        dataset = Dataset.from_dict({
            "question": questions,
            "contexts": contexts,
            "answer": answers
        })
        
        # Run Ragas metrics
        result = ragas.evaluate(
            dataset,
            metrics=[
                context_precision,
                faithfulness,
                answer_relevancy,
                context_recall
            ]
        )
        
        print("Ragas evaluation results:")
        print(result)
        return result
        
    except Exception as e:
        print(f"Error during Ragas evaluation: {e}")
        return None

def main():
    print("Starting Langchain-based evaluation process...")
    
    # Load documents
    documents = load_documents()
    if not documents:
        return
    
    # Create vector store
    vector_store = create_vector_store(documents)
    if not vector_store:
        return
    
    # Test QA chain
    print("\n" + "="*60)
    qa_results = test_qa_chain(vector_store)
    
    # Run Ragas evaluation
    print("\n" + "="*60)
    ragas_results = run_ragas_evaluation(qa_results)
    
    print("\nLangchain-based evaluation process completed.")

if __name__ == "__main__":
    main() 