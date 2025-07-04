# import os
# import subprocess
# import sys
# import getpass

# project_root = os.getcwd()
# script_dir = os.path.join(project_root, "my_rag_project") # where scripts go

# print(project_root)
# print(script_dir)

# try:
#     result = subprocess.run(["poetry", "--version"], capture_output=True, text=True, check=True)
#     print(f"Poetry version: {result.stdout.strip()}")
#     print("Poetry is installed and working.")
# except (subprocess.CalledProcessError, FileNotFoundError):
#     print("Poetry is not installed or not in PATH.")
#     print("Please install Poetry manually using: curl -sSL https://install.python-poetry.org | python3 -")

# api_key = os.environ.get("OPENAI_API_KEY")

# if not api_key:
#     # Request for new OpenAI API key if none available
#     api_key = getpass("Please enter your OpenAI API key: ")
#     os.environ["OPENAI_API_KEY"] = api_key

# if os.environ.get("OPENAI_API_KEY"):
#     print("OpenAI API key set successfully!")
# else:
#     print("Failed to set OpenAI API key.")

import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# Define constants
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # Free, lightweight model
COLLECTION_NAME = "company_documents_langchain"
CHROMA_DB_PATH = "../chroma_db_langchain"
DOC_DIR = "."

def main():
    print("Starting Langchain-based embedding process with free model...")

    # Define document paths
    doc_files = ["doc1.txt", "doc2.txt"]
    doc_paths = [os.path.join(DOC_DIR, filename) for filename in doc_files]
    
    print(f"Looking for documents in: {os.path.abspath(DOC_DIR)}")
    print(f"Document paths: {doc_paths}")

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
            print(f"Error loading document {path} with Langchain: {e}")
    
    if not all_docs:
        print("No documents loaded. Exiting.")
        return
    print(f"Total documents loaded via Langchain: {len(all_docs)}")

    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        is_separator_regex=False,
    )
    split_docs = text_splitter.split_documents(all_docs)
    print(f"Split documents into {len(split_docs)} chunks.")

    if not split_docs:
        print("No chunks generated. Exiting.")
        return

    # Initialize free embeddings model
    try:
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        print(f"Initialized HuggingFaceEmbeddings with model: {EMBEDDING_MODEL}")
    except Exception as e:
        print(f"Error initializing HuggingFaceEmbeddings: {e}")
        return

    # Initialize ChromaDB vector store
    chroma_db_full_path = os.path.abspath(os.path.join(os.path.dirname(__file__), CHROMA_DB_PATH))
    print(f"Initializing Chroma vector store at: {chroma_db_full_path}")
    
    try:
        vector_store = Chroma.from_documents(
            documents=split_docs,
            embedding=embeddings,
            collection_name=COLLECTION_NAME,
            persist_directory=chroma_db_full_path
        )
        vector_store.persist()
        print(f"Successfully created/updated Chroma vector store")
        print(f"Total items in collection: {vector_store._collection.count()}")

    except Exception as e:
        print(f"Error creating/updating Chroma vector store with Langchain: {e}")
        return

    print("Langchain-based embedding process completed with free model.")

if __name__ == "__main__":
    main()