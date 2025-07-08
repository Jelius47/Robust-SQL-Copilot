# ------------------------------------------------- Chroma Vector Store -------------------------------- #
import sys
sys.dont_write_bytecode = True
import chromadb
from chromadb.db.base import UniqueConstraintError
from chromadb.utils import embedding_functions
import uuid
import os
import logging

os.environ['ALLOW_RESET'] = 'TRUE'

class ChromaStore:
    def __init__(self, store_id, embedding_model_name='all-MiniLM-L6-v2'):
        """Initialize ChromaDB store with persistent client"""
        try:
            self.client = chromadb.PersistentClient(f"chroma_db/{store_id}")
            self.em = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=embedding_model_name
            )
            logging.info(f"ChromaStore initialized with store_id: {store_id}")
        except Exception as e:
            logging.error(f"Failed to initialize ChromaStore: {e}")
            raise

    def add_documents(self, documents, ids, meta_data, collection_name='sample'):
        """Add documents to collection with error handling"""
        try:
            # Try to get existing collection first
            try:
                self.collection = self.client.get_collection(
                    name=collection_name, 
                    embedding_function=self.em
                )
                logging.info(f"Using existing collection: {collection_name}")
            except ValueError:
                # Collection doesn't exist, create new one
                self.collection = self.client.create_collection(
                    name=collection_name, 
                    embedding_function=self.em
                )
                logging.info(f"Created new collection: {collection_name}")
            
            self.collection.add(
                documents=documents,
                ids=ids,
                metadatas=meta_data
            )
            return f"Successfully added {len(documents)} documents to {collection_name}"
            
        except UniqueConstraintError:
            logging.warning(f"Collection {collection_name} already exists, recreating...")
            self.client.reset()
            self.collection = self.client.create_collection(
                name=collection_name, 
                embedding_function=self.em
            )
            self.collection.add(
                documents=documents,
                ids=ids,
                metadatas=meta_data
            )
            return f"Recreated collection and added {len(documents)} documents"
        except Exception as e:
            logging.error(f"Error adding documents: {e}")
            raise

    def get_relevant_documents(self, query, collection_name='sample', no_of_docs=10):
        """Get relevant documents with error handling"""
        try:
            if not hasattr(self, 'collection'):
                self.collection = self.client.get_collection(
                    name=collection_name, 
                    embedding_function=self.em
                )
            
            results = self.collection.query(
                query_texts=[query] if isinstance(query, str) else query,
                n_results=no_of_docs
            )
            return results
        except Exception as e:
            logging.error(f"Error querying documents: {e}")
            raise

    def reset(self, collection_name=None):
        """Reset specific collection or entire client"""
        try:
            if collection_name:
                self.client.delete_collection(name=collection_name)
                return f"Deleted collection: {collection_name}"
            else:
                self.client.reset()
                return "Deleted all existing data"
        except Exception as e:
            logging.error(f"Error resetting: {e}")
            raise

# ------------------------------------------------- Qdrant Vector Store -------------------------------- #
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import ResponseHandlingException
from tqdm import tqdm
import time

class QdrantVectorStore:
    def __init__(self, url=None, db_location="qdrant", collection_name="schema_details",
                 dense_model="sentence-transformers/all-MiniLM-L6-v2", 
                 sparse_model="prithivida/Splade_PP_en_v1", hybrid=True, 
                 timeout=60, max_retries=3) -> None:
        """Initialize Qdrant client with retry logic and timeout handling"""
        
        self.collection_name = collection_name
        self.max_retries = max_retries
        self.timeout = timeout
        
        # Initialize client with timeout settings
        client_kwargs = {
            'timeout': {
                'connect': 30,
                'read': timeout,
                'write': timeout,
                'pool': timeout
            }
        }
        
        if url:
            client_kwargs['url'] = url
        else:
            client_kwargs['path'] = f"vector_stores/{db_location}"
        
        self.client = self._initialize_client_with_retry(**client_kwargs)
        
        # Set embedding models
        self.client.set_model(dense_model)
        if hybrid:
            self.client.set_sparse_model(sparse_model)
        
        # Create collection with retry logic
        self._create_collection_with_retry(hybrid)
    
    def _initialize_client_with_retry(self, **client_kwargs):
        """Initialize Qdrant client with retry logic"""
        for attempt in range(self.max_retries):
            try:
                client = QdrantClient(**client_kwargs)
                # Test connection
                client.get_collections()
                logging.info("Successfully connected to Qdrant")
                return client
            except Exception as e:
                if attempt < self.max_retries - 1:
                    wait_time = (attempt + 1) * 5
                    logging.warning(f"Connection attempt {attempt + 1} failed: {e}. Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    logging.error(f"Failed to connect to Qdrant after {self.max_retries} attempts: {e}")
                    raise
    
    def _create_collection_with_retry(self, hybrid=True):
        """Create collection with retry logic"""
        for attempt in range(self.max_retries):
            try:
                # Check if collection exists
                if not self.client.collection_exists(self.collection_name):
                    if hybrid:
                        self.client.recreate_collection(
                            collection_name=self.collection_name,
                            vectors_config=self.client.get_fastembed_vector_params(),
                            sparse_vectors_config=self.client.get_fastembed_sparse_vector_params(),
                        )
                    else:
                        self.client.recreate_collection(
                            collection_name=self.collection_name,
                            vectors_config=self.client.get_fastembed_vector_params()
                        )
                    logging.info(f"Created collection: {self.collection_name}")
                else:
                    logging.info(f"Collection {self.collection_name} already exists")
                return
                
            except ResponseHandlingException as e:
                if "timed out" in str(e) and attempt < self.max_retries - 1:
                    wait_time = (attempt + 1) * 10
                    logging.warning(f"Timeout on collection creation attempt {attempt + 1}. Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    logging.error(f"Failed to create collection after {self.max_retries} attempts: {e}")
                    raise
            except Exception as e:
                logging.error(f"Unexpected error creating collection: {e}")
                raise

    def add_documents_to_schema_details(self, documents, ids, collection_name=None):
        """Add documents with retry logic"""
        if collection_name is None:
            collection_name = self.collection_name
            
        for attempt in range(self.max_retries):
            try:
                self.client.add(
                    collection_name=collection_name,
                    documents=documents,
                    ids=ids
                )
                logging.info(f"Successfully added {len(documents)} documents to {collection_name}")
                return
            except ResponseHandlingException as e:
                if "timed out" in str(e) and attempt < self.max_retries - 1:
                    wait_time = (attempt + 1) * 10
                    logging.warning(f"Timeout adding documents attempt {attempt + 1}. Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    logging.error(f"Failed to add documents after {self.max_retries} attempts: {e}")
                    raise
            except Exception as e:
                logging.error(f"Unexpected error adding documents: {e}")
                raise

    def get_relevant_documents(self, text: str, collection_name: str = None, top_n_similar_docs=6):
        """Get relevant documents with retry logic"""
        if collection_name is None:
            collection_name = self.collection_name
            
        for attempt in range(self.max_retries):
            try:
                search_result = self.client.query(
                    collection_name=collection_name,
                    query_text=text,
                    limit=top_n_similar_docs,
                )
                metadata = [
                    {"id": hit.id, "document": hit.metadata.get('document', '')} 
                    for hit in search_result
                ]
                return metadata
            except ResponseHandlingException as e:
                if "timed out" in str(e) and attempt < self.max_retries - 1:
                    wait_time = (attempt + 1) * 5
                    logging.warning(f"Timeout querying documents attempt {attempt + 1}. Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    logging.error(f"Failed to query documents after {self.max_retries} attempts: {e}")
                    raise
            except Exception as e:
                logging.error(f"Unexpected error querying documents: {e}")
                raise

    def health_check(self):
        """Check if Qdrant is healthy"""
        try:
            collections = self.client.get_collections()
            return {"status": "healthy", "collections": len(collections.collections)}
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}