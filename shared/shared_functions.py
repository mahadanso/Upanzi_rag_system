import chromadb
from chromadb.utils import embedding_functions
import json
import re
import numpy as np
from typing import List, Dict, Any, Optional

# Initialize ChromaDB client
chroma_client = chromadb.HttpClient(host='localhost', port=8000)

def load_json_data(file_path: str) -> List[Dict]:
    """Load data from JSON file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            json_data = json.load(file)

        subsections = []

        # Ensure each item has required fields and normalize the structure
        for i, item in enumerate(json_data):
            # Normalize food_id to string
            if 'doc_id' not in item:
                item['doc_id'] = str(i + 1)
            else:
                item['doc_id'] = str(item['doc_id'])
            
            # Ensure required fields exist
            if 'section' not in item:
                item['section'] = ''
            if 'subsections' not in item:
                item['subsections'] = []
            if 'content' not in item:
                item['content'] = ''
            
            # Extract nested subsections if available
            for idx, sub_section in enumerate(item.get('subsections', [])):
                sub_section['doc_id'] = str(sub_section.get('doc_id', idx))
                sub_section['doc_id'] = str(item['doc_id']+'_'+sub_section['doc_id'])
                sub_section['section'] = sub_section.get('section', '')
                sub_section['content'] = sub_section.get('content', '')
                for sidx, sub_sub_section in enumerate(sub_section.get('subsections', [])):
                    sub_sub_section['doc_id'] = str(sub_sub_section.get('doc_id', sidx))
                    sub_sub_section['doc_id'] = str(sub_section['doc_id']+'_'+sub_sub_section['doc_id'])
                    sub_sub_section['section'] = sub_sub_section.get('section', '')
                    sub_sub_section['content'] = sub_sub_section.get('content', '')
                    subsections.append(sub_sub_section)
                del sub_section['subsections']
                subsections.append(sub_section)

            del item['subsections']

        result = json_data + subsections

        print(f"Successfully loaded {len(result)} items from {file_path}")
        return result

    except Exception as e:
        print(f"Error loading json data: {e}")
        return []

def create_similarity_search_collection(collection_name: str, collection_metadata: dict = None):
    """Create ChromaDB collection with sentence transformer embeddings"""
    try:
        # Try to delete existing collection to start fresh
        chroma_client.delete_collection(collection_name)
    except:
        pass
    
    # Create embedding function
    sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    
    # Create new collection
    return chroma_client.create_collection(
        name=collection_name,
        metadata=collection_metadata,
        configuration={
            "hnsw": {"space": "cosine"},
            "embedding_function": sentence_transformer_ef
        }
    )

def get_similarity_search_collection(collection_name: str):
    """Retrieve ChromaDB collection"""
    try:
        return chroma_client.get_collection(name=collection_name)
    except Exception as e:
        print(f"Error retrieving collection {collection_name}: {e}")
        return None

def populate_similarity_collection(collection, data_items: List[Dict]):
    """Populate collection with data and generate embeddings"""
    documents = []
    metadatas = []
    ids = []
    
    # Create unique IDs to avoid duplicates
    used_ids = set()
    
    for i, data in enumerate(data_items):
        if data.get("content", '') == '':
            continue
        
        # Create comprehensive text for embedding using rich JSON structure
        text = f"{data['section']}: "
        text += f"{data.get('content', '')}. "
        
        # Generate unique ID to avoid duplicates
        base_id = str(data.get('doc_id', i))
        unique_id = base_id
        counter = 1
        while unique_id in used_ids:
            unique_id = f"{base_id}_{counter}"
            counter += 1
        used_ids.add(unique_id)
        
        documents.append(text)
        ids.append(unique_id)
        metadatas.append({
            "section": data["section"]
        })
    
    # Add all data to collection
    collection.add(
        documents=documents,
        metadatas=metadatas,
        ids=ids
    )
    
    print(f"Added {len(documents)} items to collection")

def perform_similarity_search(collection, query: str, n_results: int = 5) -> List[Dict]:
    """Perform similarity search and return formatted results"""
    try:
        results = collection.query(
            query_texts=[query],
            n_results=n_results
        )
        
        if not results or not results['ids'] or len(results['ids'][0]) == 0:
            return []
        
        formatted_results = []
        for i in range(len(results['ids'][0])):
            # Calculate similarity score (1 - distance)
            similarity_score = 1 - results['distances'][0][i]
            
            result = {
                'doc_id': results['ids'][0][i],
                'section': results['metadatas'][0][i]['section'],
                'content': results['documents'][0][i],
                'similarity_score': similarity_score,
                'distance': results['distances'][0][i]
            }
            formatted_results.append(result)
        
        return formatted_results
        
    except Exception as e:
        print(f"Error in similarity search: {e}")
        return []

def perform_filtered_similarity_search(collection, query: str, section_filter: str = None, 
                                     n_results: int = 5) -> List[Dict]:
    """Perform filtered similarity search with metadata constraints"""
    where_clause = None
    
    # Build filters list
    filters = []
    if section_filter:
        filters.append({"section": section_filter})
    
    # Construct where clause based on number of filters
    if len(filters) == 1:
        where_clause = filters[0]
    elif len(filters) > 1:
        where_clause = {"$and": filters}
    
    try:
        results = collection.query(
            query_texts=[query],
            n_results=n_results,
            where=where_clause
        )
        
        if not results or not results['ids'] or len(results['ids'][0]) == 0:
            return []
        
        formatted_results = []
        for i in range(len(results['ids'][0])):
            similarity_score = 1 - results['distances'][0][i]
            
            result = {
                'doc_id': results['ids'][0][i],
                'section': results['metadatas'][0][i]['section'],
                'content': results['documents'][0][i],
                'similarity_score': similarity_score,
                'distance': results['distances'][0][i]
            }
            formatted_results.append(result)
        
        return formatted_results
        
    except Exception as e:
        print(f"Error in filtered search: {e}")
        return []

def clear_collection(collection):
    """Clear all items from the collection"""
    try:
        collection.delete()
        print("Collection cleared successfully")
    except Exception as e:
        print(f"Error clearing collection: {e}")

def delete_collection(collection_name: str):
    """Delete the entire collection"""
    try:
        chroma_client.delete_collection(collection_name)
        print(f"Collection '{collection_name}' deleted successfully")
    except Exception as e:
        print(f"Error deleting collection '{collection_name}': {e}")

def list_collections() -> List[str]:
    """List all existing collections"""
    try:
        collections = chroma_client.list_collections()
        return [col.name for col in collections]
    except Exception as e:
        print(f"Error listing collections: {e}")
        return []

def get_collection_stats(collection) -> Dict[str, Any]:
    """Get statistics about the collection"""
    try:
        stats = collection.count()
        return stats
    except Exception as e:
        print(f"Error getting collection stats: {e}")
        return {}
