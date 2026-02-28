#!/usr/bin/env python
import json
import sys
import networkx as nx
from networkx.readwrite import json_graph
import chromadb
from sentence_transformers import SentenceTransformer
import os
import argparse

# --- Configuration ---
EMBEDDING_MODEL = 'all-MiniLM-L6-v2'
INDEX_DIR = os.path.join(os.path.dirname(__file__), '..', 'index')
VECTOR_DB_PATH = os.path.join(INDEX_DIR, 'chroma_db')
GRAPH_DB_PATH = os.path.join(INDEX_DIR, 'knowledge_graph.json')

def retrieve(query, graph_query=None, n_results=5):
    results = {'vector_search': [], 'graph_search': []}

    # --- Vector Search ---
    if query:
        print(f"Performing vector search for: {query}")
        model = SentenceTransformer(EMBEDDING_MODEL)
        chroma_client = chromadb.PersistentClient(path=VECTOR_DB_PATH)
        try:
            collection = chroma_client.get_collection(name="codebase_vectors")
            query_embedding = model.encode([query])
            vector_results = collection.query(
                query_embeddings=query_embedding,
                n_results=n_results
            )
            results['vector_search'] = vector_results['documents'][0]
            print(f"  Found {len(results['vector_search'])} semantic results.")
        except Exception as e:
            print(f"  Vector search error: {e}")
            results['vector_search_error'] = str(e)

    # --- Knowledge Graph Search ---
    if graph_query:
        print(f"Performing graph traversal for: {graph_query}")
        try:
            with open(GRAPH_DB_PATH, 'r') as f:
                graph_data = json.load(f)
            G = json_graph.node_link_graph(graph_data)

            parts = graph_query.lower().split()

            if 'called by' in graph_query.lower():
                # "functions called by X" — find predecessors of X
                target_name = parts[-1].strip("'\"")
                full_node_id = next(
                    (n for n, d in G.nodes(data=True) if d.get('name') == target_name),
                    None
                )
                if full_node_id:
                    predecessors = list(G.predecessors(full_node_id))
                    results['graph_search'] = [
                        {**G.nodes[p], 'id': p} for p in predecessors
                    ]
                    print(f"  Found {len(predecessors)} calling functions.")
                else:
                    print(f"  Node '{target_name}' not found in graph.")
                    results['graph_search_error'] = f"Node '{target_name}' not found."

            elif 'calls' in graph_query.lower():
                # "X calls" — find successors of X
                target_name = parts[0].strip("'\"")
                full_node_id = next(
                    (n for n, d in G.nodes(data=True) if d.get('name') == target_name),
                    None
                )
                if full_node_id:
                    successors = list(G.successors(full_node_id))
                    results['graph_search'] = [
                        {**G.nodes[s], 'id': s} for s in successors
                    ]
                    print(f"  Found {len(successors)} called functions/methods.")
                else:
                    print(f"  Node '{target_name}' not found in graph.")
                    results['graph_search_error'] = f"Node '{target_name}' not found."

            else:
                results['graph_search_error'] = (
                    "Graph query not understood. "
                    "Try 'functions called by X' or 'X calls'."
                )

        except Exception as e:
            print(f"  Graph search error: {e}")
            results['graph_search_error'] = str(e)

    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Hybrid RAG Retrieval")
    parser.add_argument('-q', '--query', type=str, help="Natural language query for vector search.")
    parser.add_argument('-g', '--graph_query', type=str, help="Graph traversal query.")
    args = parser.parse_args()

    if not args.query and not args.graph_query:
        print("Error: At least one of --query or --graph_query must be provided.")
        sys.exit(1)

    retrieved_data = retrieve(args.query, args.graph_query)
    print("\n--- Retrieval Results ---")
    print(json.dumps(retrieved_data, indent=2))
