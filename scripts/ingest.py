#!/usr/bin/env python
import os
import ast
import json
import networkx as nx
from networkx.readwrite import json_graph
import chromadb
from sentence_transformers import SentenceTransformer
import sys

# --- Configuration ---
# Use a lightweight model suitable for code
EMBEDDING_MODEL = 'all-MiniLM-L6-v2'
# Directory to store the generated index
INDEX_DIR = os.path.join(os.path.dirname(__file__), '..', 'index')
VECTOR_DB_PATH = os.path.join(INDEX_DIR, 'chroma_db')
GRAPH_DB_PATH = os.path.join(INDEX_DIR, 'knowledge_graph.json')

# --- AST Parser for Entity and Relationship Extraction ---
class CodeParser(ast.NodeVisitor):
    def __init__(self, file_path):
        self.graph = nx.DiGraph()
        self.file_path = file_path
        self.stack = []

    def visit_FunctionDef(self, node):
        func_name = node.name
        node_id = f"{self.file_path}::{func_name}"
        self.graph.add_node(node_id, type='function', name=func_name, file=self.file_path)
        # Connect to parent (class or module)
        if self.stack:
            parent_id = self.stack[-1]
            self.graph.add_edge(parent_id, node_id, type='defines')
        self.stack.append(node_id)
        self.generic_visit(node)
        self.stack.pop()

    def visit_ClassDef(self, node):
        class_name = node.name
        node_id = f"{self.file_path}::{class_name}"
        self.graph.add_node(node_id, type='class', name=class_name, file=self.file_path)
        if self.stack:
            parent_id = self.stack[-1]
            self.graph.add_edge(parent_id, node_id, type='defines')
        self.stack.append(node_id)
        self.generic_visit(node)
        self.stack.pop()

    def visit_Call(self, node):
        if self.stack:
            caller_id = self.stack[-1]
            # Direct function call: foo()
            if isinstance(node.func, ast.Name):
                callee_name = node.func.id
                self.graph.add_edge(caller_id, callee_name, type='calls')
            # Method call: self.foo() or obj.foo()
            elif isinstance(node.func, ast.Attribute):
                callee_name = node.func.attr
                self.graph.add_edge(caller_id, callee_name, type='calls')
        self.generic_visit(node)

    def get_chunks(self, code):
        # Simple chunking by function and class
        chunks = []
        for node in ast.walk(ast.parse(code)):
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                chunk_code = ast.get_source_segment(code, node)
                chunk_id = f"{self.file_path}::{node.name}"
                chunks.append({'id': chunk_id, 'code': chunk_code})
        return chunks

# --- Main Ingestion Logic ---
def ingest_codebase(root_path):
    print(f"Starting ingestion of codebase at: {root_path}")
    os.makedirs(INDEX_DIR, exist_ok=True)

    # Initialize components
    model = SentenceTransformer(EMBEDDING_MODEL)
    chroma_client = chromadb.PersistentClient(path=VECTOR_DB_PATH)
    collection = chroma_client.get_or_create_collection(name="codebase_vectors")
    main_graph = nx.DiGraph()

    all_chunks = []

    # Walk through the codebase
    for subdir, _, files in os.walk(root_path):
        for file in files:
            if file.endswith('.py'): # Simple filter for Python files
                file_path = os.path.join(subdir, file)
                print(f"Processing: {file_path}")
                with open(file_path, 'r', encoding='utf-8') as f:
                    code = f.read()
                
                # Parse code to build graph and get chunks
                parser = CodeParser(file_path)
                try:
                    tree = ast.parse(code)
                    parser.visit(tree)
                    main_graph = nx.compose(main_graph, parser.graph)
                    chunks = parser.get_chunks(code)
                    all_chunks.extend(chunks)
                except Exception as e:
                    print(f"  - Could not parse {file_path}: {e}")

    # Batch embed and store in ChromaDB
    if all_chunks:
        print(f"Embedding {len(all_chunks)} code chunks...")
        codes = [chunk['code'] for chunk in all_chunks]
        ids = [chunk['id'] for chunk in all_chunks]
        embeddings = model.encode(codes, show_progress_bar=True)
        
        collection.add(
            embeddings=embeddings,
            documents=codes,
            ids=ids
        )
        print("Vector DB ingestion complete.")
    else:
        print("No code chunks found to ingest.")

    # Save the knowledge graph
    print("Saving knowledge graph...")
    graph_data = json_graph.node_link_data(main_graph)
    with open(GRAPH_DB_PATH, 'w') as f:
        json.dump(graph_data, f, indent=4)
    
    print("--- Ingestion Finished ---")
    print(f"Vector DB stored at: {VECTOR_DB_PATH}")
    print(f"Knowledge Graph stored at: {GRAPH_DB_PATH}")

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python ingest.py <path_to_codebase>")
        sys.exit(1)
    codebase_path = sys.argv[1]
    ingest_codebase(codebase_path)
