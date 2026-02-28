---
name: hybrid_rag
description: A hybrid RAG implementation combining vector search and knowledge graph traversal for codebases.

tools:
  - name: ingest_codebase
    description: Ingests a codebase from a local path, processing it into a hybrid vector/graph index.
    parameters:
      type: object
      properties:
        path:
          type: string
          description: The local path to the codebase to ingest.

  - name: retrieve
    description: Retrieves relevant context from the indexed codebase using a hybrid search query.
    parameters:
      type: object
      properties:
        query:
          type: string
          description: The natural language query for semantic search.
        graph_traversal:
          type: string
          description: (Optional) A query to traverse the knowledge graph (e.g., "find all functions called by 'process_data'").
---

# Hybrid RAG Skill

This skill provides a powerful Retrieval-Augmented Generation system for coding agents, combining the strengths of semantic vector search with the precision of knowledge graph traversal.

## How It Works

1.  **Ingestion (`ingest_codebase`):**
    *   The agent points the skill at a local codebase directory.
    *   The `ingest.py` script walks the directory, parsing all code files.
    *   For each file, it extracts key entities (functions, classes, variables, imports) and their relationships.
    *   Code chunks (e.g., function bodies) are embedded and stored in a ChromaDB vector store.
    *   Entities and relationships are stored in a NetworkX graph, saved as a JSON file.

2.  **Retrieval (`retrieve`):**
    *   The agent provides a natural language `query` for semantic search.
    *   Optionally, the agent can provide a `graph_traversal` query to explore relationships.
    *   The `retrieve.py` script first performs a vector search on the `query` to find semantically similar code chunks.
    *   If a `graph_traversal` query is provided, it executes it against the knowledge graph to find related entities.
    *   The results from both searches are combined, ranked, and returned to the agent as context.

## Usage

An agent equipped with this skill can answer questions like:

*   "Show me code examples similar to the `calculate_metrics` function."
*   "What functions does the `DatabaseConnection` class use?"
*   "Find the definition of the `USER_AUTHENTICATION_FAILED` error and show me where it is used."
