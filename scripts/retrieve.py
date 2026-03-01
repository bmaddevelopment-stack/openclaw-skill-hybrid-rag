#!/usr/bin/env python
"""
retrieve.py — Hybrid RAG Retrieval
Combines ChromaDB vector search (with optional metadata filters)
and NetworkX graph traversal into a single ranked result set.
"""
import json
import sys
import argparse
from pathlib import Path

import networkx as nx
from networkx.readwrite import json_graph
import chromadb
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
INDEX_DIR       = Path(__file__).parent.parent / "index"
VECTOR_DB_PATH  = str(INDEX_DIR / "chroma_db")
GRAPH_DB_PATH   = str(INDEX_DIR / "knowledge_graph.json")

# ---------------------------------------------------------------------------
# Graph query parser
# ---------------------------------------------------------------------------
def graph_query(G: nx.DiGraph, query: str) -> list[dict]:
    """
    Supported patterns:
      "X calls"              — successors of X (what X calls)
      "functions called by X"— predecessors of X (what calls X)
      "class X"              — all methods defined in class X
      "file X"               — all chunks from file X
    Returns list of node attribute dicts.
    """
    q = query.strip().lower()
    results = []

    def find_node(name: str):
        """Find node by name or qualified_name (case-insensitive)."""
        name_l = name.lower()
        for nid, data in G.nodes(data=True):
            if (data.get("name", "").lower() == name_l or
                    data.get("qualified", "").lower() == name_l):
                return nid
        return None

    def node_info(nid: str) -> dict:
        d = dict(G.nodes[nid])
        d["id"] = nid
        return d

    if "called by" in q:
        target = q.split("called by")[-1].strip().strip("'\"")
        nid = find_node(target)
        if nid:
            results = [node_info(p) for p in G.predecessors(nid)]
        else:
            results = [{"error": f"Node '{target}' not found"}]

    elif q.endswith("calls"):
        target = q[:-5].strip().strip("'\"")
        nid = find_node(target)
        if nid:
            results = [node_info(s) for s in G.successors(nid)]
        else:
            results = [{"error": f"Node '{target}' not found"}]

    elif q.startswith("class "):
        target = q[6:].strip().strip("'\"")
        nid = find_node(target)
        if nid:
            results = [node_info(s) for s in G.successors(nid)
                       if G.nodes[s].get("type") in ("method", "function")]
        else:
            results = [{"error": f"Class '{target}' not found"}]

    elif q.startswith("file "):
        target = q[5:].strip().strip("'\"")
        results = [node_info(n) for n, d in G.nodes(data=True)
                   if target in d.get("file", "").lower()]

    else:
        results = [{"error": (
            "Graph query not understood. Try:\n"
            "  'X calls'  |  'functions called by X'  |  'class X'  |  'file X'"
        )}]

    return results


# ---------------------------------------------------------------------------
# Main retrieval
# ---------------------------------------------------------------------------
def retrieve(
    query: str | None = None,
    graph_q: str | None = None,
    n_results: int = 5,
    filter_kind: str | None = None,   # e.g. "function", "class", "method"
    filter_file: str | None = None,   # partial file path match
    filter_lang: str | None = None,   # e.g. "python", "ts"
) -> dict:
    results = {"vector_search": [], "graph_search": []}

    # ---- Vector search ----
    if query:
        print(f"Vector search: {query}")
        model  = SentenceTransformer(EMBEDDING_MODEL)
        chroma = chromadb.PersistentClient(path=VECTOR_DB_PATH)
        try:
            col = chroma.get_collection("codebase_vectors")

            # Build optional metadata where-filter
            where_clauses = []
            if filter_kind:
                where_clauses.append({"chunk_kind": {"$eq": filter_kind}})
            if filter_lang:
                where_clauses.append({"language": {"$eq": filter_lang}})
            if filter_file:
                where_clauses.append({"file_path": {"$contains": filter_file}})

            where = None
            if len(where_clauses) == 1:
                where = where_clauses[0]
            elif len(where_clauses) > 1:
                where = {"$and": where_clauses}

            kwargs = dict(
                query_embeddings=model.encode([query]).tolist(),
                n_results=n_results,
                include=["documents", "metadatas", "distances"],
            )
            if where:
                kwargs["where"] = where

            raw = col.query(**kwargs)

            for doc, meta, dist in zip(
                raw["documents"][0],
                raw["metadatas"][0],
                raw["distances"][0],
            ):
                results["vector_search"].append({
                    "score":         round(1 - dist, 4),   # cosine similarity
                    "chunk_id":      meta.get("chunk_id"),
                    "chunk_kind":    meta.get("chunk_kind"),
                    "name":          meta.get("name"),
                    "qualified_name":meta.get("qualified_name"),
                    "file_path":     meta.get("file_path"),
                    "line_start":    meta.get("line_start"),
                    "line_end":      meta.get("line_end"),
                    "token_estimate":meta.get("token_estimate"),
                    "docstring":     meta.get("docstring"),
                    "language":      meta.get("language"),
                    "code":          doc,
                })
            print(f"  {len(results['vector_search'])} results (cosine similarity)")

        except Exception as e:
            print(f"  Vector search error: {e}")
            results["vector_search_error"] = str(e)

    # ---- Graph traversal ----
    if graph_q:
        print(f"Graph traversal: {graph_q}")
        try:
            with open(GRAPH_DB_PATH) as f:
                G = json_graph.node_link_graph(json.load(f))
            results["graph_search"] = graph_query(G, graph_q)
            print(f"  {len(results['graph_search'])} nodes")
        except Exception as e:
            print(f"  Graph error: {e}")
            results["graph_search_error"] = str(e)

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Hybrid RAG Retrieval")
    p.add_argument("-q",  "--query",       help="Semantic search query")
    p.add_argument("-g",  "--graph_query", help="Graph traversal query")
    p.add_argument("-n",  "--n_results",   type=int, default=5)
    p.add_argument("--kind", help="Filter by chunk kind: function|class|method|module_chunk")
    p.add_argument("--file", help="Filter by partial file path")
    p.add_argument("--lang", help="Filter by language: python|ts|js|tsx|jsx")
    args = p.parse_args()

    if not args.query and not args.graph_query:
        print("Error: provide --query and/or --graph_query")
        sys.exit(1)

    data = retrieve(
        query=args.query,
        graph_q=args.graph_query,
        n_results=args.n_results,
        filter_kind=args.kind,
        filter_file=args.file,
        filter_lang=args.lang,
    )
    print("\n--- Results ---")
    print(json.dumps(data, indent=2))
