#!/usr/bin/env python
"""
ingest.py — Hybrid RAG Ingestion
Walks a codebase, extracts structured chunks with rich metadata,
embeds them into ChromaDB, and builds a NetworkX knowledge graph.

Supported file types: .py, .js, .ts, .jsx, .tsx
"""
import os
import ast
import json
import hashlib
import textwrap
import sys
import re
from pathlib import Path
from datetime import datetime

import networkx as nx
from networkx.readwrite import json_graph
import chromadb
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
INDEX_DIR = Path(__file__).parent.parent / "index"
VECTOR_DB_PATH = str(INDEX_DIR / "chroma_db")
GRAPH_DB_PATH = str(INDEX_DIR / "knowledge_graph.json")

# Chunking knobs
MAX_CHUNK_TOKENS = 400      # ~300 words; keeps embeddings focused
OVERLAP_LINES    = 3        # lines of overlap between adjacent file-level chunks
SUPPORTED_EXTS   = {".py", ".js", ".ts", ".jsx", ".tsx"}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def stable_id(file_path: str, name: str, kind: str) -> str:
    """Deterministic chunk ID: sha1 of (file, name, kind)."""
    raw = f"{file_path}::{kind}::{name}"
    return hashlib.sha1(raw.encode()).hexdigest()[:16]

def count_tokens(text: str) -> int:
    """Rough token estimate (1 token ≈ 4 chars)."""
    return max(1, len(text) // 4)

def split_into_line_chunks(lines: list[str], max_tokens: int, overlap: int) -> list[list[str]]:
    """
    Greedy line-based chunker with overlap.
    Returns list of line-groups, each under max_tokens.
    """
    chunks, current, current_tokens = [], [], 0
    for line in lines:
        lt = count_tokens(line)
        if current_tokens + lt > max_tokens and current:
            chunks.append(current)
            current = current[-overlap:] if overlap else []
            current_tokens = sum(count_tokens(l) for l in current)
        current.append(line)
        current_tokens += lt
    if current:
        chunks.append(current)
    return chunks

def docstring_of(node) -> str:
    """Extract the first docstring from a function/class node."""
    try:
        first = node.body[0]
        if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant):
            return first.value.s.strip()
    except (IndexError, AttributeError):
        pass
    return ""

def base_metadata(file_path: str, rel_path: str) -> dict:
    """Fields shared by every chunk."""
    stat = os.stat(file_path)
    return {
        "file_path":    rel_path,
        "file_name":    Path(file_path).name,
        "extension":    Path(file_path).suffix,
        "file_size_b":  stat.st_size,
        "ingested_at":  datetime.utcnow().isoformat(),
    }

# ---------------------------------------------------------------------------
# Python AST Parser
# ---------------------------------------------------------------------------
class PythonParser:
    def __init__(self, file_path: str, rel_path: str, source: str):
        self.file_path = file_path
        self.rel_path  = rel_path
        self.source    = source
        self.lines     = source.splitlines()
        self.graph     = nx.DiGraph()
        self._stack: list[str] = []   # node-id stack for nesting

    # ---- graph helpers ----
    def _add_node(self, node_id, **attrs):
        self.graph.add_node(node_id, **attrs)

    def _add_edge(self, src, dst, edge_type):
        self.graph.add_edge(src, dst, type=edge_type)

    # ---- chunk builders ----
    def _chunk_from_ast_node(self, node, kind: str, parent_class: str = "") -> dict | None:
        src = ast.get_source_segment(self.source, node)
        if not src:
            return None

        name      = node.name
        chunk_id  = stable_id(self.rel_path, name, kind)
        doc       = docstring_of(node)
        lineno    = node.lineno
        end_lineno = getattr(node, "end_lineno", lineno)

        # Decorator names
        decorators = []
        for d in getattr(node, "decorator_list", []):
            if isinstance(d, ast.Name):
                decorators.append(d.id)
            elif isinstance(d, ast.Attribute):
                decorators.append(d.attr)

        # Argument names (functions only)
        args = []
        if kind == "function" and hasattr(node, "args"):
            args = [a.arg for a in node.args.args]

        # Return annotation
        return_annotation = ""
        if kind == "function" and node.returns:
            return_annotation = ast.unparse(node.returns)

        # Base classes (classes only)
        bases = []
        if kind == "class":
            bases = [ast.unparse(b) for b in node.bases]

        meta = {
            **base_metadata(self.file_path, self.rel_path),
            "chunk_id":          chunk_id,
            "chunk_kind":        kind,           # function | class | method | module_chunk
            "name":              name,
            "qualified_name":    f"{parent_class}.{name}" if parent_class else name,
            "parent_class":      parent_class,
            "line_start":        lineno,
            "line_end":          end_lineno,
            "line_count":        end_lineno - lineno + 1,
            "token_estimate":    count_tokens(src),
            "docstring":         doc,
            "decorators":        json.dumps(decorators),
            "args":              json.dumps(args),
            "return_annotation": return_annotation,
            "base_classes":      json.dumps(bases),
            "language":          "python",
        }
        return {"id": chunk_id, "text": src, "metadata": meta}

    # ---- AST walk ----
    def parse(self) -> tuple[list[dict], nx.DiGraph]:
        chunks = []
        try:
            tree = ast.parse(self.source)
        except SyntaxError:
            return chunks, self.graph

        # Module-level node in graph
        module_id = stable_id(self.rel_path, "__module__", "module")
        self._add_node(module_id, type="module", name=self.rel_path, file=self.rel_path)
        self._stack.append(module_id)

        # Walk top-level nodes
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.ClassDef):
                chunks.extend(self._visit_class(node, parent_id=module_id))
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                chunk = self._visit_function(node, parent_id=module_id, parent_class="")
                if chunk:
                    chunks.append(chunk)

        self._stack.pop()

        # Fall-back: if no AST chunks, do line-based chunking for the whole file
        if not chunks:
            chunks.extend(self._file_level_chunks())

        # Collect call edges from AST
        self._collect_calls(tree, module_id)

        return chunks, self.graph

    def _visit_class(self, node: ast.ClassDef, parent_id: str) -> list[dict]:
        chunks = []
        class_name = node.name
        class_id   = stable_id(self.rel_path, class_name, "class")

        self._add_node(class_id, type="class", name=class_name, file=self.rel_path,
                       bases=json.dumps([ast.unparse(b) for b in node.bases]))
        self._add_edge(parent_id, class_id, "defines")

        chunk = self._chunk_from_ast_node(node, "class")
        if chunk:
            chunks.append(chunk)

        self._stack.append(class_id)
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                c = self._visit_function(child, parent_id=class_id, parent_class=class_name)
                if c:
                    chunks.append(c)
        self._stack.pop()
        return chunks

    def _visit_function(self, node, parent_id: str, parent_class: str) -> dict | None:
        kind      = "method" if parent_class else "function"
        func_id   = stable_id(self.rel_path, f"{parent_class}.{node.name}" if parent_class else node.name, kind)

        self._add_node(func_id, type=kind, name=node.name,
                       qualified=f"{parent_class}.{node.name}" if parent_class else node.name,
                       file=self.rel_path)
        self._add_edge(parent_id, func_id, "defines")

        return self._chunk_from_ast_node(node, kind, parent_class)

    def _collect_calls(self, tree, module_id: str):
        """Second pass: add CALLS edges between resolved node IDs."""
        # Build name -> node_id lookup
        name_map = {}
        for nid, data in self.graph.nodes(data=True):
            name_map[data.get("name", "")] = nid
            name_map[data.get("qualified", "")] = nid

        class CallVisitor(ast.NodeVisitor):
            def __init__(self_, graph, stack_ref):
                self_.graph = graph
                self_.stack = stack_ref

            def visit_FunctionDef(self_, node):
                key = node.name
                nid = name_map.get(key)
                if nid:
                    self_.stack.append(nid)
                self_.generic_visit(node)
                if nid:
                    self_.stack.pop()

            visit_AsyncFunctionDef = visit_FunctionDef

            def visit_Call(self_, node):
                if not self_.stack:
                    self_.generic_visit(node)
                    return
                caller = self_.stack[-1]
                callee_name = None
                if isinstance(node.func, ast.Name):
                    callee_name = node.func.id
                elif isinstance(node.func, ast.Attribute):
                    callee_name = node.func.attr
                if callee_name:
                    callee_id = name_map.get(callee_name, callee_name)
                    self_.graph.add_edge(caller, callee_id, type="calls")
                self_.generic_visit(node)

        CallVisitor(self.graph, []).visit(tree)

    def _file_level_chunks(self) -> list[dict]:
        """Fallback: chunk the whole file by lines with overlap."""
        groups = split_into_line_chunks(self.lines, MAX_CHUNK_TOKENS, OVERLAP_LINES)
        chunks = []
        for i, group in enumerate(groups):
            text = "\n".join(group)
            cid  = stable_id(self.rel_path, f"chunk_{i}", "module_chunk")
            meta = {
                **base_metadata(self.file_path, self.rel_path),
                "chunk_id":       cid,
                "chunk_kind":     "module_chunk",
                "name":           f"{Path(self.rel_path).stem}_chunk_{i}",
                "qualified_name": f"{Path(self.rel_path).stem}_chunk_{i}",
                "parent_class":   "",
                "line_start":     self.lines.index(group[0]) + 1 if group[0] in self.lines else 0,
                "line_end":       0,
                "line_count":     len(group),
                "token_estimate": count_tokens(text),
                "docstring":      "",
                "decorators":     "[]",
                "args":           "[]",
                "return_annotation": "",
                "base_classes":   "[]",
                "language":       "python",
            }
            chunks.append({"id": cid, "text": text, "metadata": meta})
        return chunks


# ---------------------------------------------------------------------------
# Generic line-based parser (JS/TS/etc.)
# ---------------------------------------------------------------------------
class GenericParser:
    """
    Regex-based chunker for JS/TS files.
    Extracts function/class declarations; falls back to line chunks.
    """
    FUNC_RE  = re.compile(
        r"(?:export\s+)?(?:async\s+)?(?:function\s+(\w+)|(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s+)?\()"
    )
    CLASS_RE = re.compile(r"(?:export\s+)?class\s+(\w+)")

    def __init__(self, file_path: str, rel_path: str, source: str):
        self.file_path = file_path
        self.rel_path  = rel_path
        self.source    = source
        self.lines     = source.splitlines()
        self.graph     = nx.DiGraph()

    def parse(self) -> tuple[list[dict], nx.DiGraph]:
        chunks = []
        module_id = stable_id(self.rel_path, "__module__", "module")
        self.graph.add_node(module_id, type="module", name=self.rel_path, file=self.rel_path)

        # Simple: chunk by top-level function/class declarations
        groups = split_into_line_chunks(self.lines, MAX_CHUNK_TOKENS, OVERLAP_LINES)
        for i, group in enumerate(groups):
            text = "\n".join(group)
            # Try to find a name
            name = f"chunk_{i}"
            for line in group:
                m = self.FUNC_RE.search(line) or self.CLASS_RE.search(line)
                if m:
                    name = next(g for g in m.groups() if g)
                    break
            kind = "function" if self.FUNC_RE.search(text) else "module_chunk"
            cid  = stable_id(self.rel_path, name, kind)
            ext  = Path(self.file_path).suffix.lstrip(".")
            meta = {
                **base_metadata(self.file_path, self.rel_path),
                "chunk_id":          cid,
                "chunk_kind":        kind,
                "name":              name,
                "qualified_name":    name,
                "parent_class":      "",
                "line_start":        i * MAX_CHUNK_TOKENS,
                "line_end":          i * MAX_CHUNK_TOKENS + len(group),
                "line_count":        len(group),
                "token_estimate":    count_tokens(text),
                "docstring":         "",
                "decorators":        "[]",
                "args":              "[]",
                "return_annotation": "",
                "base_classes":      "[]",
                "language":          ext,
            }
            chunks.append({"id": cid, "text": text, "metadata": meta})
            self.graph.add_node(cid, type=kind, name=name, file=self.rel_path)
            self.graph.add_edge(module_id, cid, type="defines")

        return chunks, self.graph


# ---------------------------------------------------------------------------
# Main ingestion
# ---------------------------------------------------------------------------
def ingest_codebase(root_path: str):
    root = Path(root_path).resolve()
    print(f"Ingesting: {root}")
    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    model        = SentenceTransformer(EMBEDDING_MODEL)
    chroma       = chromadb.PersistentClient(path=VECTOR_DB_PATH)
    collection   = chroma.get_or_create_collection(
        name="codebase_vectors",
        metadata={"hnsw:space": "cosine"}
    )
    main_graph   = nx.DiGraph()
    all_chunks: list[dict] = []

    for fpath in root.rglob("*"):
        if fpath.suffix not in SUPPORTED_EXTS:
            continue
        rel = str(fpath.relative_to(root))
        print(f"  {rel}")
        try:
            source = fpath.read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            print(f"    skip (read error): {e}")
            continue

        if fpath.suffix == ".py":
            parser = PythonParser(str(fpath), rel, source)
        else:
            parser = GenericParser(str(fpath), rel, source)

        chunks, graph = parser.parse()
        main_graph = nx.compose(main_graph, graph)
        all_chunks.extend(chunks)

    if not all_chunks:
        print("No chunks found.")
        return

    # Deduplicate by chunk_id
    seen, unique = set(), []
    for c in all_chunks:
        if c["id"] not in seen:
            seen.add(c["id"])
            unique.append(c)

    print(f"\nEmbedding {len(unique)} chunks...")
    texts     = [c["text"] for c in unique]
    ids       = [c["id"]   for c in unique]
    metas     = [c["metadata"] for c in unique]
    embeddings = model.encode(texts, batch_size=32, show_progress_bar=True)

    # Upsert in batches of 500
    BATCH = 500
    for i in range(0, len(unique), BATCH):
        collection.upsert(
            embeddings=embeddings[i:i+BATCH].tolist(),
            documents=texts[i:i+BATCH],
            ids=ids[i:i+BATCH],
            metadatas=metas[i:i+BATCH],
        )

    print("Vector DB done.")

    # Save graph
    graph_data = json_graph.node_link_data(main_graph)
    with open(GRAPH_DB_PATH, "w") as f:
        json.dump(graph_data, f, indent=2)

    print(f"Graph saved: {main_graph.number_of_nodes()} nodes, {main_graph.number_of_edges()} edges")
    print(f"Vector DB : {VECTOR_DB_PATH}")
    print(f"Graph DB  : {GRAPH_DB_PATH}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python ingest.py <path_to_codebase>")
        sys.exit(1)
    ingest_codebase(sys.argv[1])
