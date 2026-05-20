"""
AST-aware code chunker: uses tree-sitter to split code files at function/class
boundaries while preserving logical structure and parent-child relationships.

Supports: Python, JavaScript, TypeScript, GLSL, Rust, Go, Java, C, C++, Ruby, Swift.
"""

import logging
import uuid
from pathlib import Path

from tree_sitter_language_pack import get_parser

from .chunker import Chunk

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Language configuration: maps each language to its tree-sitter node types
# ---------------------------------------------------------------------------

LANGUAGE_CONFIG = {
    "python": {
        "extensions": [".py"],
        "function_nodes": ["function_definition"],
        "class_nodes": ["class_definition"],
        "method_nodes": ["function_definition"],
        "body_field": "body",
        "name_field": "name",
    },
    "javascript": {
        "extensions": [".js", ".jsx", ".mjs"],
        "function_nodes": ["function_declaration"],
        "class_nodes": ["class_declaration"],
        "method_nodes": ["method_definition"],
        "body_field": "body",
        "name_field": "name",
    },
    "typescript": {
        "extensions": [".ts", ".tsx"],
        "function_nodes": ["function_declaration"],
        "class_nodes": ["class_declaration"],
        "method_nodes": ["method_definition"],
        "body_field": "body",
        "name_field": "name",
    },
    "glsl": {
        "extensions": [".glsl", ".vert", ".frag", ".comp"],
        "function_nodes": ["function_definition"],
        "class_nodes": [],
        "method_nodes": [],
        "body_field": "body",
        "name_field": "name",
    },
    "rust": {
        "extensions": [".rs"],
        "function_nodes": ["function_item"],
        "class_nodes": ["struct_item", "enum_item", "impl_item"],
        "method_nodes": ["function_item"],
        "body_field": "body",  # impl uses "declaration_list" children directly
        "name_field": "name",
    },
    "go": {
        "extensions": [".go"],
        "function_nodes": ["function_declaration", "method_declaration"],
        "class_nodes": ["type_declaration"],
        "method_nodes": [],
        "body_field": "body",
        "name_field": "name",
    },
    "java": {
        "extensions": [".java"],
        "function_nodes": [],
        "class_nodes": ["class_declaration", "interface_declaration", "enum_declaration"],
        "method_nodes": ["method_declaration", "constructor_declaration"],
        "body_field": "body",
        "name_field": "name",
    },
    "c": {
        "extensions": [".c", ".h"],
        "function_nodes": ["function_definition"],
        "class_nodes": ["struct_specifier"],
        "method_nodes": [],
        "body_field": "body",
        "name_field": "name",
    },
    "cpp": {
        "extensions": [".cpp", ".cc", ".cxx", ".hpp"],
        "function_nodes": ["function_definition"],
        "class_nodes": ["class_specifier", "struct_specifier"],
        "method_nodes": ["function_definition"],
        "body_field": "body",
        "name_field": "name",
    },
    "ruby": {
        "extensions": [".rb"],
        "function_nodes": ["method"],
        "class_nodes": ["class", "module"],
        "method_nodes": ["method"],
        "body_field": "body",
        "name_field": "name",
    },
    "swift": {
        "extensions": [".swift"],
        "function_nodes": ["function_declaration"],
        "class_nodes": ["class_declaration", "struct_declaration", "protocol_declaration"],
        "method_nodes": ["function_declaration"],
        "body_field": "body",
        "name_field": "name",
    },
}

# Reverse map: file extension → language name
EXTENSION_TO_LANGUAGE: dict[str, str] = {}
for _lang, _cfg in LANGUAGE_CONFIG.items():
    for _ext in _cfg["extensions"]:
        EXTENSION_TO_LANGUAGE[_ext] = _lang

# Cache parsers to avoid re-creation
_parser_cache: dict[str, object] = {}


def _get_parser(language: str):
    """Get or create a cached tree-sitter parser for the given language."""
    if language not in _parser_cache:
        _parser_cache[language] = get_parser(language)
    return _parser_cache[language]


def _node_text(source: bytes, node) -> str:
    """Extract the text content of a tree-sitter node."""
    return source[node.start_byte:node.end_byte].decode("utf-8", errors="replace")


def _node_lines(node) -> int:
    """Number of lines spanned by a node (1-indexed inclusive)."""
    return node.end_point[0] - node.start_point[0] + 1


def _node_name(node, language: str) -> str:
    """Extract the name identifier from a function/class node."""
    # Try the standard 'name' field first
    name_node = node.child_by_field_name("name")
    if name_node:
        return name_node.text.decode("utf-8", errors="replace")

    # Rust impl_item: name is the type_identifier child
    if node.type == "impl_item":
        for child in node.children:
            if child.type == "type_identifier":
                return child.text.decode("utf-8", errors="replace")

    # C/C++ function_definition: name is inside the declarator
    if node.type == "function_definition":
        declarator = node.child_by_field_name("declarator")
        if declarator:
            # Could be function_declarator or pointer_declarator
            name_node = declarator.child_by_field_name("declarator")
            if name_node:
                return name_node.text.decode("utf-8", errors="replace")
            # Or the declarator itself might be the identifier
            if declarator.type == "identifier":
                return declarator.text.decode("utf-8", errors="replace")

    # Fallback: first identifier child
    for child in node.children:
        if child.type in ("identifier", "type_identifier", "constant"):
            return child.text.decode("utf-8", errors="replace")

    return "<anonymous>"


def _get_class_methods(node, lang_config: dict) -> list:
    """Extract method nodes from a class/struct/impl body."""
    method_types = set(lang_config.get("method_nodes", []))
    if not method_types:
        return []

    methods = []

    # Try the body field first
    body = node.child_by_field_name(lang_config.get("body_field", "body"))
    if body:
        for child in body.children:
            if child.type in method_types:
                methods.append(child)
        return methods

    # Rust impl_item: methods are inside declaration_list child
    for child in node.children:
        if child.type == "declaration_list":
            for m in child.children:
                if m.type in method_types:
                    methods.append(m)
            return methods

    return methods


def chunk_code_file(
    file_path: str | Path,
    language: str,
    min_chunk_lines: int = 5,
    max_chunk_lines: int = 150,
) -> list[Chunk]:
    """
    Chunk a code file by AST structure (functions, classes, methods).

    - Top-level functions → individual chunks
    - Classes small enough → single chunk
    - Large classes → parent chunk (full class) + child chunks (per method)
    - Module preamble (imports, constants) → one chunk
    """
    file_path = Path(file_path)
    file_name = file_path.name
    lang_config = LANGUAGE_CONFIG.get(language)

    if not lang_config:
        logger.warning("No language config for '%s', skipping %s", language, file_name)
        return []

    source = file_path.read_bytes()
    if not source.strip():
        return []

    parser = _get_parser(language)
    tree = parser.parse(source)
    root = tree.root_node

    function_types = set(lang_config["function_nodes"])
    class_types = set(lang_config["class_nodes"])

    chunks: list[Chunk] = []
    preamble_nodes: list = []

    for child in root.children:
        if child.type in function_types:
            # Flush preamble before first function/class
            if preamble_nodes:
                chunks.extend(_flush_preamble(preamble_nodes, source, file_name, language))
                preamble_nodes = []

            chunks.append(_make_function_chunk(child, source, file_name, language))

        elif child.type in class_types:
            if preamble_nodes:
                chunks.extend(_flush_preamble(preamble_nodes, source, file_name, language))
                preamble_nodes = []

            chunks.extend(
                _chunk_class_node(child, source, file_name, language, lang_config, max_chunk_lines)
            )

        else:
            # Non-function/class: accumulate as preamble
            preamble_nodes.append(child)

    # Flush remaining preamble
    if preamble_nodes:
        chunks.extend(_flush_preamble(preamble_nodes, source, file_name, language))

    # Filter out chunks that are too small and merge them
    chunks = _merge_tiny_chunks(chunks, min_chunk_lines)

    return chunks


def _flush_preamble(nodes: list, source: bytes, file_name: str, language: str) -> list[Chunk]:
    """Create a module preamble chunk from accumulated non-function/class nodes."""
    if not nodes:
        return []

    # Combine all preamble nodes into one text block
    start_line = nodes[0].start_point[0] + 1  # 1-indexed
    end_line = nodes[-1].end_point[0] + 1
    text = source[nodes[0].start_byte:nodes[-1].end_byte].decode("utf-8", errors="replace").strip()

    if not text:
        return []

    return [Chunk(
        text=text,
        file_name=file_name,
        section_title=f"{file_name} (module preamble)",
        start_line=start_line,
        end_line=end_line,
        chunk_id=str(uuid.uuid4()),
        chunk_type="code",
        language=language,
        symbol_name=f"{file_name} preamble",
        symbol_type="module_preamble",
    )]


def _make_function_chunk(node, source: bytes, file_name: str, language: str) -> Chunk:
    """Create a chunk from a top-level function node."""
    name = _node_name(node, language)
    text = _node_text(source, node).strip()
    return Chunk(
        text=text,
        file_name=file_name,
        section_title=name,
        start_line=node.start_point[0] + 1,
        end_line=node.end_point[0] + 1,
        chunk_id=str(uuid.uuid4()),
        chunk_type="code",
        language=language,
        symbol_name=name,
        symbol_type="function",
    )


def _chunk_class_node(
    node, source: bytes, file_name: str, language: str,
    lang_config: dict, max_chunk_lines: int,
) -> list[Chunk]:
    """Chunk a class/struct/impl node, splitting into parent+children if too large."""
    name = _node_name(node, language)
    full_text = _node_text(source, node).strip()
    total_lines = _node_lines(node)
    class_id = str(uuid.uuid4())

    start_line = node.start_point[0] + 1
    end_line = node.end_point[0] + 1

    # Determine the right symbol_type based on node type
    symbol_type = "class"
    if node.type in ("struct_item", "struct_specifier", "struct_declaration"):
        symbol_type = "class"  # treat structs like classes for RAG purposes
    elif node.type == "impl_item":
        symbol_type = "class"  # impl blocks are method containers

    if total_lines <= max_chunk_lines:
        # Small enough — one chunk for the whole class
        return [Chunk(
            text=full_text,
            file_name=file_name,
            section_title=name,
            start_line=start_line,
            end_line=end_line,
            chunk_id=class_id,
            chunk_type="code",
            language=language,
            symbol_name=name,
            symbol_type=symbol_type,
        )]

    # Too large — create parent chunk + child chunks per method
    chunks = []

    # Parent chunk: full class text (for context retrieval)
    chunks.append(Chunk(
        text=full_text,
        file_name=file_name,
        section_title=name,
        start_line=start_line,
        end_line=end_line,
        chunk_id=class_id,
        chunk_type="code",
        language=language,
        symbol_name=name,
        symbol_type=symbol_type,
    ))

    # Child chunks: one per method
    methods = _get_class_methods(node, lang_config)
    for method_node in methods:
        method_name = _node_name(method_node, language)
        method_text = _node_text(source, method_node).strip()

        chunks.append(Chunk(
            text=method_text,
            file_name=file_name,
            section_title=f"{name}.{method_name}",
            start_line=method_node.start_point[0] + 1,
            end_line=method_node.end_point[0] + 1,
            chunk_id=str(uuid.uuid4()),
            chunk_type="code",
            language=language,
            symbol_name=f"{name}.{method_name}",
            symbol_type="method",
            parent_id=class_id,
        ))

    return chunks


def _merge_tiny_chunks(chunks: list[Chunk], min_lines: int) -> list[Chunk]:
    """Merge chunks that are too small into adjacent chunks."""
    if not chunks or min_lines <= 0:
        return chunks

    result: list[Chunk] = []

    for chunk in chunks:
        num_lines = chunk.end_line - chunk.start_line + 1
        if result and num_lines < min_lines and result[-1].language == chunk.language:
            # Merge into previous chunk
            prev = result[-1]
            merged_text = prev.text + "\n\n" + chunk.text
            result[-1] = Chunk(
                text=merged_text.strip(),
                file_name=chunk.file_name,
                section_title=prev.section_title,
                start_line=prev.start_line,
                end_line=chunk.end_line,
                chunk_id=prev.chunk_id,
                chunk_type="code",
                language=chunk.language,
                symbol_name=prev.symbol_name,
                symbol_type=prev.symbol_type,
                parent_id=prev.parent_id,
            )
        else:
            result.append(chunk)

    return result
