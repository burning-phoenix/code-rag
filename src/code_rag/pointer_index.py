"""
Static pointer index generator: scans markdown and code files and builds a
concept → file/line-range mapping for zero-hallucination lookups.
"""

import json
import logging
import re
from pathlib import Path
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)

@dataclass
class IndexEntry:
    """A pointer from a concept to its location in a document."""

    concept: str
    file_name: str
    file_path: str
    start_line: int
    end_line: int
    entry_type: str  # "section", "definition", "theorem", ..., "class", "function", "method"


# Patterns for formal math objects: Definition X.Y, Theorem X.Y, etc.
_FORMAL_PATTERN = re.compile(
    r"\*\*("
    r"Definition|Theorem|Proposition|Corollary|Lemma|Remark|Example"
    r")\s+([\d.]+)"
    r"(?:\*\*|\s)",
    re.IGNORECASE,
)

_LEAN_DECL_PATTERN = re.compile(
    r"^\s*"
    r"(?:(?:private|protected|noncomputable|unsafe|partial|scoped|local)\s+)*"
    r"(def|theorem|lemma|structure|class|inductive|abbrev|instance|axiom|example)\s+"
    r"([^\s(:{]+)"
)


def _strip_lean_comments(line: str, depth: int) -> tuple[str, int]:
    """Remove Lean line/block comments while tracking nested block depth."""
    out: list[str] = []
    i = 0

    while i < len(line):
        if depth == 0 and line.startswith("--", i):
            break
        if line.startswith("/-", i):
            depth += 1
            i += 2
            continue
        if depth > 0 and line.startswith("-/", i):
            depth -= 1
            i += 2
            continue
        if depth == 0:
            out.append(line[i])
        i += 1

    return "".join(out), depth


def _lean_entry_type(kind: str) -> str:
    """Map Lean declarations onto existing pointer-index entry types."""
    if kind == "def":
        return "definition"
    if kind in {"theorem", "lemma"}:
        return kind
    if kind in {"structure", "class", "inductive"}:
        return "class"
    if kind == "abbrev":
        return "definition"
    return kind


def _scan_file(file_path: Path, key_terms: list[str] | None = None) -> list[IndexEntry]:
    """Scan a single markdown file and extract index entries."""
    entries: list[IndexEntry] = []
    file_name = file_path.name
    abs_path = str(file_path.resolve())

    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # Track sections
    section_start = 1
    section_title = ""

    for i, line in enumerate(lines, start=1):
        stripped = line.strip()

        # Section headers
        header_match = re.match(r"^(#{1,6})\s+(.*)", stripped)
        if header_match:
            # Close previous section
            if section_title:
                entries.append(IndexEntry(
                    concept=section_title,
                    file_name=file_name,
                    file_path=abs_path,
                    start_line=section_start,
                    end_line=i - 1,
                    entry_type="section",
                ))

            section_title = header_match.group(2).strip()
            section_start = i
            continue

        # Formal definitions/theorems
        formal_match = _FORMAL_PATTERN.search(stripped)
        if formal_match:
            obj_type = formal_match.group(1).lower()
            obj_id = formal_match.group(2)
            # Find the end of this formal block (next empty line or next header)
            end = i
            for j in range(i, min(i + 50, len(lines))):
                next_line = lines[j].strip() if j < len(lines) else ""
                if re.match(r"^#{1,6}\s+", next_line):
                    end = j
                    break
                if next_line == "" and j > i + 1:
                    # Check if this is just a formula gap
                    if j + 1 < len(lines) and not lines[j + 1].strip().startswith("$$"):
                        end = j
                        break
                end = j + 1

            entries.append(IndexEntry(
                concept=f"{formal_match.group(1)} {obj_id}",
                file_name=file_name,
                file_path=abs_path,
                start_line=i,
                end_line=end,
                entry_type=obj_type,
            ))

    # Close final section
    if section_title:
        entries.append(IndexEntry(
            concept=section_title,
            file_name=file_name,
            file_path=abs_path,
            start_line=section_start,
            end_line=len(lines),
            entry_type="section",
        ))

    # Scan for key terms and associate with their containing section
    if key_terms:
        _add_term_entries(entries, lines, file_name, abs_path, key_terms)

    return entries


def _add_term_entries(
    entries: list[IndexEntry],
    lines: list[str],
    file_name: str,
    abs_path: str,
    key_terms: list[str],
) -> None:
    """Add entries for key terms found in the text."""
    full_text = "".join(lines).lower()

    for term in key_terms:
        if term.lower() in full_text:
            # Find the first occurrence and its line
            for i, line in enumerate(lines, start=1):
                if term.lower() in line.lower():
                    # Check we don't already have an entry covering this exact concept
                    already_exists = any(
                        e.concept.lower() == term.lower() and e.file_name == file_name
                        for e in entries
                    )
                    if not already_exists:
                        entries.append(IndexEntry(
                            concept=term,
                            file_name=file_name,
                            file_path=abs_path,
                            start_line=max(1, i - 2),
                            end_line=min(len(lines), i + 10),
                            entry_type="term",
                        ))
                    break


def _scan_code_file(file_path: Path, language: str) -> list[IndexEntry]:
    """Scan a code file and extract function/class definitions as index entries."""
    from .code_chunker import LANGUAGE_CONFIG, _get_parser, _node_name

    lang_config = LANGUAGE_CONFIG.get(language)
    if not lang_config:
        return []

    file_name = file_path.name
    abs_path = str(file_path.resolve())

    source = file_path.read_bytes()
    if not source.strip():
        return []

    parser = _get_parser(language)
    tree = parser.parse(source)
    root = tree.root_node

    function_types = set(lang_config["function_nodes"])
    class_types = set(lang_config["class_nodes"])
    method_types = set(lang_config.get("method_nodes", []))

    entries: list[IndexEntry] = []

    for child in root.children:
        start_line = child.start_point[0] + 1
        end_line = child.end_point[0] + 1

        if child.type in function_types:
            name = _node_name(child, language)
            entries.append(IndexEntry(
                concept=name,
                file_name=file_name,
                file_path=abs_path,
                start_line=start_line,
                end_line=end_line,
                entry_type="function",
            ))

        elif child.type in class_types:
            name = _node_name(child, language)
            entries.append(IndexEntry(
                concept=name,
                file_name=file_name,
                file_path=abs_path,
                start_line=start_line,
                end_line=end_line,
                entry_type="class",
            ))

            # Also index methods within the class
            if method_types:
                body = child.child_by_field_name(lang_config.get("body_field", "body"))
                children_to_scan = body.children if body else []
                # Rust impl_item: methods are in declaration_list
                if not children_to_scan:
                    for sub in child.children:
                        if sub.type == "declaration_list":
                            children_to_scan = sub.children
                            break

                for method_node in children_to_scan:
                    if method_node.type in method_types:
                        method_name = _node_name(method_node, language)
                        entries.append(IndexEntry(
                            concept=f"{name}.{method_name}",
                            file_name=file_name,
                            file_path=abs_path,
                            start_line=method_node.start_point[0] + 1,
                            end_line=method_node.end_point[0] + 1,
                            entry_type="method",
                        ))

    return entries


def _scan_lean_file(file_path: Path, key_terms: list[str] | None = None) -> list[IndexEntry]:
    """Scan a Lean file for top-level declarations and configured key terms."""
    entries: list[IndexEntry] = []
    file_name = file_path.name
    abs_path = str(file_path.resolve())

    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()

    declarations: list[tuple[int, str, str]] = []
    depth = 0

    for line_no, line in enumerate(lines, start=1):
        code, depth = _strip_lean_comments(line, depth)
        match = _LEAN_DECL_PATTERN.match(code)
        if match:
            declarations.append((line_no, match.group(1), match.group(2)))

    for idx, (start_line, kind, name) in enumerate(declarations):
        end_line = declarations[idx + 1][0] - 1 if idx + 1 < len(declarations) else len(lines)
        entries.append(IndexEntry(
            concept=name,
            file_name=file_name,
            file_path=abs_path,
            start_line=start_line,
            end_line=end_line,
            entry_type=_lean_entry_type(kind),
        ))

    if key_terms:
        _add_term_entries(entries, lines, file_name, abs_path, key_terms)

    return entries


def build_index(
    data_dir: str | Path,
    key_terms: list[str] | None = None,
    file_extensions: list[str] | None = None,
) -> list[dict]:
    """
    Build the pointer index for all files in a directory.
    Routes .md files to the markdown scanner and code files to the AST scanner.
    Returns a list of index entry dicts.
    """
    from .code_chunker import EXTENSION_TO_LANGUAGE

    data_dir = Path(data_dir)
    extensions = set(file_extensions or [".md"])
    all_entries: list[IndexEntry] = []

    for file_path in sorted(data_dir.rglob("*")):
        if not file_path.is_file():
            continue
        ext = file_path.suffix.lower()
        if ext not in extensions:
            continue

        try:
            if ext == ".md":
                all_entries.extend(_scan_file(file_path, key_terms=key_terms))
            elif ext == ".lean":
                all_entries.extend(_scan_lean_file(file_path, key_terms=key_terms))
            else:
                language = EXTENSION_TO_LANGUAGE.get(ext)
                if language:
                    all_entries.extend(_scan_code_file(file_path, language))
        except Exception as e:
            logger.error("Failed to scan %s: %s", file_path, e)

    return [asdict(e) for e in all_entries]


def save_index(
    data_dir: str | Path,
    output_path: str | Path,
    key_terms: list[str] | None = None,
    file_extensions: list[str] | None = None,
) -> int:
    """Build and save the pointer index to a JSON file. Returns entry count."""
    index = build_index(data_dir, key_terms=key_terms, file_extensions=file_extensions)
    output_path = Path(output_path)
    output_path.write_text(json.dumps(index, indent=2, ensure_ascii=False))
    return len(index)


def load_index(index_path: str | Path) -> list[dict]:
    """Load a previously saved pointer index."""
    return json.loads(Path(index_path).read_text())


def search_index(index: list[dict], query: str) -> list[dict]:
    """
    Search the pointer index for entries matching a query.
    Uses case-insensitive substring matching.
    """
    query_lower = query.lower()
    results = []

    for entry in index:
        concept = entry["concept"].lower()
        if query_lower in concept or concept in query_lower:
            results.append(entry)

    # Sort: exact matches first, then by entry_type priority
    type_priority = {
        "definition": 0, "theorem": 1, "proposition": 2,
        "corollary": 3, "lemma": 4, "section": 5, "term": 6,
        "class": 7, "function": 8, "method": 9,
    }

    results.sort(key=lambda e: (
        0 if e["concept"].lower() == query_lower else 1,
        type_priority.get(e["entry_type"], 99),
    ))

    return results
