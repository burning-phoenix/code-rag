"""Helpers shared between chunker ``scan`` implementations."""

from ..models import IndexEntry


def add_term_entries(
    entries: list[IndexEntry],
    lines: list[str],
    file_name: str,
    abs_path: str,
    key_terms: list[str],
) -> None:
    """Append pointer entries for configured key terms found in the text.

    Each term is associated with the line of its first occurrence (plus a small
    surrounding window), unless an entry for that concept already exists.
    """
    full_text = "".join(lines).lower()

    for term in key_terms:
        if term.lower() in full_text:
            for i, line in enumerate(lines, start=1):
                if term.lower() in line.lower():
                    already_exists = any(
                        e.concept.lower() == term.lower() and e.file_name == file_name
                        for e in entries
                    )
                    if not already_exists:
                        entries.append(
                            IndexEntry(
                                concept=term,
                                file_name=file_name,
                                file_path=abs_path,
                                start_line=max(1, i - 2),
                                end_line=min(len(lines), i + 10),
                                entry_type="term",
                            )
                        )
                    break
