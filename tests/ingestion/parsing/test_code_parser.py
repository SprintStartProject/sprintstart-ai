from ingestion.parsing.code_parser import (
    PATTERNS,
    compute_boundaries,
    fallback_regex_parser,
)
from ingestion.parsing.parser import parse


def test_parse_python_multiple_functions():
    code = b"""
import os

def foo():
    return 1

def bar():
    return 2
"""

    result = parse("test.py", code)

    assert len(result) >= 2

    contents = " ".join(chunk.content for chunk in result)
    assert "foo" in contents
    assert "bar" in contents


def test_preamble_is_present_in_each_chunk():
    code = b"""
import os
import sys

def foo():
    return 1

def bar():
    return 2
"""

    result = parse("test.py", code)

    for chunk in result:
        assert "import os" in chunk.content
        assert "import sys" in chunk.content


def test_fallback_when_no_definitions():
    code = b"""
print("hello")
x = 1 + 1
"""

    result = parse("test.py", code)

    assert all(chunk.kind == "text" for chunk in result)


def test_parse_c_function():
    code = b"""
int add(int a, int b) {
    return a + b;
}
"""

    result = parse("test.c", code)

    assert len(result) >= 1
    assert all(chunk.kind == "code" for chunk in result)
    assert any("add" in chunk.content for chunk in result)


def test_class_is_detected():
    code = b"""
    import os

    class User:
        def __init__(self):
            pass
    """

    result = parse("test.py", code)

    assert any("class User" in chunk.content for chunk in result)
    assert any(chunk.metadata["symbol_name"] == "User" for chunk in result)


def test_chunk_order_is_preserved():
    code = b"""
import os

def a():
    pass

def b():
    pass
"""

    result = parse("test.py", code)

    indices = [chunk.metadata["chunk_index"] for chunk in result]

    assert indices == sorted(indices)


def test_no_empty_chunks_are_generated():
    code = b"""
def foo():

def bar():
    pass
"""

    result = parse("test.py", code)

    for chunk in result:
        assert chunk.content.strip() != ""


def test_compute_boundaries_detects_functions():
    lines = [
        "import os",
        "def foo():",
        "    pass",
        "def bar():",
        "    pass",
    ]

    pattern = PATTERNS[".py"]

    boundaries = compute_boundaries(lines, pattern)

    assert boundaries == [1, 3]


def test_parse_javascript_arrow_functions_and_classes():
    code = b"""
import React from "react";

const add = (a, b) => a + b;

class User {
    constructor(name) {
        this.name = name;
    }
}

"""

    result = parse("test.js", code)

    assert len(result) >= 2
    assert any("add" in chunk.content for chunk in result)
    assert any("User" in chunk.content for chunk in result)


def test_parse_typescript_functions_and_classes():
    code = b"""
export const sum = (a: number, b: number) => a + b;

export class User {
    name: string;
    constructor(name: string) {
        this.name = name;
    }
}
"""

    result = parse("test.ts", code)

    assert len(result) >= 2
    assert any("sum" in chunk.content for chunk in result)
    assert any("class User" in chunk.content for chunk in result)


def test_parse_go_functions():
    code = b"""
package main

import "fmt"

func add(a int, b int) int {
    return a + b
}

func main() {
    fmt.Println(add(1, 2))
}
"""

    result = parse("test.go", code)

    assert len(result) >= 2
    assert any("add" in chunk.content for chunk in result)
    assert any("main" in chunk.content for chunk in result)


def test_tree_sitter_adds_symbol_metadata():
    code = b"""
def hello():
    return "world"
    """

    result = parse("test.py", code)

    chunk = result[0]

    assert chunk.metadata["symbol_name"] == "hello"
    assert chunk.metadata["symbol_kind"] == "function_definition"


def test_java_class_is_detected_as_top_level_symbol():
    code = b"""
public class UserService {

    public void createUser() {
    }

    public void deleteUser() {
    }
}
"""

    result = parse("UserService.java", code)

    assert len(result) >= 1

    assert any("UserService" in chunk.content for chunk in result)

    assert any(chunk.metadata["symbol_kind"] == "class_declaration" for chunk in result)


def test_regex_fallback_parser():
    code = b"""
def foo():
    pass

def bar():
    pass
"""

    result = fallback_regex_parser("test.py", code)

    assert len(result) >= 2
    assert any("foo" in chunk.content for chunk in result)
    assert any("bar" in chunk.content for chunk in result)


def test_javascript_function_metadata():

    code = b"""
function hello() {
    return 1;
}
"""

    result = parse("test.js", code)

    assert result[0].metadata["symbol_kind"] == "function_declaration"


def test_go_function_metadata():

    code = b"""
package main

func hello() {

}
"""

    result = parse("test.go", code)

    assert result[0].metadata["symbol_kind"] == "function_declaration"


def test_non_ascii_source_is_not_corrupted():
    # tree-sitter reports byte offsets; a multi-byte char before the symbol used
    # to shift str-based slicing and garble the extracted name + content.
    code = "# café ☕ module header\n\ndef greet():\n    return 'hi'\n".encode()

    result = parse("test.py", code)

    contents = " ".join(chunk.content for chunk in result)
    assert "def greet():" in contents
    assert "return 'hi'" in contents
    assert any(chunk.metadata.get("symbol_name") == "greet" for chunk in result)


def test_trailing_module_level_code_is_kept():
    # Top-level statements after the first symbol must stay searchable.
    code = b"""
def main():
    return 1


SENTINEL_CONSTANT = 42

if __name__ == "__main__":
    main()
"""

    result = parse("test.py", code)

    contents = " ".join(chunk.content for chunk in result)
    assert "SENTINEL_CONSTANT = 42" in contents
    assert '__name__ == "__main__"' in contents


def test_duplicate_symbols_get_unique_chunk_positions():
    # Two byte-identical definitions must not collide on the deterministic chunk
    # id (which is derived from content + chunk_index).
    code = b"""
def helper():
    return 1


def helper_copy():
    return 1
"""

    result = parse("test.py", code)

    indices = [chunk.metadata["chunk_index"] for chunk in result]
    assert len(indices) == len(set(indices))


def test_javascript_exported_function_is_a_symbol():
    code = b"""
export function multiply(a, b) {
    return a * b;
}
"""

    result = parse("test.js", code)

    assert any("multiply" in chunk.content for chunk in result)
    assert any(chunk.metadata["symbol_kind"] == "export_statement" for chunk in result)


def test_tree_sitter_chunk_start_line_points_at_symbol_not_preamble():
    code = b"""import os

def foo():
    return 1

def bar():
    return 2
"""

    result = parse("test.py", code)

    by_symbol = {chunk.metadata.get("symbol_name"): chunk for chunk in result}

    # "def foo():" is on line 3, "def bar():" on line 6 - the preamble
    # ("import os") is prefixed onto both chunks but must not shift either
    # start_line back to line 1.
    assert by_symbol["foo"].metadata["start_line"] == "3"
    assert by_symbol["bar"].metadata["start_line"] == "6"


def test_regex_fallback_chunk_start_line_points_at_definition():
    code = b"""import os

def foo():
    pass

def bar():
    pass
"""

    result = fallback_regex_parser("test.py", code)

    foo_chunk = next(c for c in result if "def foo" in c.content)
    bar_chunk = next(c for c in result if "def bar" in c.content)

    assert foo_chunk.metadata["start_line"] == "3"
    assert bar_chunk.metadata["start_line"] == "6"
