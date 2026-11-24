import re

TOP_LEVEL_NODES_FOR_SUPPORTED_LANGUAGES = {
    ".py": {
        "function_definition",
        "class_definition",
        "decorated_definition",
    },
    ".js": {
        "function_declaration",
        "class_declaration",
        "lexical_declaration",
        "export_statement",
    },
    ".ts": {
        "function_declaration",
        "class_declaration",
        "interface_declaration",
        "type_alias_declaration",
        "enum_declaration",
        "lexical_declaration",
        "export_statement",
    },
    ".go": {
        "function_declaration",
        "method_declaration",
        "type_declaration",
        "const_declaration",
        "var_declaration",
    },
    ".java": {
        "class_declaration",
        "interface_declaration",
        "enum_declaration",
        "annotation_type_declaration",
        "method_declaration",
    },
    ".tsx": {
        "function_declaration",
        "class_declaration",
        "interface_declaration",
        "type_alias_declaration",
        "enum_declaration",
        "lexical_declaration",
        "export_statement",
    },
    ".kt": {
        "class_declaration",
        "object_declaration",
        "function_declaration",
    },
    ".rs": {
        "function_item",
        "struct_item",
        "enum_item",
        "trait_item",
        "impl_item",
    },
    ".cpp": {
        "function_definition",
        "class_specifier",
        "struct_specifier",
        "namespace_definition",
    },
    ".cc": {
        "function_definition",
        "class_specifier",
        "struct_specifier",
        "namespace_definition",
    },
    ".cxx": {
        "function_definition",
        "class_specifier",
        "struct_specifier",
        "namespace_definition",
    },
    ".c": {
        "function_definition",
        "struct_specifier",
        "enum_specifier",
    },
    ".cs": {
        "class_declaration",
        "struct_declaration",
        "interface_declaration",
        "enum_declaration",
        "method_declaration",
    },
    ".php": {
        "function_definition",
        "class_declaration",
        "interface_declaration",
        "trait_declaration",
    },
    ".rb": {
        "method",
        "class",
        "module",
    },
}


PATTERNS: dict[str, re.Pattern[str]] = {
    ".py": re.compile(r"^(async\s+def |def |class )"),
    ".js": re.compile(r"^(export\s+)?(async\s+)?(function|class|const|let|var)\b"),
    ".ts": re.compile(
        r"^(export\s+)?(async\s+)?(function|class|const|let|var|interface|type|enum)\b"
    ),
    ".tsx": re.compile(
        r"^(export\s+)?(async\s+)?(function|class|const|let|var|interface|type|enum)\b"
    ),
    ".go": re.compile(r"func\s+\w+|type\s+\w+|const\s+\w+|var\s+\w+"),
    ".java": re.compile(
        r"^((public|private|protected|static|final|abstract)\s+)*(class|interface|enum|@interface)\b"
    ),
    ".kt": re.compile(
        r"^((public|private|internal|protected|open|abstract|final|sealed|data)\s+)*(class|interface|object|fun)\b"
    ),
    ".rs": re.compile(
        r"^(pub(\([^)]*\))?\s+)?(async\s+)?(unsafe\s+)?(fn|struct|enum|trait|impl)\b"
    ),
    ".cs": re.compile(
        r"^((public|private|internal|protected)\s+)?(partial\s+|static\s+|abstract\s+|sealed\s+)*(class|struct|interface|enum|record)\b"
    ),
    ".php": re.compile(
        r"^(abstract\s+|final\s+)?(class|interface|trait)\b|^function\b"
    ),
    ".rb": re.compile(r"^(def|class|module)\b"),
}
