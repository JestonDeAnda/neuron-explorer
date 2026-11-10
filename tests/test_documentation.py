from __future__ import annotations

import re
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit


ROOT = Path(__file__).parents[1]
USAGE_DOCS = (
    ROOT / "README.md",
    ROOT / "docs" / "explorer.md",
    ROOT / "docs" / "sweeps.md",
    ROOT / "docs" / "attribution.md",
    ROOT / "docs" / "interventions.md",
    ROOT / "docs" / "development.md",
)


class _LocalLinkParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.targets: list[str] = []

    def handle_starttag(
        self, tag: str, attrs: list[tuple[str, str | None]]
    ) -> None:
        relevant = {"href"} if tag in {"a", "link"} else {"src"}
        for name, value in attrs:
            if name.lower() in relevant and value:
                self.targets.append(value)


def _local_target(target: str) -> str | None:
    if target.startswith(("#", "//")):
        return None
    parsed = urlsplit(target)
    if parsed.scheme or not parsed.path or parsed.path.startswith("/"):
        return None
    return unquote(parsed.path)


def test_every_script_is_covered_by_usage_documentation() -> None:
    documented = "\n".join(
        path.read_text(encoding="utf-8") for path in USAGE_DOCS if path.is_file()
    )
    executables = sorted(
        path
        for path in (ROOT / "scripts").iterdir()
        if path.is_file() and path.suffix in {".py", ".sh"}
    )

    missing = [path.name for path in executables if path.name not in documented]

    assert not missing, f"undocumented scripts: {missing}"


def test_local_markdown_links_resolve() -> None:
    markdown_files = [ROOT / "README.md", *sorted((ROOT / "docs").rglob("*.md"))]
    missing: list[str] = []
    for document in markdown_files:
        text = document.read_text(encoding="utf-8")
        for target in re.findall(r"(?<!!)\[[^\]]+\]\(([^)]+)\)", text):
            target = target.split("#", 1)[0]
            if not target or "://" in target or target.startswith("mailto:"):
                continue
            if not (document.parent / target).resolve().exists():
                missing.append(f"{document.relative_to(ROOT)} -> {target}")

    assert not missing, "broken local Markdown links:\n" + "\n".join(missing)


def test_local_html_links_resolve() -> None:
    missing: list[str] = []
    for document in sorted((ROOT / "docs").rglob("*.html")):
        parser = _LocalLinkParser()
        parser.feed(document.read_text(encoding="utf-8"))
        for target in parser.targets:
            local = _local_target(target)
            if local is not None and not (document.parent / local).resolve().exists():
                missing.append(f"{document.relative_to(ROOT)} -> {target}")

    assert not missing, "broken local HTML links:\n" + "\n".join(missing)


def test_html_link_filter_ignores_nonlocal_targets() -> None:
    for target in (
        "#slide-2",
        "https://example.com/file.css",
        "//cdn.example.com/file.js",
        "mailto:author@example.com",
        "javascript:void(0)",
        "data:image/svg+xml;base64,PHN2Zy8+",
    ):
        assert _local_target(target) is None

    assert _local_target("../research/memo.md#section") == "../research/memo.md"
    assert _local_target("asset.js?v=1") == "asset.js"


def test_docs_and_source_do_not_reference_legacy_executable_paths() -> None:
    candidates = [
        ROOT / "README.md",
        *sorted(
            path
            for path in (ROOT / "docs").rglob("*")
            if path.is_file() and "superpowers" not in path.parts
        ),
        *sorted((ROOT / "src").rglob("*.py")),
        *sorted(path for path in (ROOT / "scripts").iterdir() if path.is_file()),
    ]
    legacy = re.compile(r"(?<![\w./-])(?:sweep/|explorer/[^`\s)\"']*\.py)")
    violations: list[str] = []
    for path in candidates:
        text = path.read_text(encoding="utf-8")
        if matches := sorted(set(legacy.findall(text))):
            violations.append(f"{path.relative_to(ROOT)}: {matches}")

    assert not violations, "legacy executable paths:\n" + "\n".join(violations)
