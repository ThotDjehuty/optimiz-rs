"""Inject matplotlib plots inline below every executable Python code-block
across the entire `docs/source/` tree.

For each `.md` and `.rst` file:

1. Parse out the Python code-blocks (fenced ```python``` for Markdown,
   `.. code-block:: python` for reStructuredText).
2. Execute each block in its own namespace with a non-interactive matplotlib
   backend.  Every figure produced by the block is saved as a PNG under
   `docs/source/_static/auto/<page-slug>/<idx>.png`.
3. Rewrite the page so that, immediately after the executable code-block, a
   markdown image (`![caption](path)`) or RST `.. image::` directive points
   to the captured PNG.
4. Blocks that fail to execute (missing imports, illustrative pseudo-code,
   external data dependencies …) are left untouched — no image is inserted
   and the existing source is preserved.

Idempotent: previously injected image directives are detected and refreshed
in place rather than duplicated.

The script is purely additive on the v2.0.0 branch — it touches no Rust or
binding code, only the doc tree and the auto-generated `_static/auto`
folder.
"""
from __future__ import annotations

import io
import os
import re
import sys
import traceback
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DOC_SRC = ROOT / "docs" / "source"
STATIC_DIR = DOC_SRC / "_static" / "auto"
SKIP_DIRS = {"_static", "_build", "_templates"}

# A small allow-list of pages we KNOW exercise the bindings in a way that is
# self-contained.  Every other page is still scanned, but if a code block
# fails we silently leave it alone.

# Auto-injected marker so that re-runs are idempotent.
MD_BEGIN = "<!-- AUTO-PLOT-BEGIN -->"
MD_END   = "<!-- AUTO-PLOT-END -->"
RST_BEGIN = ".. AUTO-PLOT-BEGIN"
RST_END   = ".. AUTO-PLOT-END"


# ---------------------------------------------------------------------------

def slugify(path: Path) -> str:
    rel = path.relative_to(DOC_SRC).with_suffix("")
    return str(rel).replace(os.sep, "__")


def execute_block(code: str, page_ns: dict) -> list[Path] | None:
    """Run `code` in `page_ns`. Return a list of PNG paths for the figures
    it created, or `None` if the block raised."""
    plt.close("all")
    buf_out, buf_err = io.StringIO(), io.StringIO()
    try:
        with redirect_stdout(buf_out), redirect_stderr(buf_err):
            exec(compile(code, "<doc-block>", "exec"), page_ns)
    except BaseException:  # noqa: BLE001 — also catch pyo3 PanicException
        return None
    return list(plt.get_fignums())


def save_figs(page_slug: str, block_idx: int) -> list[str]:
    out_dir = STATIC_DIR / page_slug
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    for i, num in enumerate(plt.get_fignums(), start=1):
        fig = plt.figure(num)
        fname = f"block_{block_idx:02d}_fig_{i:02d}.png"
        fpath = out_dir / fname
        fig.savefig(fpath, bbox_inches="tight", dpi=110)
        written.append(fpath.as_posix())
    plt.close("all")
    return written


def relative_to_doc(target: Path, page_path: Path) -> str:
    return os.path.relpath(target, page_path.parent).replace(os.sep, "/")


# ---------------------------------------------------------------------------
# Markdown
# ---------------------------------------------------------------------------

MD_BLOCK_RE = re.compile(
    r"(?P<head>```(?:python|py)\s*\n)(?P<body>.*?)(?P<tail>```)",
    re.DOTALL,
)
MD_AUTO_BLOCK_RE = re.compile(
    re.escape(MD_BEGIN) + r".*?" + re.escape(MD_END) + r"\n?",
    re.DOTALL,
)


def process_markdown(path: Path) -> bool:
    text = path.read_text()
    # Strip previously-injected auto blocks so we re-run from a clean slate.
    text = MD_AUTO_BLOCK_RE.sub("", text)

    page_slug = slugify(path)
    page_ns: dict = {"__name__": "__doc__"}
    out_chunks: list[str] = []
    last = 0
    block_idx = 0
    n_injected = 0

    for m in MD_BLOCK_RE.finditer(text):
        block_idx += 1
        out_chunks.append(text[last:m.end()])
        last = m.end()

        code = m.group("body")
        nums = execute_block(code, page_ns)
        if not nums:
            continue
        pngs = save_figs(page_slug, block_idx)
        if not pngs:
            continue

        block = ["", MD_BEGIN]
        for png in pngs:
            rel = relative_to_doc(Path(png), path)
            block.append(f"![Generated plot]({rel})")
        block.append(MD_END)
        block.append("")
        out_chunks.append("\n".join(block))
        n_injected += len(pngs)

    out_chunks.append(text[last:])
    new_text = "".join(out_chunks)
    if new_text != path.read_text():
        path.write_text(new_text)
    return n_injected > 0


# ---------------------------------------------------------------------------
# reStructuredText
# ---------------------------------------------------------------------------

RST_BLOCK_RE = re.compile(
    r"(?P<indent>[ \t]*)\.\. code-block::[ \t]+python\s*\n"
    r"(?:[ \t]*:.*\n)*"
    r"\n?"
    r"(?P<body>(?:(?:\1[ \t]+.*|[ \t]*)\n)+)",
)
RST_AUTO_BLOCK_RE = re.compile(
    re.escape(RST_BEGIN) + r".*?" + re.escape(RST_END) + r"\n?",
    re.DOTALL,
)


def dedent_rst_body(body: str, indent: str) -> str:
    inner_indent = indent + "   "
    out = []
    for line in body.splitlines():
        if line.startswith(inner_indent):
            out.append(line[len(inner_indent):])
        elif line.strip() == "":
            out.append("")
        else:
            # End of code-block (different indentation): stop early.
            break
    return "\n".join(out)


def process_rst(path: Path) -> bool:
    text = path.read_text()
    text = RST_AUTO_BLOCK_RE.sub("", text)

    page_slug = slugify(path)
    page_ns: dict = {"__name__": "__doc__"}
    out_chunks: list[str] = []
    last = 0
    block_idx = 0
    n_injected = 0

    for m in RST_BLOCK_RE.finditer(text):
        block_idx += 1
        out_chunks.append(text[last:m.end()])
        last = m.end()

        indent = m.group("indent")
        code = dedent_rst_body(m.group("body"), indent)
        nums = execute_block(code, page_ns)
        if not nums:
            continue
        pngs = save_figs(page_slug, block_idx)
        if not pngs:
            continue

        lines = ["", f"{indent}{RST_BEGIN}"]
        for png in pngs:
            rel = relative_to_doc(Path(png), path)
            lines.append(f"{indent}.. image:: {rel}")
            lines.append(f"{indent}   :align: center")
            lines.append(f"{indent}   :width: 80%")
            lines.append("")
        lines.append(f"{indent}{RST_END}")
        lines.append("")
        out_chunks.append("\n".join(lines))
        n_injected += len(pngs)

    out_chunks.append(text[last:])
    new_text = "".join(out_chunks)
    if new_text != path.read_text():
        path.write_text(new_text)
    return n_injected > 0


# ---------------------------------------------------------------------------

def iter_doc_pages():
    for dirpath, dirnames, filenames in os.walk(DOC_SRC):
        # Prune
        rel = Path(dirpath).relative_to(DOC_SRC)
        parts = set(rel.parts)
        if parts & SKIP_DIRS:
            dirnames.clear()
            continue
        for fn in filenames:
            if fn.endswith((".md", ".rst")) and not fn.endswith(".backup"):
                yield Path(dirpath) / fn


def main() -> None:
    STATIC_DIR.mkdir(parents=True, exist_ok=True)
    n_pages = 0
    n_with_plots = 0
    total_imgs = 0
    for page in sorted(iter_doc_pages()):
        n_pages += 1
        try:
            if page.suffix == ".md":
                injected = process_markdown(page)
            else:
                injected = process_rst(page)
        except Exception:  # noqa: BLE001
            print(f"!! {page.relative_to(DOC_SRC)} crashed:")
            traceback.print_exc()
            continue
        if injected:
            n_with_plots += 1
        # Count images in _static/auto/<page_slug>
        img_dir = STATIC_DIR / slugify(page)
        if img_dir.exists():
            n_imgs = len(list(img_dir.glob("*.png")))
            total_imgs += n_imgs
            print(f"   {page.relative_to(DOC_SRC)}  ->  {n_imgs} plot(s)")
        else:
            print(f"   {page.relative_to(DOC_SRC)}  ->  no plots")
    print()
    print(f"Scanned {n_pages} pages, {n_with_plots} got new images, "
          f"{total_imgs} PNGs total.")


if __name__ == "__main__":
    main()
