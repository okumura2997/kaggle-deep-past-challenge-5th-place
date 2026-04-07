"""PDF page rendering via pypdfium2."""

from __future__ import annotations

import re
import threading
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

# pypdfium2 is NOT thread-safe. Serialize all document operations.
_PDFIUM_LOCK = threading.Lock()

PAGE_IMAGE_RE = re.compile(
    r"^page_(\d{3,})\.(?:png|jpg|jpeg|webp)$", flags=re.IGNORECASE
)


def list_pdf_pages(
    pdf_path: Path,
    start_page: int = 1,
    end_page: int = -1,
) -> list[int]:
    """List 1-indexed page numbers in a PDF within the given range."""
    import pypdfium2 as pdfium

    with _PDFIUM_LOCK:
        doc = pdfium.PdfDocument(str(pdf_path))
        total = len(doc)
        doc.close()
    if end_page < 1 or end_page > total:
        end_page = total
    start_page = max(1, start_page)
    return list(range(start_page, end_page + 1))


def extract_pdf_page_bytes(pdf_path: Path, page_numbers: list[int]) -> bytes:
    """Extract specific 1-indexed pages from a PDF and return as new PDF bytes."""
    import io

    import pypdfium2 as pdfium

    with _PDFIUM_LOCK:
        src_doc = pdfium.PdfDocument(str(pdf_path))
        new_doc = pdfium.PdfDocument.new()
        page_indices = [p - 1 for p in page_numbers if 0 < p <= len(src_doc)]
        new_doc.import_pages(src_doc, page_indices)
        buf = io.BytesIO()
        new_doc.save(buf)
        new_doc.close()
        src_doc.close()
    return buf.getvalue()


def _render_pdf_page_chunk(
    pdf_path_str: str,
    page_tasks: list[tuple[int, str]],
    scale: float,
) -> tuple[int, list[tuple[int, str]]]:
    import pypdfium2 as pdfium

    doc = pdfium.PdfDocument(pdf_path_str)
    try:
        rendered_count = 0
        rendered_paths: list[tuple[int, str]] = []
        for page_num, image_path_str in page_tasks:
            image_path = Path(image_path_str)
            if not image_path.exists():
                page = doc[page_num - 1]
                pil_image = page.render(scale=scale).to_pil()
                pil_image.save(image_path, format="PNG")
                rendered_count += 1
            rendered_paths.append((page_num, image_path_str))
        return rendered_count, rendered_paths
    finally:
        doc.close()


def _chunk_page_tasks(
    page_tasks: list[tuple[int, Path]],
    chunk_count: int,
) -> list[list[tuple[int, Path]]]:
    if not page_tasks:
        return []
    chunk_size = max(1, (len(page_tasks) + chunk_count - 1) // chunk_count)
    return [
        page_tasks[i : i + chunk_size]
        for i in range(0, len(page_tasks), chunk_size)
    ]


def render_pdf_pages(
    pdf_path: Path,
    images_dir: Path,
    start_page: int,
    end_page: int,
    dpi: int,
    render_workers: int = 1,
) -> list[tuple[int, Path]]:
    """Render PDF pages to PNG images at specified DPI.

    Returns list of (page_number, image_path) tuples.
    """
    images_dir.mkdir(parents=True, exist_ok=True)

    try:
        import pypdfium2 as pdfium
    except ImportError:
        existing: list[tuple[int, Path]] = []
        for path in sorted(images_dir.glob("page_*")):
            match = PAGE_IMAGE_RE.match(path.name)
            if match is None:
                continue
            existing.append((int(match.group(1)), path))

        if not existing:
            raise ImportError(
                "pypdfium2 is required. Install with: pip install pypdfium2"
            )

        if start_page < 1:
            raise ValueError(f"start_page must be >= 1, got {start_page}")
        max_existing_page = max(page for page, _ in existing)
        effective_end_page = max_existing_page if end_page < 1 else end_page
        rendered = [
            (page, path)
            for page, path in existing
            if start_page <= page <= effective_end_page
        ]
        if not rendered:
            raise RuntimeError(
                "No cached page images found in requested range. "
                "Install pypdfium2 to render PDF pages."
            )
        print(
            "[warn] pypdfium2 is not installed. "
            f"Using existing cached images only: pdf={pdf_path.name}, pages={len(rendered)}"
        )
        return rendered

    doc = pdfium.PdfDocument(str(pdf_path))
    try:
        total_pages = len(doc)
        if start_page < 1:
            raise ValueError(f"start_page must be >= 1, got {start_page}")

        effective_end_page = total_pages if end_page < 1 else end_page
        if effective_end_page < start_page:
            raise ValueError(
                f"Invalid page range: start_page={start_page}, end_page={effective_end_page}"
            )
        if effective_end_page > total_pages:
            raise ValueError(
                f"end_page={effective_end_page} exceeds PDF pages={total_pages}"
            )

        rendered: list[tuple[int, Path]] = []
        num_skipped = 0
        pending_page_tasks: list[tuple[int, Path]] = []
        for page_num in range(start_page, effective_end_page + 1):
            image_path = images_dir / f"page_{page_num:03d}.png"
            if image_path.exists():
                num_skipped += 1
                rendered.append((page_num, image_path))
                continue
            pending_page_tasks.append((page_num, image_path))

        scale = dpi / 72.0
        num_rendered = 0
        worker_count = (
            max(1, min(render_workers, len(pending_page_tasks)))
            if pending_page_tasks
            else 0
        )
        if worker_count <= 1:
            for page_num, image_path in tqdm(
                pending_page_tasks,
                desc=f"render_pdf[{pdf_path.stem}]",
            ):
                page = doc[page_num - 1]
                pil_image = page.render(scale=scale).to_pil()
                pil_image.save(image_path, format="PNG")
                num_rendered += 1
                rendered.append((page_num, image_path))
        else:
            page_task_chunks = _chunk_page_tasks(
                pending_page_tasks, worker_count
            )
            with ProcessPoolExecutor(max_workers=worker_count) as executor:
                futures = [
                    executor.submit(
                        _render_pdf_page_chunk,
                        str(pdf_path),
                        [
                            (page_num, str(image_path))
                            for page_num, image_path in chunk
                        ],
                        scale,
                    )
                    for chunk in page_task_chunks
                ]
                with tqdm(
                    total=len(pending_page_tasks),
                    desc=f"render_pdf[{pdf_path.stem}]",
                ) as pbar:
                    for future in as_completed(futures):
                        rendered_count, rendered_paths = future.result()
                        num_rendered += rendered_count
                        rendered.extend(
                            (page_num, Path(image_path_str))
                            for page_num, image_path_str in rendered_paths
                        )
                        pbar.update(len(rendered_paths))

        rendered.sort(key=lambda item: item[0])

        print(
            f"render_pdf done. pdf={pdf_path.name}, pages={len(rendered)}, "
            f"rendered={num_rendered}, skipped_existing={num_skipped}"
        )
        return rendered
    finally:
        doc.close()


def render_pdf_pages_by_list(
    pdf_path: Path,
    images_dir: Path,
    pages: list[int],
    dpi: int,
    render_workers: int = 1,
) -> tuple[dict[int, Path], list[int]]:
    """Render specific PDF pages. Returns (rendered_map, invalid_pages)."""
    images_dir.mkdir(parents=True, exist_ok=True)
    requested_pages = sorted(set(page for page in pages if page >= 1))
    if not requested_pages:
        return {}, []

    try:
        import pypdfium2 as pdfium
    except ImportError:
        rendered: dict[int, Path] = {}
        for page_num in requested_pages:
            for path in sorted(images_dir.glob(f"page_{page_num:03d}.*")):
                if PAGE_IMAGE_RE.match(path.name) is None:
                    continue
                rendered[page_num] = path
                break
        missing_pages = [
            page_num
            for page_num in requested_pages
            if page_num not in rendered
        ]
        if missing_pages:
            raise ImportError(
                "pypdfium2 is required to render missing pages. "
                "Install with: pip install pypdfium2"
            )
        print(
            "[warn] pypdfium2 is not installed. "
            f"Using existing cached images only: pdf={pdf_path.name}, pages={len(rendered)}"
        )
        return rendered, []

    doc = pdfium.PdfDocument(str(pdf_path))
    try:
        total_pages = len(doc)
        valid_pages = [
            page_num for page_num in requested_pages if page_num <= total_pages
        ]
        invalid_pages = [
            page_num
            for page_num in requested_pages
            if page_num > total_pages
        ]

        scale = dpi / 72.0
        rendered: dict[int, Path] = {}
        num_skipped = 0
        pending_page_tasks: list[tuple[int, Path]] = []
        for page_num in valid_pages:
            image_path = images_dir / f"page_{page_num:03d}.png"
            if image_path.exists():
                num_skipped += 1
                rendered[page_num] = image_path
                continue
            pending_page_tasks.append((page_num, image_path))

        num_rendered = 0
        worker_count = (
            max(1, min(render_workers, len(pending_page_tasks)))
            if pending_page_tasks
            else 0
        )
        if worker_count <= 1:
            for page_num, image_path in tqdm(
                pending_page_tasks,
                desc=f"render:{pdf_path.stem}",
                leave=False,
            ):
                page = doc[page_num - 1]
                pil_image = page.render(scale=scale).to_pil()
                pil_image.save(image_path, format="PNG")
                num_rendered += 1
                rendered[page_num] = image_path
        else:
            page_task_chunks = _chunk_page_tasks(
                pending_page_tasks, worker_count
            )
            with ProcessPoolExecutor(max_workers=worker_count) as executor:
                futures = [
                    executor.submit(
                        _render_pdf_page_chunk,
                        str(pdf_path),
                        [
                            (page_num, str(image_path))
                            for page_num, image_path in chunk
                        ],
                        scale,
                    )
                    for chunk in page_task_chunks
                ]
                with tqdm(
                    total=len(pending_page_tasks),
                    desc=f"render:{pdf_path.stem}",
                    leave=False,
                ) as pbar:
                    for future in as_completed(futures):
                        rendered_count, rendered_paths = future.result()
                        num_rendered += rendered_count
                        for page_num, image_path_str in rendered_paths:
                            rendered[page_num] = Path(image_path_str)
                        pbar.update(len(rendered_paths))

        print(
            f"render_pdf done. pdf={pdf_path.name}, pages={len(valid_pages)}, "
            f"rendered={num_rendered}, skipped_existing={num_skipped}, invalid={len(invalid_pages)}"
        )
        return rendered, invalid_pages
    finally:
        doc.close()
