
import argparse
import json
import traceback
from pathlib import Path

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling_core.types.doc import (
    DoclingDocument,
    ImageRefMode,
    PictureItem,
    TableItem,
)

INPUT_DIR = Path("papers_avatar")
OUTPUT_DIR = Path("parsed")
PAGE_BREAK_PLACEHOLDER = "<!-- docling-page-break-placeholder -->"

# Configure Docling to retain visual content.
pipeline_options = PdfPipelineOptions(
    generate_picture_images=True,
    generate_page_images=False,
    images_scale=2.0,
)

converter = None


def get_converter():
    global converter
    if converter is None:
        converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_options=pipeline_options
                )
            }
        )
    return converter


def save_markdown(doc, paper_dir):
    images_dir = paper_dir / "images"
    markdown_path = paper_dir / "document.md"
    doc.save_as_markdown(
        markdown_path,
        image_mode=ImageRefMode.REFERENCED,
        artifacts_dir=images_dir,
        page_break_placeholder=PAGE_BREAK_PLACEHOLDER,
    )
    page_numbers = []
    previous_page = None
    for item, _level in doc.iterate_items(with_groups=True, traverse_pictures=True):
        provenance = getattr(item, "prov", None)
        if not provenance:
            continue
        page_number = provenance[0].page_no
        if previous_page is None:
            page_numbers.append(page_number)
            previous_page = page_number
        elif page_number > previous_page:
            page_numbers.append(page_number)
            previous_page = page_number

    markdown = markdown_path.read_text(encoding="utf-8")
    transitions = page_numbers[1:]
    if markdown.count(PAGE_BREAK_PLACEHOLDER) != len(transitions):
        raise ValueError(
            f"Page marker count does not match Docling provenance in {markdown_path}"
        )
    for page_number in transitions:
        markdown = markdown.replace(
            PAGE_BREAK_PLACEHOLDER,
            f"<!-- page-break:{page_number} -->",
            1,
        )
    page_start = page_numbers[0] if page_numbers else 1
    markdown_path.write_text(
        f"<!-- page-start:{page_start} -->\n\n{markdown}",
        encoding="utf-8",
    )


def parse_paper(pdf_path):
    # Use a separate directory for every PDF.
    relative_path = pdf_path.resolve().relative_to(INPUT_DIR.resolve())
    paper_dir = OUTPUT_DIR / relative_path.with_suffix("")
    paper_dir.mkdir(parents=True, exist_ok=True)

    images_dir = paper_dir / "images"
    tables_dir = paper_dir / "tables"

    images_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nConverting: {pdf_path.name}")

    result = get_converter().convert(str(pdf_path))
    doc = result.document

    # 1. Save the complete structured document.
    doc.save_as_json(paper_dir / "document.json")

    # 2. Export readable Markdown with image references.
    save_markdown(doc, paper_dir)

    pictures = []
    tables = []

    # 3. Export individual figures and tables.
    for element, _level in doc.iterate_items():

        if isinstance(element, PictureItem):
            picture_num = len(pictures) + 1
            image_path = (
                images_dir / f"figure_{picture_num:03d}.png"
            )

            try:
                image = element.get_image(doc)

                if image is not None:
                    image.save(image_path, "PNG")

                    pictures.append({
                        "figure_number": picture_num,
                        "path": str(image_path),
                        "caption": element.caption_text(doc),
                        "provenance": [
                            item.model_dump(mode="json")
                            for item in element.prov
                        ],
                    })

            except Exception as exc:
                print(f"Figure export warning: {exc}")

        elif isinstance(element, TableItem):
            table_num = len(tables) + 1
            table_dir = tables_dir / f"table_{table_num:03d}"
            table_dir.mkdir(parents=True, exist_ok=True)

            table_record = {
                "table_number": table_num,
                "caption": element.caption_text(doc),
                "provenance": [
                    item.model_dump(mode="json")
                    for item in element.prov
                ],
            }

            # Preserve table structure as HTML and CSV.
            try:
                html = element.export_to_html(doc=doc)
                (table_dir / "table.html").write_text(
                    html, encoding="utf-8"
                )

                dataframe = element.export_to_dataframe(doc=doc)
                dataframe.to_csv(
                    table_dir / "table.csv",
                    index=False,
                    encoding="utf-8-sig",
                )

                table_record["html_path"] = str(
                    table_dir / "table.html"
                )
                table_record["csv_path"] = str(
                    table_dir / "table.csv"
                )

            except Exception as exc:
                print(f"Table data export warning: {exc}")

            # Save a visual crop of the table, if available.
            try:
                image = element.get_image(doc)

                if image is not None:
                    image_path = table_dir / "table.png"
                    image.save(image_path, "PNG")
                    table_record["image_path"] = str(image_path)

            except Exception as exc:
                print(f"Table image warning: {exc}")

            tables.append(table_record)

    # 4. Write a compact manifest for downstream RAG processing.
    manifest = {
        "document_id": pdf_path.stem,
        "source_file": str(relative_path),
        "markdown_path": str(paper_dir / "document.md"),
        "structured_json_path": str(paper_dir / "document.json"),
        "picture_count": len(pictures),
        "table_count": len(tables),
        "pictures": pictures,
        "tables": tables,
    }

    with (paper_dir / "manifest.json").open(
        "w", encoding="utf-8"
    ) as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"Figures exported: {len(pictures)}")
    print(f"Tables exported: {len(tables)}")
    print(f"Output: {paper_dir}")


def main():
    global INPUT_DIR, OUTPUT_DIR

    parser = argparse.ArgumentParser(description="Parse research-paper PDFs with Docling.")
    parser.add_argument("--input-dir", type=Path, default=Path("papers_avatar"))
    parser.add_argument("--output-dir", type=Path, default=Path("parsed"))
    parser.add_argument(
        "--pdf",
        type=Path,
        help="Parse one PDF, specified relative to --input-dir or as an absolute path",
    )
    parser.add_argument(
        "--markdown-only",
        action="store_true",
        help="Regenerate Markdown with page markers from existing Docling JSON",
    )
    args = parser.parse_args()

    INPUT_DIR = args.input_dir.resolve()
    OUTPUT_DIR = args.output_dir.resolve()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.pdf:
        selected_pdf = args.pdf if args.pdf.is_absolute() else INPUT_DIR / args.pdf
        selected_pdf = selected_pdf.resolve()
        try:
            selected_pdf.relative_to(INPUT_DIR)
        except ValueError:
            parser.error("--pdf must point to a file inside --input-dir")
        if not selected_pdf.is_file() or selected_pdf.suffix.lower() != ".pdf":
            parser.error(f"Not a PDF file: {selected_pdf}")
        pdf_paths = [selected_pdf]
    else:
        pdf_paths = sorted(
            path
            for path in INPUT_DIR.rglob("*")
            if path.is_file() and path.suffix.lower() == ".pdf"
        )

    failure_path = OUTPUT_DIR / "failures.json"
    failed = []
    if args.pdf and failure_path.exists():
        try:
            failed = json.loads(failure_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            failed = []
        selected_name = str(selected_pdf)
        failed = [item for item in failed if item.get("file") != selected_name]

    successful = 0
    for pdf_path in pdf_paths:
        try:
            if args.markdown_only:
                relative_path = pdf_path.relative_to(INPUT_DIR)
                paper_dir = OUTPUT_DIR / relative_path.with_suffix("")
                document_json = paper_dir / "document.json"
                if not document_json.is_file():
                    raise FileNotFoundError(str(document_json))
                doc = DoclingDocument.load_from_json(document_json)
                save_markdown(doc, paper_dir)
            else:
                parse_paper(pdf_path)
            successful += 1
        except Exception as exc:
            failed.append({
                "file": str(pdf_path),
                "error": str(exc),
                "traceback": traceback.format_exc(),
            })
            print(f"FAILED: {pdf_path.name}: {exc}")

    failure_path.write_text(
        json.dumps(failed, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\n========== CORPUS SUMMARY ==========")
    print(f"PDFs selected: {len(pdf_paths)}")
    print(f"Successful: {successful}")
    print(f"Failed: {len(failed)}")
    print(f"Output directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
