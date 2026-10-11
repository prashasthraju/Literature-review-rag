
from pathlib import Path
import json
import traceback

import pandas as pd

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling_core.types.doc import (
    ImageRefMode,
    PictureItem,
    TableItem,
)

INPUT_DIR = Path(
    r"C:\Users\prash\OneDrive\Desktop\atten\ragalu\papers_avatar"
)
OUTPUT_DIR = Path("parsed")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Configure Docling to retain visual content.
pipeline_options = PdfPipelineOptions(
    generate_picture_images=True,
    generate_page_images=False,
    images_scale=2.0,
)

converter = DocumentConverter(
    format_options={
        InputFormat.PDF: PdfFormatOption(
            pipeline_options=pipeline_options
        )
    }
)

failed = []
total = 0
successful = 0


def parse_paper(pdf_path):
    # Use a separate directory for every PDF.
    relative_path = pdf_path.relative_to(INPUT_DIR)
    paper_dir = OUTPUT_DIR / relative_path.with_suffix("")
    paper_dir.mkdir(parents=True, exist_ok=True)

    images_dir = paper_dir / "images"
    tables_dir = paper_dir / "tables"

    images_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nConverting: {pdf_path.name}")

    result = converter.convert(str(pdf_path))
    doc = result.document

    # 1. Save the complete structured document.
    doc.save_as_json(paper_dir / "document.json")

    # 2. Export readable Markdown with image references.
    doc.save_as_markdown(
        paper_dir / "document.md",
        image_mode=ImageRefMode.REFERENCED,
        artifacts_dir=images_dir,
    )

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


# 5. Process all PDFs recursively.
for pdf_path in INPUT_DIR.rglob("*"):
    if not pdf_path.is_file():
        continue

    if pdf_path.suffix.lower() != ".pdf":
        continue

    total += 1

    try:
        parse_paper(pdf_path)
        successful += 1

    except Exception as exc:
        failed.append({
            "file": str(pdf_path),
            "error": str(exc),
            "traceback": traceback.format_exc(),
        })

        print(f"FAILED: {pdf_path.name}: {exc}")


# 6. Save a corpus-level failure report.
with (OUTPUT_DIR / "failures.json").open(
    "w", encoding="utf-8"
) as f:
    json.dump(failed, f, ensure_ascii=False, indent=2)

print("\n========== CORPUS SUMMARY ==========")
print(f"PDFs found: {total}")
print(f"Successful: {successful}")
print(f"Failed: {len(failed)}")
print(f"Output directory: {OUTPUT_DIR.resolve()}")
