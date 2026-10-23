"""The paper collection as the API exposes it, built from chunks.jsonl.

PDF paths come only from this catalog, never from request input.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PAPERS_DIR = ROOT / "papers_avatar"
TOPICS = {
    "audio_speech_driven": "Audio-driven talking heads",
    "avatar_fingerprinting": "Avatar fingerprinting and verification",
    "text_driven_generation": "Text-driven generation",
    "text_to3dmodel_orfullbody": "3D and full-body avatars",
    "video_pose_driven_generation": "Video and pose-driven generation",
}
# Papers stored at the top level of papers_avatar/ have no topic folder.
TOPIC_BY_PAPER = {
    "2401.08559v2": "Text-driven generation",
    "facecraft": "3D and full-body avatars",
    "avatar imposter attack scenario": "Avatar fingerprinting and verification",
}
DEFAULT_TOPIC = "Deepfake detection"
MAX_SECTIONS = 40


class PaperCatalog:
    def __init__(self, records, papers_dir=PAPERS_DIR):
        self.papers_dir = Path(papers_dir).resolve()
        self.papers = {}
        for record in records:
            paper = self.papers.setdefault(record["document_id"], {
                "id": record["document_id"],
                "title": record.get("document_title") or record["document_id"],
                "source_file": record.get("source_file") or "",
                "chunks": 0,
                "pages": 0,
                "sections": [],
            })
            paper["chunks"] += 1
            paper["pages"] = max([paper["pages"], *(record.get("page_numbers") or [])])
            path = record.get("section_path") or []
            section = path[1] if len(path) > 1 else None
            if section and section not in paper["sections"] and len(paper["sections"]) < MAX_SECTIONS:
                paper["sections"].append(section)

    def __contains__(self, paper_id):
        return paper_id in self.papers

    def __len__(self):
        return len(self.papers)

    def pdf_path(self, paper_id):
        paper = self.papers.get(paper_id)
        if not paper or not paper["source_file"]:
            return None
        path = (self.papers_dir / paper["source_file"]).resolve()
        if not path.is_relative_to(self.papers_dir) or not path.is_file():
            return None
        return path

    def views(self):
        views = []
        for paper in self.papers.values():
            folder = Path(paper["source_file"]).parts[0] if len(Path(paper["source_file"]).parts) > 1 else ""
            views.append({
                "id": paper["id"],
                "title": paper["title"],
                "topic": TOPIC_BY_PAPER.get(paper["id"]) or TOPICS.get(folder, DEFAULT_TOPIC),
                "chunks": paper["chunks"],
                "pages": paper["pages"],
                "sections": paper["sections"],
                "has_pdf": self.pdf_path(paper["id"]) is not None,
            })
        return sorted(views, key=lambda view: (view["topic"], view["title"].lower()))
