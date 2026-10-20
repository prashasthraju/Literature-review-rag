"""One-off migration (already applied): attach chunking-independent evidence
anchors to the dev set.

Each anchor was written from a chunk-ID label and is checked against the
chunking those labels were made for, so run it on the original chunks:

    git show a0735cf:chunks.jsonl > /tmp/chunks_a0735cf.jsonl
    python scripts/add_evidence_anchors.py --chunks /tmp/chunks_a0735cf.jsonl [--write]
"""
import json, sys
from pathlib import Path
sys.path.insert(0, ".")
from src.evaluation.evidence import EvidenceResolver

# query_id -> [(grade, anchor(s), source_chunk_id prefixes)]
E = {
 "retrieval-001": [(2, "users can specify a multi-track timeline of multiple prompts organized in temporal intervals that may overlap", ["606d896b"]),
                   (1, "the multi-track timeline enables users to define multiple intervals, each linked to a natural language prompt", ["4015a1b4"])],
 "retrieval-002": [(2, ["driven by audio and a reference portrait image", "we extract 3D intermediate representations from audio"], ["d9d35450"])],
 "retrieval-003": [(2, "which comprises 40 distinct deepfake techniques", ["98520abb"])],
 "retrieval-004": [(2, "existing methods either require extensive multiview data or struggle with shape accuracy and identity consistency", ["ee806e33"]),
                   (1, "The goal of this work is to create a 4D facial avatar from just a single image", ["04d39a48"])],
 "retrieval-005": [(2, "raising security concerns regarding identity impersonation in AI-mediated communication", ["abf39be0"]),
                   (1, "synthetic media can be exploited for impersonation attacks, social engineering, and identity fraud", ["e680621b"]),
                   (1, "One emerging threat is impersonation: an attacker can steal a user's avatar", ["4f3987d7"])],
 "retrieval-006": [(2, ["Different Improvement Points of FFP-ChT", "DF(c23) - ACC", "DF(c23) - AUC", "Celeb-DF(v2) - ACC", "Celeb-DF(v2) - AUC"], ["b05f46e8", "c408a6ec"])],
 "retrieval-007": [(2, "generates 3D motion coefficients (head pose, expression) of the 3DMM from audio", ["dcd95adf"]),
                   (1, "we consider the motion coefficients of the 3DMM as the intermediate representation", ["ae060bc0"])],
 "retrieval-008": [(2, ["creating realistic human avatars with dynamic 3D appearances from a single video", "introducing animatable 3D Gaussians"], ["f5353f79"])],
 "retrieval-009": [(2, "The proposed framework comprises two modules, namely Audio2Lmk and Lmk2Video", ["f4e58912"]),
                   (2, "our proposed Lmk2Video module creates a temporally consistent portrait animation", ["93ce9f41"])],
 "retrieval-010": [(2, ["pre-train a large Image-to-Plane (I2P) model", "we design an effective facial Motion Adapter (MA)"], ["01822e95"]),
                   (2, "we design a Head-Torso-Background Super-Resolution (HTB-SR) model", ["20c193da"])],
 "retrieval-011": [(2, "It consists of 44970 text annotations for 14616 motions", ["927e829f"])],
 "retrieval-012": [(2, ["fed into the BlazeFace model for face detection", "fed into the FaceMesh model to extract 468 FFPs"], ["c11421dd"])],
 "retrieval-013": [(2, "consecutive feature maps are subtracted in the learned deep feature space", ["63f70080"]),
                   (1, "We propose a preprocessing-free, end-to-end avatar fingerprinting framework that combines", ["bf567533"])],
 "retrieval-014": [(2, "Our method segments the speech input into consecutive temporal windows for motion generation", ["06dc433e"]),
                   (1, "we model these codes with a Transformer that is autoregressive across scales", ["39581a2e"])],
 "retrieval-015": [(2, ["Comparison across methods, including style adaptation during inference", "ARTalk (ours) is the only method to achieve all three features"], ["55fc452b"])],
 "retrieval-016": [(2, ["MotionDiffuse, the first diffusion model-based text-driven motion generation framework", "1) Probabilistic Mapping"], ["d2cb665f"])],
 "retrieval-017": [(2, "generating motion in real-time under causal constraints and learning expressive, vibrant reactions without additional labeled data", ["331dc959"])],
 "retrieval-018": [(2, "leverages a discrete codebook as an intermediate feature to establish a connection between text and avatars", ["d9adf041"]),
                   (1, "The codebook corresponds to the attribute information with the corresponding number in the text library", ["e5640d97"])],
 "retrieval-019": [(2, "our key insight is to project the unstructured 3D Gaussian into a structured 2D space", ["65a6e105", "ab764454"])],
 "retrieval-020": [(2, "verifying who drives a synthetic talking-head video rather than whether it is real", ["63f70080"]),
                   (1, "the task of avatar fingerprinting aims to determine whether two avatar videos are driven by the same human operator or not", ["abf39be0"]),
                   (2, "given two avatar videos, the goal is to determine whether they are driven by the same underlying identity or not", ["87691632"])],
 "retrieval-021": [(2, "comprehensive evaluations using 4 standard evaluation protocols and 8 representative detection methods, resulting in over 2,000 evaluations", ["98520abb"]),
                   (2, "we conducted over 2,000+ evaluations using 8 representative detectors under 4 standard evaluation protocols", ["47d410d5"])],
 "retrieval-022": [(2, "At each denoising step, STMC first applies the diffusion model on each text prompt in the timeline independently", ["17e80f6d"])],
 "retrieval-023": [(2, "We use the TFHP dataset proposed by Sun et al. [2024] to train our model, which consists of 1,052 video clips from 588 subjects", ["e1f8e406"])],
 "retrieval-024": [(2, "we propose to individually model the head, torso, and background segments and compose them into the final image during the rendering process", ["20c193da"]),
                   (1, "we propose to model the torso part with a 2D warping-based renderer", ["996804a6"])],
 "retrieval-025": [(2, "controlled evaluation of avatar fingerprinting systems across multiple state-of-the-art synthesis pipelines and source datasets", ["82b2543a"]),
                   (1, "the present article introduces AVAPrintDB, a new publicly available multi-generator talking-head avatar database for avatar fingerprinting", ["abf39be0"]),
                   (2, ["We define a standardized and reproducible benchmark for avatar fingerprinting", "including identity-disjoint splits"], ["59dce459"])],
 "retrieval-026": [(2, ["FFP-ChT (ours)", "98.52", "99.70"], ["c408a6ec"])],
}

path = Path("data/evaluation/retrieval_dev.jsonl")
rows = [json.loads(l) for l in path.open(encoding="utf-8") if l.strip()]
chunks_path = sys.argv[sys.argv.index("--chunks") + 1] if "--chunks" in sys.argv else "chunks.jsonl"
chunks = [json.loads(l) for l in open(chunks_path, encoding="utf-8") if l.strip()]
by_prefix = {c["chunk_id"][:8]: c for c in chunks}
resolver = EvidenceResolver(chunks)
ok = True
for row in rows:
    units = []
    for grade, anchor, prefixes in E[row["query_id"]]:
        sources = [by_prefix[p]["chunk_id"] for p in prefixes]
        doc = by_prefix[prefixes[0]]["document_id"]
        unit = {"grade": grade, "document_id": doc, "anchor": anchor, "source_chunk_ids": sources}
        matched = resolver.resolve_unit(unit)
        missing = set(sources) - matched
        extra = matched - set(sources)
        if missing:
            ok = False; print("MISSING", row["query_id"], anchor, missing)
        if extra:
            print("extra", row["query_id"], str(anchor)[:60], len(extra))
        units.append(unit)
    covered = {cid for u in units for cid in u["source_chunk_ids"]}
    labeled = {cid for cid, g in row["relevant_chunk_ids"].items() if g > 0}
    if covered != labeled:
        ok = False; print("LABEL MISMATCH", row["query_id"], labeled ^ covered)
    row["evidence"] = units
print("all anchors resolve to their source chunks:", ok)
if ok and "--write" in sys.argv:
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    print("written")
