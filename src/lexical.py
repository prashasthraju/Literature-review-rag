import math
import re
from collections import Counter, defaultdict

TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text):
    return TOKEN_RE.findall((text or "").lower())


class BM25Index:
    def __init__(self, documents, k1=1.5, b=0.75):
        self.k1 = k1
        self.b = b
        self.doc_ids = []
        self.doc_len = []
        self.tf = []
        df = Counter()

        for doc_id, text in documents:
            tokens = tokenize(text)
            counts = Counter(tokens)
            self.doc_ids.append(doc_id)
            self.doc_len.append(len(tokens))
            self.tf.append(counts)
            df.update(counts.keys())

        self.n_docs = len(self.doc_ids)
        self.avgdl = (sum(self.doc_len) / self.n_docs) if self.n_docs else 0.0
        self.idf = {
            term: math.log(1 + (self.n_docs - freq + 0.5) / (freq + 0.5))
            for term, freq in df.items()
        }

    def search(self, query, limit=20):
        q_tokens = tokenize(query)
        if not q_tokens or not self.doc_ids:
            return []

        scores = defaultdict(float)
        q_counts = Counter(q_tokens)
        for term, qf in q_counts.items():
            idf = self.idf.get(term)
            if not idf:
                continue
            for idx, term_freqs in enumerate(self.tf):
                tf = term_freqs.get(term)
                if not tf:
                    continue
                denom = tf + self.k1 * (
                    1 - self.b + self.b * self.doc_len[idx] / self.avgdl
                )
                scores[idx] += qf * idf * (tf * (self.k1 + 1)) / denom

        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)[:limit]
        return [(self.doc_ids[idx], score) for idx, score in ranked]


def reciprocal_rank_fusion(*ranked_id_lists, k=60, limit=5):
    fused = defaultdict(float)
    for ranked in ranked_id_lists:
        for rank, doc_id in enumerate(ranked, start=1):
            if not doc_id:
                continue
            fused[doc_id] += 1.0 / (k + rank)
    return [
        doc_id
        for doc_id, _score in sorted(fused.items(), key=lambda item: item[1], reverse=True)[:limit]
    ]
