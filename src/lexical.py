import math
import re
from collections import Counter, defaultdict

TOKEN_RE = re.compile(r"[a-z0-9]+")

# Function words and question words carry no evidence but still add BM25 mass
# to long chunks, e.g. "what", "does", "each" in "what does each one produce?".
STOPWORDS = frozenset(
    """
    a about above after again against all am an and any are as at be because been
    before being below between both but by can could did do does doing down during
    each few for from further had has have having he her here hers him his how i if
    in into is it its itself just me more most my no nor not now of off on once only
    or other our ours out over own same she should so some such than that the their
    theirs them then there these they this those through to too under until up very
    was we were what when where which while who whom why will with would you your
    """.split()
)


def tokenize(text, stopwords=STOPWORDS):
    return [token for token in TOKEN_RE.findall((text or "").lower()) if token not in stopwords]


class BM25Index:
    def __init__(self, documents, k1=1.5, b=0.75, stopwords=STOPWORDS):
        self.k1 = k1
        self.b = b
        self.stopwords = stopwords
        self.doc_ids = []
        self.doc_len = []
        self.postings = defaultdict(list)

        for doc_id, text in documents:
            tokens = tokenize(text, stopwords)
            index = len(self.doc_ids)
            self.doc_ids.append(doc_id)
            self.doc_len.append(len(tokens))
            for term, tf in Counter(tokens).items():
                self.postings[term].append((index, tf))

        self.n_docs = len(self.doc_ids)
        self.avgdl = (sum(self.doc_len) / self.n_docs) if self.n_docs else 0.0
        self.idf = {
            term: math.log(1 + (self.n_docs - len(posting) + 0.5) / (len(posting) + 0.5))
            for term, posting in self.postings.items()
        }

    def search(self, query, limit=20):
        q_tokens = tokenize(query, self.stopwords)
        if not q_tokens or not self.doc_ids:
            return []

        scores = defaultdict(float)
        for term, qf in Counter(q_tokens).items():
            idf = self.idf.get(term)
            if not idf:
                continue
            for idx, tf in self.postings[term]:
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
