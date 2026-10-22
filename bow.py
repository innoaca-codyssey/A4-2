"""NumPy TF-IDF, 코사인 검색, BM25와 역색인."""

import numpy as np
from preprocess import tokenize


class NumpyTfidf:
    """학습 문서의 빈도/IDF만 사용해 벡터를 만든다."""

    def __init__(self, tf="log", smooth_idf=True, norm="l2"):
        if tf not in ("raw", "log", "double") or norm not in (None, "l2"):
            raise ValueError("지원하지 않는 TF/정규화 설정")
        self.tf = tf
        self.smooth_idf = smooth_idf
        self.norm = norm

    def counts(self, texts):
        result = np.zeros((len(texts), len(self.vocabulary_)), dtype=np.float64)
        for row, text in enumerate(texts):
            indices = [
                self.vocabulary_[w] for w in tokenize(text) if w in self.vocabulary_
            ]
            result[row] = np.bincount(indices, minlength=len(self.vocabulary_))
        return result

    def fit(self, texts):
        texts = list(texts)
        if not texts:
            raise ValueError("학습 문서가 없습니다")
        words = sorted({word for text in texts for word in tokenize(text)})
        if not words:
            raise ValueError("사용 가능한 어휘가 없습니다")
        self.vocabulary_ = {word: i for i, word in enumerate(words)}
        self.features_ = np.asarray(words)
        self.counts_ = self.counts(texts)
        df = np.count_nonzero(self.counts_, axis=0)
        smooth = int(self.smooth_idf)
        self.idf_ = np.log((len(texts) + smooth) / (df + smooth)) + 1
        self.matrix_ = self.weight(self.counts_)
        return self

    def weight(self, counts):
        tf = counts.copy().astype(np.float64)
        nonzero = counts > 0
        if self.tf == "log":
            tf[nonzero] = 1 + np.log(counts[nonzero])
        elif self.tf == "double":
            maximum = counts.max(axis=1, keepdims=True)
            scaled = np.divide(
                counts, maximum, out=np.zeros_like(tf), where=maximum > 0
            )
            tf = np.where(nonzero, 0.5 + 0.5 * scaled, 0)
        matrix = tf * self.idf_
        if self.norm == "l2":
            lengths = np.linalg.norm(matrix, axis=1, keepdims=True)
            matrix = np.divide(
                matrix, lengths, out=np.zeros_like(matrix), where=lengths > 0
            )
        return matrix

    def transform(self, texts):
        return self.weight(self.counts(list(texts)))

    def fit_transform(self, texts):
        return self.fit(texts).matrix_


def cosine(matrix, query):
    """입력 정규화 여부에 관계없이 코사인을 직접 계산한다."""
    matrix = np.asarray(matrix, dtype=np.float64)
    query = np.asarray(query, dtype=np.float64).reshape(-1)
    denominator = np.linalg.norm(matrix, axis=1) * np.linalg.norm(query)
    return np.divide(
        matrix @ query, denominator, out=np.zeros(len(matrix)), where=denominator > 0
    )


class SearchIndex:
    """문서 벡터와 비제로 전치 목록을 연결한다."""

    def __init__(self, texts, tf="log"):
        self.vectorizer = NumpyTfidf(tf=tf).fit(texts)
        self.matrix = self.vectorizer.matrix_
        self.postings = {
            j: (
                np.flatnonzero(self.matrix[:, j]),
                self.matrix[self.matrix[:, j] != 0, j],
            )
            for j in range(self.matrix.shape[1])
        }

    def scores(self, query, inverted=False):
        q = self.vectorizer.transform([query])[0]
        if not inverted:
            return self.matrix @ q
        result = np.zeros(len(self.matrix))
        # 양쪽이 L2정규화돼 있어 같은 내적을 비제로 목록에서 누적한다.
        for j in np.flatnonzero(q):
            ids, values = self.postings[j]
            result[ids] += values * q[j]
        return result

    def top(self, query, k=5, inverted=False):
        if k <= 0:
            raise ValueError("k는 양수여야 합니다")
        scores = self.scores(query, inverted)
        if not np.any(scores):
            return []
        order = np.lexsort((np.arange(len(scores)), -scores))
        return [(int(i), float(scores[i])) for i in order[:k] if scores[i] > 0]


class BM25:
    """양의 log1p IDF를 사용하는 BM25를 직접 계산한다."""

    def __init__(self, texts, k1=1.5, b=0.75):
        if k1 <= 0 or not 0 <= b <= 1:
            raise ValueError("BM25 설정 오류")
        self.vectorizer = NumpyTfidf(tf="raw", norm=None).fit(texts)
        self.counts = self.vectorizer.counts_
        self.length = self.counts.sum(axis=1)
        self.average = float(self.length.mean())
        self.k1 = k1
        self.b = b
        n = len(texts)
        df = np.count_nonzero(self.counts, axis=0)
        self.idf = np.log1p((n - df + 0.5) / (df + 0.5))

    def scores(self, query):
        terms = np.flatnonzero(self.vectorizer.counts([query])[0])
        if len(terms) == 0:
            return np.zeros(len(self.counts))
        frequency = self.counts[:, terms]
        normalizer = self.k1 * (
            1 - self.b + self.b * self.length[:, None] / max(self.average, 1e-12)
        )
        denominator = frequency + normalizer
        score = np.divide(
            self.idf[terms] * frequency * (self.k1 + 1),
            denominator,
            out=np.zeros_like(frequency),
            where=denominator > 0,
        )
        return score.sum(axis=1)

    def top(self, query, k=5):
        scores = self.scores(query)
        order = np.lexsort((np.arange(len(scores)), -scores))
        return [(int(i), float(scores[i])) for i in order[:k] if scores[i] > 0]
