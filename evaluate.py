"""직접 작성한 추출 문장과 실제 감성 레이블의 평가."""

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

from bow import NumpyTfidf
from extract import extract, mask_pii
from sentiment import analyze, aspects
from preprocess import preview


def key(item):
    return item["kind"], json.dumps(item["value"], sort_keys=True, ensure_ascii=False)


def evaluate_extraction(output):
    """종류와 정규화 값의 다중집합이 일치하는지 확인한다."""
    cases = json.loads(Path("evaluation/extraction-cases.json").read_text())
    totals = {}
    results = []
    for case in cases:
        expected = Counter(key(item) for item in case["expected"])
        actual_items = extract(case["text"])
        actual = Counter(key(item) for item in actual_items)
        correct = expected & actual
        missing = expected - actual
        extra = actual - expected
        for kind in ("email", "phone", "date", "money", "url"):
            row = totals.setdefault(kind, {"tp": 0, "fp": 0, "fn": 0})
            row["tp"] += sum(n for (k, _), n in correct.items() if k == kind)
            row["fp"] += sum(n for (k, _), n in extra.items() if k == kind)
            row["fn"] += sum(n for (k, _), n in missing.items() if k == kind)
        results.append(
            {**case, "predicted": actual_items, "success": not missing and not extra}
        )
    for row in totals.values():
        row["precision"] = (
            row["tp"] / (row["tp"] + row["fp"]) if row["tp"] + row["fp"] else 0.0
        )
        row["recall"] = (
            row["tp"] / (row["tp"] + row["fn"]) if row["tp"] + row["fn"] else 0.0
        )
    result = {
        "cases": len(cases),
        "standard_cases": 50,
        "scope_challenges": 8,
        "metrics": totals,
        "results": results,
        "masking_example": mask_pii(
            "000000-1000000 / 0000 0000 0000 0000 / 서울특별시 예시구 샘플로 123 / user@example.test / 010-0000-1001"
        ),
    }
    (output / "extraction.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    )
    print(
        json.dumps(
            {
                "cases": len(cases),
                "metrics": totals,
                "failures": [r["id"] for r in results if not r["success"]],
            }
        ),
        flush=True,
    )


def metric(y, prediction):
    return {
        "accuracy": float(accuracy_score(y, prediction)),
        "f1": float(f1_score(y, prediction)),
        "macro_f1": float(f1_score(y, prediction, average="macro")),
        "confusion_matrix": confusion_matrix(y, prediction).tolist(),
    }


def evaluate_sentiment(data, split, output):
    """동일 Test를 고정해 규칙 전후와 라벨 수에 따른 TF-IDF를 비교한다."""
    import matplotlib.pyplot as plt

    records = [json.loads(line) for line in data.read_text().splitlines()]
    indices = json.loads(split.read_text())
    tr = np.array(indices["train"])
    te = np.array(indices["test"])
    labels = np.array([r["label"] for r in records])
    texts = [r["text"] for r in records]
    variants = {}
    predictions = {}
    errors = []
    for name, rules, domain in (
        ("base_dictionary", False, False),
        ("domain_dictionary", False, True),
        ("context_rules", True, True),
    ):
        predicted = np.array(
            [analyze(texts[i], rules=rules, domain=domain)["label"] for i in te]
        )
        variants[name] = metric(labels[te], predicted)
        predictions[name] = predicted
    for pos in np.flatnonzero(predictions["context_rules"] != labels[te])[:15]:
        i = int(te[pos])
        analysis = analyze(texts[i])
        errors.append(
            {
                "id": records[i]["id"],
                "row": i,
                "true": int(labels[i]),
                "predicted": int(predictions["context_rules"][pos]),
                "score": analysis["score"],
                "snippet": preview(texts[i], 700),
                "strong_terms": sorted(
                    analysis["contributions"],
                    key=lambda c: abs(c["score"]),
                    reverse=True,
                )[:10],
            }
        )
    budgets = []
    rng = np.random.default_rng(42)
    positive = rng.permutation(tr[labels[tr] == 1])
    negative = rng.permutation(tr[labels[tr] == 0])
    for n in (10, 100, 1000):
        selected = np.r_[positive[: n // 2], negative[: n // 2]]
        vectorizer = NumpyTfidf(tf="log")
        x = vectorizer.fit_transform([texts[i] for i in selected])
        model = LogisticRegression(C=1, max_iter=2000, random_state=42).fit(
            x, labels[selected]
        )
        predicted = model.predict(vectorizer.transform([texts[i] for i in te]))
        budgets.append(
            {
                "labeled_train_rows": n,
                "tfidf": metric(labels[te], predicted),
                "fixed_rules": variants["context_rules"],
            }
        )
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(
        [r["labeled_train_rows"] for r in budgets],
        [r["tfidf"]["f1"] for r in budgets],
        marker="o",
        label="TF-IDF logistic",
    )
    ax.axhline(
        variants["context_rules"]["f1"], label="fixed lexicon/rules", color="orange"
    )
    ax.set(xscale="log", xlabel="labeled Train rows", ylabel="Test F1", ylim=(0, 1))
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "label-budget.png", dpi=150)
    plt.close(fig)
    examples = [
        "The story is excellent but the acting is terrible.",
        "The music is wonderful. The visual effects are awful.",
        "음식은 매우 좋다. 가격은 비싸다.",
    ]
    result = {
        "test_rows": len(te),
        "train_pool": len(tr),
        "split": "same A4-1 stratified fixed split",
        "variants": variants,
        "label_budgets": budgets,
        "misclassified": errors,
        "aspect_examples": [{"text": text, **aspects(text)} for text in examples],
    }
    (output / "sentiment.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    )
    print(
        json.dumps(
            {
                "variants": variants,
                "label_budgets": budgets,
                "misclassified_saved": len(errors),
            }
        ),
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("task", choices=["extraction", "sentiment"])
    parser.add_argument("--data", type=Path, default=Path("data/reviews.jsonl"))
    parser.add_argument("--split", type=Path, default=Path("data/split.json"))
    parser.add_argument("--output", type=Path, default=Path("outputs"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.task == "extraction":
        evaluate_extraction(args.output)
    else:
        evaluate_sentiment(args.data, args.split, args.output)


if __name__ == "__main__":
    main()
