import math


def precision_at_k(recommended_items, relevant_items, k=10):
    recommended_at_k = recommended_items[:k]

    hits = len(
        set(recommended_at_k).intersection(
            set(relevant_items)
        )
    )

    return hits / k


def recall_at_k(recommended_items, relevant_items, k=10):
    if not relevant_items:
        return 0.0

    recommended_at_k = recommended_items[:k]

    hits = len(
        set(recommended_at_k).intersection(
            set(relevant_items)
        )
    )

    return hits / len(relevant_items)


def hit_rate_at_k(recommended_items, relevant_items, k=10):
    recommended_at_k = recommended_items[:k]

    hit = any(
        item in relevant_items
        for item in recommended_at_k
    )

    return 1.0 if hit else 0.0


def ndcg_at_k(recommended_items, relevant_items, k=10):
    recommended_at_k = recommended_items[:k]

    dcg = 0.0

    for index, item in enumerate(recommended_at_k):
        if item in relevant_items:
            dcg += 1 / math.log2(index + 2)

    ideal_hits = min(
        len(relevant_items),
        k
    )

    idcg = sum(
        1 / math.log2(index + 2)
        for index in range(ideal_hits)
    )

    if idcg == 0:
        return 0.0

    return dcg / idcg