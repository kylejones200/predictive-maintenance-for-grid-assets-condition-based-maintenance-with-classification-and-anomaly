"""Notebook steps (auto-split)."""


def sort_descending_by_prediction_range() -> None:
    interesting_units.sort(key=lambda x: x[1], reverse=True)
    top_units = interesting_units[:5]
    for unit_id, score, mean, std in top_units:
        print(f"Unit {unit_id} | Δ = {score:.3f} | μ = {mean:.3f} | σ = {std:.3f}")
