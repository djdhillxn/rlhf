from rlhf.trl_train_reward import _calibration


def test_reward_calibration_bins_confidence_against_empirical_accuracy():
    rows = [
        {"preference_probability": 0.90, "correct": True},
        {"preference_probability": 0.10, "correct": False},
        {"preference_probability": 0.60, "correct": True},
        {"preference_probability": 0.40, "correct": False},
    ]
    bins = _calibration(rows, bins=5)

    assert sum(row["count"] for row in bins) == 4
    assert all(0.5 <= row["mean_confidence"] <= 1.0 for row in bins)
    assert all("mean_predicted_probability" not in row for row in bins)
    high_confidence = max(bins, key=lambda row: row["mean_confidence"])
    assert high_confidence["count"] == 2
    assert high_confidence["empirical_accuracy"] == 0.5
