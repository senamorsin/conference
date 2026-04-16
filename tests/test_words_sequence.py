import numpy as np
import torch

from src.words.sequence_model import (
    TemporalWordCNN,
    encode_word_labels,
    ordered_word_labels,
    reshape_flat_word_features,
)


def test_reshape_flat_word_features_builds_time_major_tensor() -> None:
    flat = np.arange(1430, dtype=np.float32)
    sequence = reshape_flat_word_features(flat)

    assert sequence.shape == (12, 110)
    assert sequence[0, 0] == 0
    assert sequence[-1, -1] == 1319


def test_ordered_word_labels_follows_project_label_order() -> None:
    labels = ordered_word_labels(["WHO", "HELLO", "NO"])
    assert labels == ("HELLO", "NO", "WHO")


def test_encode_word_labels_matches_label_indices() -> None:
    labels = ("HELLO", "NO", "WHO")
    encoded = encode_word_labels(["WHO", "HELLO"], labels)
    assert encoded.tolist() == [2, 0]


def test_temporal_word_cnn_returns_logits_for_each_class() -> None:
    model = TemporalWordCNN(input_dim=110, hidden_dim=32, dropout=0.1, num_classes=4)
    batch = torch.randn(3, 12, 110)
    logits = model(batch)

    assert logits.shape == (3, 4)
