import numpy as np


def sample_features_and_labels_for_visualization(features, labels, n_samples=10, seed=42):

    rng = np.random.default_rng(seed=seed)

    features_sampled = []
    labels_sampled = []
    for label in np.unique(labels):
        idx = np.where(labels == label)[0]
        idx = rng.choice(idx, size=n_samples, replace=False)
        features_sampled.append(features[idx])
        labels_sampled.append(labels[idx])

    return np.concatenate(features_sampled), np.concatenate(labels_sampled)

