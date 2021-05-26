from Evus import Evus
from sklearn.datasets import make_classification
from collections import Counter
import numpy as np

colors = ["#FFE066", "#4590b0"]
import matplotlib
import matplotlib.pyplot as plt
from CV_experiment import CV_experiment


def create_dataset(n_samples=1000, weights=(0.05, 0.95), n_classes=2,
                   class_sep=1.5, n_clusters=2):
    return make_classification(n_samples=n_samples, n_features=2,
                               n_informative=2, n_redundant=0, n_repeated=0,
                               n_classes=n_classes,
                               n_clusters_per_class=n_clusters,
                               weights=list(weights),
                               class_sep=class_sep, random_state=42)


if __name__ == '__main__':
    X, y = create_dataset()
    print('Initial distribution:', sorted(Counter(y).items()))

    x_resampled, y_resampled = Evus(
        X_train, y_train, alpha=1, t=3, beta=self.beta).resampling()

    print('Resampled distribution:', sorted(Counter(y).items()))
