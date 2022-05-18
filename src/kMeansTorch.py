import torch


class KMeansTorch:
    def __init__(
        self,
        n_clusters=8,
        *,
        max_iter=300,
    ) -> None:
        self.n_clusters = n_clusters
        self.max_iter = max_iter

        self._is_fitted = False

    def _assert_parameters(self):
        if self.max_iter <= 0:
            raise ValueError(f"max_iter should be > 0, got {self.max_iter} instead.")
        if self.n_clusters <= 2:
            raise ValueError(
                f"n_clusters should be >= 2, got {self.n_clusters} instead."
            )

    def fit(self, X: torch.TensorType):
        init_idx = torch.randperm(X.shape[0], device=X.device)[: self.n_clusters]
        self.cluster_centers_ = X[init_idx]
        self.labels_ = torch.zeros(X.shape[0], device=X.device)
        for i in range(self.max_iter):
            new_labels = self.predict(X)

            if torch.all(new_labels == self.labels_):
                break

            self.labels_ = new_labels

            for n in range(self.n_clusters):
                mask = self.labels_ == n
                samples = X[mask]
                self.cluster_centers_[n] = torch.mean(samples, dim=0)

            self._is_fitted = True

            return self

    def predict(self, X):
        """Get labels for samples in X"""
        distances = self.transform(X)
        return torch.argmin(distances, dim=-1)

    def fit_predict(self, X):
        """Get labels for samples in X after fitting"""
        return self.fit(X).labels_

    def transform(self, X):
        """Transform X to distance space"""
        return torch.norm(
            X.unsqueeze(-1) - self.cluster_centers_.T.unsqueeze(0).to(X.device),
            dim=1,
        )

    def fit_transform(self, X):
        """Transform X to distance space after fitting"""
        return self.fit(X).transform(X)


if __name__ == "__main__":
    from sklearn.cluster import KMeans
    import time

    # set to roughly match NTU dataset size
    num_samples = int(6e4)
    dim_samples = 8  # 64
    num_clusters = 8
    num_iter = 10

    data = torch.randn(num_samples, dim_samples, device="cpu")
    start = time.time()
    for i in range(num_iter):
        clusterer = KMeansTorch(num_clusters)
        clusterer.fit(data)
    stop = time.time()
    diff = stop - start
    print(f"CPU runtime = {diff/num_iter} s")

    data = torch.randn(num_samples, dim_samples, device="cuda")
    start = time.time()
    for i in range(num_iter):
        clusterer = KMeansTorch(num_clusters)
        clusterer.fit(data)
    stop = time.time()
    diff = stop - start
    print(f"GPU runtime = {diff/num_iter} s")

    data = torch.randn(num_samples, dim_samples, device="cpu").numpy()
    start = time.time()
    for i in range(num_iter):
        clusterer = KMeans(n_clusters=num_clusters, random_state=2021)
        clusterer.fit(data)
    stop = time.time()
    diff = stop - start
    print(f"SKlearn runtime = {diff/num_iter} s")
