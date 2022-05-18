import torch


class PCATorch:
    def __init__(
        self,
        n_components=8,
        *,
        whiten=False,
    ) -> None:
        self.n_components = n_components
        self.whiten = whiten

    def fit(self, X: torch.TensorType):
        n_samples = X.shape[0]

        # center data
        self._mean  = torch.mean(X, dim=0)
        X -= self._mean

        # decompose
        U, S, Vh = torch.linalg.svd(X, full_matrices=False)
        # flip eigenvectors' sign to enforce deterministic output
        max_abs_cols = torch.argmax(torch.abs(U), dim=0)
        signs = torch.sign(U[max_abs_cols, range(U.shape[1])])
        U *= signs
        Vh *= signs.unsqueeze(-1)
        # store values
        self._U = U
        self._S = S
        self._Vh = Vh
        self.components_ = Vh[:self.n_components]

        return self

    def transform(self, X):
        """Transform X to PCA space"""
        X -= self._mean
        P = X @ self.components_.T
        return P

    def fit_transform(self, X):
        """Transform X to distance space after fitting"""
        return self.fit(X).transform(X)


if __name__ == "__main__":
    from sklearn.decomposition import PCA
    import time

    # set to roughly match NTU dataset size
    num_samples = int(6e4)
    dim_samples = 64
    num_components = 8
    num_iter = 10

    data = torch.randn(num_samples, dim_samples, device="cpu")
    start = time.time()
    for i in range(num_iter):
        transformer = PCATorch(num_components)
        transformer.fit(data)
    stop = time.time()
    diff = stop - start
    print(f"CPU runtime = {diff/num_iter} s")

    data = torch.randn(num_samples, dim_samples, device="cuda")
    start = time.time()
    for i in range(num_iter):
        transformer = PCATorch(num_components)
        transformer.fit(data)
    stop = time.time()
    diff = stop - start
    print(f"GPU runtime = {diff/num_iter} s")

    data = torch.randn(num_samples, dim_samples, device="cpu").numpy()
    start = time.time()
    for i in range(num_iter):
        transformer = PCA(n_components=num_components, random_state=2021)
        transformer.fit(data)
    stop = time.time()
    diff = stop - start
    print(f"SKlearn runtime = {diff/num_iter} s")
