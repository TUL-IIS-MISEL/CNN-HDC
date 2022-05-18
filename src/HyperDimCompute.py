import torch


class HDC:
    @staticmethod
    def hypervector(width=10_000):
        base = 1 * (torch.rand((width,)) > 0.5).int()
        return base + base - 1

    @staticmethod
    def hyperdictionary(n_elements=32, width=10_000):
        return torch.stack([HDC.hypervector(width) for _ in range(n_elements)])

    @staticmethod
    def _assert_matching_length(*vectors):
        assert all(
            vec.shape[-1] == vectors[0].shape[-1] for vec in vectors
        ), "Vector lengths should match"

    @staticmethod
    def hyperdistance(a, b):
        HDC._assert_matching_length(a, b)
        return (a.shape[-1] - a @ b.T) // 2

    @staticmethod
    def hyperpermute(vector):
        return torch.roll(vector, 1, dims=-1)

    @staticmethod
    def hyperpermutationmatrix(n_rows, width=10_000):
        # TODO
        raise NotImplementedError

    @staticmethod
    def hypersum(*vectors):
        HDC._assert_matching_length(*vectors)
        ternary = torch.zeros_like(vectors[0])
        for val in vectors:
            ternary = ternary + val
        return torch.sign(ternary + ((ternary == 0) * torch.randn(ternary.shape))).int()

    @staticmethod
    def hypersequence(*vectors):
        HDC._assert_matching_length(*vectors)
        result = torch.zeros_like(vectors[0])
        for idx, vector in enumerate(reversed(vectors)):
            for _ in range(idx):
                vector = HDC.hyperpermute(vector)
            result = HDC.hypersum(result, vector)
        return result

    @staticmethod
    def to_symbol(vector: torch.Tensor, dictionary: torch.Tensor):
        match = HDC.hyperdistance(dictionary, vector) == 0
        if not any(match):
            return -1
        else:
            return (1 * match).argmax()

    @staticmethod
    def to_approx_symbol(vector: torch.Tensor, dictionary: torch.Tensor):
        return HDC.hyperdistance(dictionary, vector).argmin()

    @staticmethod
    def to_vector(symbol: int, dictionary: torch.Tensor):
        assert symbol < len(dictionary)
        return dictionary[symbol]

    @staticmethod
    def recover(vector: torch.Tensor, dictionary: torch.Tensor):
        return HDC.to_vector(HDC.to_approx_symbol(vector, dictionary), dictionary)


if __name__ == "__main__":
    dictionary = HDC.hyperdictionary(128)

    a = HDC.to_vector(32, dictionary)
    b = HDC.to_vector(64, dictionary)
    c = HDC.to_vector(96, dictionary)

    d = HDC.to_vector(34, dictionary)
    e = HDC.to_vector(55, dictionary)
    f = HDC.to_vector(89, dictionary)

    x = HDC.to_vector(42, dictionary)
    y = HDC.to_vector(84, dictionary)
    z = HDC.to_vector(126, dictionary)

    assert (
        HDC.hyperdistance(a, a) == 0
    ), "Distance between point ant itself should be zero"

    assert all(a * b == b * a), "Multiplication should commute"

    assert all(a * a == 1), "Multiplication should be its own inverse"

    assert all(
        ((c * a) * (c * b)) == (a * b)
    ), "Multiplication should preserve distance"

    assert HDC.hyperdistance(a, b) == HDC.hyperdistance(
        HDC.hyperpermute(a), HDC.hyperpermute(b)
    ), "Permuation should maintain distance"

    assert all((x * a) * x == a), "Variable binding should be invertible"

    assert all(
        HDC.recover(HDC.hypersum(x * a, y * b, z * c) * x, dictionary) == a
    ), "Variable should be recoverable from record"

    assert all(
        HDC.recover(
            (HDC.hypersum(x * a, y * b, z * c) * HDC.hypersum(a * d, b * e, c * f)) * x,
            dictionary,
        )
        == d
    ), "Variable should be recoverable even after substitution"

    assert all(
        HDC.recover(
            HDC.hypersum(HDC.hypersequence(a, b, c), -1 * HDC.hypersequence(a, b)),
            dictionary,
        )
        == c
    ), "Next element of sequence should be recoverable"
