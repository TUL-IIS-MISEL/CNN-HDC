from typing import Union, Dict, List

import torch
import torch.nn.functional as F

HyperVector = torch.Tensor
HyperIndex = int


class CIM:
    def __init__(
        self,
        name: str,
        low: float,
        high: float,
        *,
        steps: int = 16,
        width: int = 10_000,
    ) -> None:
        self.name = name
        self.width = width

        self.bin_width = (high - low) / steps
        self.bins = torch.linspace(
            low + self.bin_width / 2, high - self.bin_width / 2, steps
        )

        vectors = [HDC._generate_vector(self.width)]
        indexes = torch.arange(self.width)
        flip_mask = torch.ones(self.width).bool()
        bits_to_flip = int(self.width / 2 / (steps - 1))

        for idx in range(1, len(self.bins)):
            candidates = indexes[flip_mask]
            shuffled = candidates[torch.randperm(len(candidates))]
            chosen = shuffled[:bits_to_flip]

            flip_mask[chosen] = False

            flips = torch.ones_like(vectors[idx - 1])
            flips[chosen] = -1
            next_vector = vectors[idx - 1] * flips

            vectors.append(next_vector)

        self.vectors = torch.stack(vectors)

    def map(self, value: float) -> HyperVector:
        assert (
            (self.bins[0] - self.bin_width / 2)
            <= value
            <= (self.bins[-1] + self.bin_width / 2)
        ), "Value must be within range."

        idx = (value - self.bins).abs().argmin()
        return self.vectors[idx]

    def get(self, vector: HyperVector) -> float:
        similarity = torch.Tensor(HDC.cosine(vector, self.vectors))
        idx = similarity.argmax()
        return float(self.bins[idx].item())

    def values(self):
        for idx, (bin, vector) in enumerate(zip(self.bins, self.vectors)):
            yield idx, bin, vector


class SDM:
    def __init__(
        self, name: str, centroids: torch.Tensor, *, width: int = 10_000
    ) -> None:
        self.name = name
        self.width = width

        self.epsilon = 1e-3
        # pretending to use 4-bit ints
        self.cosine_min = 0
        self.cosine_max = 16

        centroids = torch.as_tensor(centroids)
        self.value_min = torch.min(centroids)
        self.value_max = torch.max(centroids)

        self.centroids = centroids
        self.centroids_cnt = len(centroids)

        self.seed = HDC._generate_vector(self.width)
        self.vectors = torch.stack(
            [HDC.permute(self.seed, i) for i in range(self.centroids_cnt)]
        )

    def _similarity(self, value: torch.Tensor):
        value = torch.as_tensor(value)
        similarities = F.cosine_similarity(
            self.centroids, value.tile((self.centroids_cnt, 1))
        )
        similarities_rescaled = self.cosine_min + (
            (self.cosine_max - self.cosine_min) * (similarities + 1) / 2
        )
        return torch.round(similarities_rescaled).int()

    def encode(self, value: torch.Tensor):
        vectors_scaled = self._similarity(value).view(-1, 1) * self.vectors
        return HDC.sum(*vectors_scaled)

    def values(self):
        for idx, vector in enumerate(self.vectors):
            yield idx, idx, vector


class HDC:
    def __init__(self, *, width: int = 10_000, size: int = 0) -> None:
        self.width = width

        if size != 0:
            self._vectors: HyperVector = torch.stack(
                [HDC._generate_vector(self.width) for _ in range(size)]
            )

            self._symbols = [f"{idx}" for idx in range(size)]
        else:
            self._vectors = torch.zeros((1, self.width))
            self._symbols = ["NULL"]

        self.maps: Dict[str, Union[CIM, SDM]] = dict()

    def size(self):
        return len(self._vectors)

    def _get_exact_vector_match(self, vector: HyperVector) -> HyperIndex:
        matching = (self._vectors == vector).all(dim=-1).int()
        if matching.sum() != 1:
            raise KeyError("Vector does not exist.")
        return HyperIndex(int(matching.argmax().item()))

    def _get_exact_symbol_match(self, symbol: str) -> HyperIndex:
        if symbol not in self._symbols:
            raise KeyError(f"Symbol '{symbol}' does not exist.")
        return HyperIndex(self._symbols.index(symbol))

    def memorize(self, symbol: str, vector: HyperVector) -> None:
        if symbol in self._symbols:
            raise KeyError(f"Symbol '{symbol}' already exists.")

        try:
            # Statistically this should NEVER succeed for a new random vector
            self._get_exact_vector_match(vector)
            raise KeyError("Hyperdimensional vector already exists.")
        except KeyError:
            pass

        self._symbols.append(symbol)
        self._vectors = torch.cat((self._vectors, vector.unsqueeze(0)), dim=0)

    def memorize_map(self, map: Union[CIM, SDM]):
        for idx, _, vector in map.values():
            self.memorize(f"{map.name}_{idx}", vector)
        self.maps[map.name] = map

    def new(self, label: str) -> HyperVector:
        candidate = HDC._generate_vector(self.width)
        self.memorize(label, candidate)
        return candidate

    def get_symbol(self, vector: HyperVector) -> str:
        idx = self._get_exact_vector_match(vector)
        return self._symbols[idx]

    def get_vector(self, symbol: str) -> HyperVector:
        idx = self._get_exact_symbol_match(symbol)
        return self._vectors[idx]

    def recover(self, vector: HyperVector) -> HyperVector:
        similarity = torch.Tensor(HDC.cosine(vector, self._vectors))
        idx = similarity.argmax()
        return self._vectors[idx]

    # --------------------------------------------------------------------------
    # ----------------------------------STATIC----------------------------------
    # --------------------------------------------------------------------------
    @staticmethod
    def _generate_vector(width) -> HyperVector:
        seed = torch.rand(width)
        base = (seed > 0.5).int()
        return base + (base - 1)

    @staticmethod
    def _assert_matching_length(*vectors: HyperVector):
        assert all(
            vec.shape[-1] == vectors[0].shape[-1] for vec in vectors
        ), "Vector lengths should match."

    @staticmethod
    def hamming(a: HyperVector, b: HyperVector) -> Union[float, torch.Tensor]:
        HDC._assert_matching_length(a, b)
        similarity = HDC.cosine(a, b)
        similarity = 1 - similarity  # range furthest to closest: [2 , 0]
        similarity /= 2  # range furthest to closest: [1 , 0]
        return similarity

    @staticmethod
    def cosine(a: HyperVector, b: HyperVector) -> Union[float, torch.Tensor]:
        HDC._assert_matching_length(a, b)
        # range furthest to closest: [-width , width]
        similarity = (a @ b.T).float()
        similarity /= a.shape[-1]  # range furthest to closest: [-1 , 1]
        return similarity

    @staticmethod
    def permute(vector: HyperVector, step: int = 1) -> HyperVector:
        return torch.roll(vector, step, dims=-1)

    @staticmethod
    def sum(*vectors: HyperVector) -> HyperVector:
        assert len(vectors) >= 2, "Addition requires at least two elements."
        HDC._assert_matching_length(*vectors)
        is_even = (len(vectors) % 2) == 0

        result = torch.zeros_like(vectors[0])
        for val in vectors:
            result = result + val

        # Tiebreaker as described in Rahimi et al. 2019
        if is_even:
            result = result + (vectors[0] * vectors[-1])

        return torch.sign(result)

    @staticmethod
    def negate(vector: HyperVector) -> HyperVector:
        return -1 * vector

    @staticmethod
    def sequence(*vectors: HyperVector) -> HyperVector:
        assert len(vectors) >= 2, "Sequence requires at least two elements."
        HDC._assert_matching_length(*vectors)
        return HDC.sum(
            *[HDC.permute(val, idx) for idx, val in enumerate(reversed(vectors))]
        )


# ------------------------------------------------------------------------------
# ------------------------------------VERIFY------------------------------------
# ------------------------------------------------------------------------------
# this should probably eventually transition to unittest
if __name__ == "__main__":
    HD = HDC(size=42)
    # add custom symbols
    a = HD.new("a")
    b = HD.new("b")
    c = HD.new("c")
    d = HD.new("d")
    e = HD.new("e")
    f = HD.new("f")
    # use existing symbols
    x = HD.get_vector("1")
    y = HD.get_vector("2")
    z = HD.get_vector("3")
    # CIM
    cim = CIM("test", 0.0, 1.0, steps=16)
    HD.memorize_map(cim)
    # SDM
    sdm = SDM("centroid", torch.randn(11, 8))
    HD.memorize_map(sdm)

    assert HDC.hamming(a, b) == HDC.hamming(
        b, a
    ), "Hamming distance should be symmetrical"
    assert HDC.cosine(a, b) == HDC.cosine(b, a), "Cosine distance should be symmetrical"

    assert all(
        HDC.sum(a, b, c, d) == HDC.sum(a, b, c, d)
    ), "Summation ties should be broken in repeatable manner"

    assert (
        abs(HDC.cosine(cim.map(0.0), cim.map(1.0))) < 0.01
    ), "Edge values from CIM should be orthogonal"

    assert HDC.cosine(cim.map(0.0), cim.map(0.25)) > HDC.cosine(
        cim.map(0.0), cim.map(0.5)
    ), "HD similarity should fall as value distance increases"

    assert (
        HDC.hamming(a, a) == 0
    ), "Hamming distance between point and itself should be zero"

    assert HDC.cosine(a, a) == 1, "Cosine between point and itself should be one"

    assert all(a * b == b * a), "Multiplication should commute"

    assert all(a * a == 1), "Multiplication should be its own inverse"

    assert all(
        ((c * a) * (c * b)) == (a * b)
    ), "Multiplication should preserve distance"

    assert HDC.hamming(a, b) == HDC.hamming(
        HDC.permute(a), HDC.permute(b)
    ), "Permuation should maintain distance"
    assert HDC.cosine(a, b) == HDC.cosine(
        HDC.permute(a), HDC.permute(b)
    ), "Permuation should maintain distance"

    assert all((x * a) * x == a), "Variable binding should be invertible"

    assert all(
        HD.recover(HDC.sum(x * a, y * b, z * c) * x) == a
    ), "Variable should be recoverable from record"

    assert all(
        HD.recover(
            (HDC.sum(x * a, y * b, z * c) * HDC.sum(a * d, b * e, c * f)) * x,
        )
        == d
    ), "Variable should be recoverable even after substitution"

    assert all(
        HD.recover(
            HDC.sum(HDC.sequence(a, b, c), HDC.negate(HDC.sequence(a, b))),
        )
        == c
    ), "Next element of sequence should be recoverable"

    for i in range(len(sdm.centroids)):
        encoded = sdm.encode(sdm.centroids[i])
        weights = sdm._similarity(sdm.centroids[i])
        similarities = HDC.cosine(encoded, sdm.vectors)
        assert (
            i == similarities.argmax()
        ), "Encoded value should be most similar to closest centroid"

    print("All checks passed")
