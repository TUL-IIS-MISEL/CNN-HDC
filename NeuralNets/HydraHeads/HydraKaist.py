# %%
from enum import Enum

import comet_ml

# %%
import torch

# %%
from Datasets.KAIST_event import KaistSquaresDataset
from Datasets.KAIST_sequence import KaistSequencesDataset

from NeuralNets.HydraHeads.Hydra import (
    HydraDataset,
    HydraDatasetHeadComposite,
    ClassifierHydraDatasetHeadComposite,
    ClustererHydraDatasetHeadComposite,
    HydraBase,
    HydraHead,
    HydraEncoder,
    HydraClassifier,
    HydraDecoder,
    HydraClusterer,
    Hydra,
    powerset,
    run_experiment,
    evaluate_clustering,
    evaluate_hdc,
)

# %%
from typing import List, Tuple


# %%
class HydraDatasetKaist(HydraDataset):
    def __init__(self, spectrum="visible") -> None:
        super().__init__(event_dataset=None)
        self.head_datasets: List[HydraDatasetHeadComposite] = []
        dataset_path = (
            "/data/HDD_4_B/MISEL-datasets/KAIST_rgbt-ped-detection/data/kaist-rgbt"
        )
        self.spectrum = spectrum
        trainset_raw = KaistSquaresDataset(
            path=dataset_path, split="train", spectrum=spectrum
        )
        trainset_raw[0]  # force label load
        self.trainset = torch.utils.data.Subset(
            trainset_raw,
            torch.nonzero(
                torch.tensor(
                    trainset_raw.labels[:, 0] != KaistPedObjectType.UNKNOWN.value
                ),
            )
            .squeeze()
            .tolist(),
        )
        testset_raw = KaistSquaresDataset(
            path=dataset_path, split="test", spectrum=self.spectrum
        )
        testset_raw[0]  # force label load
        self.testset = torch.utils.data.Subset(
            testset_raw,
            torch.nonzero(
                torch.tensor(
                    testset_raw.labels[:, 0] != KaistPedObjectType.UNKNOWN.value
                ),
            )
            .squeeze()
            .tolist(),
        )
        self.event_dataset = torch.utils.data.ConcatDataset(
            (self.trainset, self.testset)
        )

    def train_test_split(
        self,
        ratio: float = 0.5,
        generator: torch.Generator = torch.Generator().manual_seed(2021),
    ) -> Tuple[torch.utils.data.Subset, torch.utils.data.Subset]:
        self.train_indexes = torch.arange(0, len(self.trainset)).tolist()
        self.test_indexes = torch.arange(
            len(self.trainset), len(self.trainset) + len(self.testset)
        ).tolist()

        return (
            torch.utils.data.Subset(self, self.train_indexes),
            torch.utils.data.Subset(self, self.test_indexes),
        )

    def get_base_item(self, idx: int) -> torch.Tensor:
        image, _ = self.event_dataset[idx]
        image = image / 255.0
        return image


# %%
from Datasets.KAIST import KaistPedObjectType


def _remove_unknowns(dataset_raw: KaistSequencesDataset):
    dataset_labels = torch.tensor([label[0][0] for label in dataset_raw.labels])
    dataset_train = torch.utils.data.Subset(
        dataset_raw,
        torch.nonzero(
            dataset_labels != KaistPedObjectType.UNKNOWN.value,
        )
        .squeeze()
        .tolist(),
    )

    return dataset_train


SPECTRUM = "lwir"  # has to be at module lever for external averride


def prepare_sequential_dataset(window_size):
    dataset_train_raw = KaistSequencesDataset(
        path="/datasets/KAIST_rgbt-ped-detection/data/kaist-rgbt",
        window_size=window_size,
        spectrum=SPECTRUM,
        split="train",
    )
    dataset_test_raw = KaistSequencesDataset(
        path="/datasets/KAIST_rgbt-ped-detection/data/kaist-rgbt",
        window_size=window_size,
        spectrum=SPECTRUM,
        split="test",
    )
    dataset_train = _remove_unknowns(dataset_train_raw)
    dataset_test = _remove_unknowns(dataset_test_raw)
    return (
        dataset_test,
        dataset_train,
        Enum(
            "KaistSubset",
            [
                (a.name, a.value)
                for a in KaistPedObjectType
                if a.value != KaistPedObjectType.UNKNOWN.value
            ],
        ),
    )


# %%
if __name__ == "__main__":
    # Set GPU config
    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = str(1)

    SPECTRUM = "lwir"
    LETTERS = 16

    for length in [3, 8, 12]:
        for params in powerset(
            [
                (HydraClassifier, {"num_classes": 3}, 2.0),
                (HydraDecoder, {}, 0.02),
                (HydraClusterer, {"num_classes": LETTERS, "num_pca": 8}, 2.5),
            ]
        ):
            try:
                heads, configs, weights = zip(*params)
                experiment, model = run_experiment(
                    HydraEncoder,
                    heads,
                    configs,
                    weights,
                    HydraDatasetKaist(spectrum=SPECTRUM),
                )
                experiment.add_tag("KAIST")
                experiment.log_parameter("letters", LETTERS)
                experiment.log_parameter("spectrum", SPECTRUM)
                with torch.no_grad():
                    model.cpu()
                    model.update_children_location()
                    # ClustererHydraDatasetHeadComposite.num_cluster = 8
                    dataset = HydraDatasetKaist(spectrum=SPECTRUM)
                    dataset.add_composite(ClassifierHydraDatasetHeadComposite)
                    dataset.add_composite(ClustererHydraDatasetHeadComposite)
                    clusterer = evaluate_clustering(experiment, model, dataset)
                    evaluate_hdc(
                        experiment,
                        model.base,
                        clusterer,
                        window_size=length,
                        reduction=0.0,
                        sequential_dataset_func=prepare_sequential_dataset,
                    )
            except ValueError as err:
                print(err)
                print("Runtime failure, moving on to next configuration")
