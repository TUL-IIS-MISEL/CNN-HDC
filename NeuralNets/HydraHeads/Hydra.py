# %%
from enum import Enum
from collections import Counter

import comet_ml

# %%
import torch
import torch.nn as nn
import torch.nn.functional as F

# %%
import pytorch_lightning as pl

# %%
from pytorch_lightning.loggers import CometLogger

# %%
from tqdm import tqdm

# %%
from math import pi as PI  # π
from math import cos as COS  # π

# %%
import os

# %%
from typing import Any, List, Tuple, Type, Sequence, Optional, Iterable, Dict, Callable


# %%
class HydraBase(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self.device = "cpu"

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def loss(self, latent: torch.Tensor, epoch: int = 1) -> torch.Tensor:
        """
        Loss for latent space conditioning.
        """
        raise NotImplementedError

    def log_metrics(self, latent: torch.Tensor, logger: CometLogger) -> None:
        """
        If there are any metrics to be logged, log them here
        """
        pass

    def log_final_metrics(self, latent: torch.Tensor, logger: CometLogger) -> None:
        """
        If there are any metrics to be logged, log them here
        """
        pass


class HydraEncoder(HydraBase):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.model = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Conv2d(8, 8, kernel_size=3, stride=2),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Conv2d(8, 8, kernel_size=3, stride=2),
            nn.Conv2d(8, 4, kernel_size=3, stride=1, bias=False),
            nn.BatchNorm2d(4),
            nn.Tanh(),
            nn.Flatten(start_dim=1),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.model(images)

    def _quantization_loss(self, latent: torch.Tensor) -> torch.Tensor:
        # 4 bits + sign = 16 values on each side of OY axis
        return torch.sum(1 + torch.cos((2 * 16 - 1) * PI * latent))

    def _distribution_loss(self, latent: torch.Tensor) -> torch.Tensor:
        # try to spread the values as much as possible in the allowed range
        # TODO: verify if this is correct
        target_distr = torch.distributions.uniform.Uniform(-1.0, 1.0)
        target_vals = target_distr.sample(latent.shape).to(latent.device)

        return F.kl_div(
            F.log_softmax(latent, dim=0), F.softmax(target_vals, dim=0), reduction="sum"
        )

    def loss(self, latent: torch.Tensor, epoch: int = 1) -> torch.Tensor:
        quantization_loss = self._quantization_loss(latent)
        # kl_loss = self._distribution_loss(latent)
        return quantization_loss  # + kl_loss

    def log_metrics(self, latent: torch.Tensor, logger: CometLogger) -> None:
        # log only first four latents as images each epoch
        latent_img = latent[:8].squeeze().numpy()
        logger.experiment.log_image(latent_img, name=f"{type(self).__name__}")

        latent = latent.numpy()
        logger.experiment.log_histogram_3d(latent, f"{type(self).__name__}")

    def log_final_metrics(self, latent: torch.Tensor, logger: CometLogger) -> None:
        latent = latent.numpy()
        logger.experiment.log_table(f"{type(self).__name__}.tsv", latent)


# %% [markdown]
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------

# %%
from Datasets.ViratEventClassificationDataset import ViratEventClassificationDataset


class HydraDatasetHeadComposite:
    def __init__(self, parent) -> None:
        self.parent = parent

    def __getitem__(self, idx) -> torch.Tensor:
        raise NotImplementedError

    def on_epoch_start(
        self, base: HydraBase, train_indexes: Sequence[int], test_indexes: Sequence[int]
    ) -> None:
        """
        Override as necessary, also called at the very start of training
        """

    def log_dataset_metric(self, logger: CometLogger) -> None:
        """
        Override as necessary, called after on_epoch_start()
        """


class HydraDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        event_dataset: ViratEventClassificationDataset,
    ) -> None:
        super().__init__()
        self.head_datasets: List[HydraDatasetHeadComposite] = []
        self.event_dataset = event_dataset

    def add_composite(self, composite: Type[HydraDatasetHeadComposite]):
        self.head_datasets.append(composite(self))

    def get_base_item(self, idx: int) -> torch.Tensor:
        image, _ = self.event_dataset[idx]
        image = torch.tensor(image).unsqueeze(0)
        image = image / 255.0
        return image

    def train_test_split(
        self,
        ratio: float = 0.8,
        generator: torch.Generator = torch.Generator().manual_seed(2021),
    ) -> Tuple[torch.utils.data.Subset, torch.utils.data.Subset]:
        dataset_size = len(self.event_dataset)
        train_size = int(dataset_size * ratio)
        test_size = dataset_size - train_size

        trainset, testset = torch.utils.data.random_split(
            dataset=self,
            lengths=(train_size, test_size),
            generator=generator,
        )
        self.train_indexes = trainset.indices
        self.test_indexes = testset.indices

        return trainset, testset

    def __len__(self):
        return len(self.event_dataset)

    def __getitem__(self, idx: int):
        image = self.get_base_item(idx)
        targets = [dataset[idx] for dataset in self.head_datasets]
        return image, targets

    def on_epoch_start(self, base: HydraBase) -> None:
        for head in self.head_datasets:
            head.on_epoch_start(base, self.train_indexes, self.test_indexes)

    def log_dataset_metric(self, logger: CometLogger) -> None:
        for head in self.head_datasets:
            head.log_dataset_metric(logger)


# %% [markdown]
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------

# %%
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import normalized_mutual_info_score
from sklearn.metrics import calinski_harabasz_score
from sklearn.metrics import silhouette_score


class HydraHead(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self.device = "cpu"

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        return self.model(latent)

    def postprocess(self, outputs: torch.Tensor) -> torch.Tensor:
        """
        Called after loss computation to get final result values i.e. converted
         with softmax.
        """
        raise NotImplementedError

    def loss(self, targets: torch.Tensor, outputs: torch.Tensor) -> torch.Tensor:
        """
        Loss for head's output
        """
        raise NotImplementedError

    def log_metrics(
        self, targets: torch.Tensor, outputs: torch.Tensor, logger: CometLogger
    ) -> None:
        """
        If there are any metrics to be logged, log them here
        """
        pass

    def log_final_metrics(
        self, targets: torch.Tensor, outputs: torch.Tensor, logger: CometLogger
    ) -> None:
        """
        If there are any metrics to be logged, log them here
        """
        pass

    def get_dataset_composite(self) -> Type[HydraDatasetHeadComposite]:
        """
        Used to generate the target values for training.
        """
        raise NotImplementedError


class DecoderHydraDatasetHeadComposite(HydraDatasetHeadComposite):
    def __init__(self, parent: HydraDataset) -> None:
        super().__init__(parent)

    def __getitem__(self, idx) -> torch.Tensor:
        image = self.parent.get_base_item(idx)
        return image


class HydraDecoder(HydraHead):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.model = nn.Sequential(
            nn.Unflatten(dim=1, unflattened_size=(4, 4, 4)),
            nn.ConvTranspose2d(4, 8, kernel_size=3, padding=0, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.ConvTranspose2d(8, 8, kernel_size=6, padding=1, stride=2),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.ConvTranspose2d(8, 8, kernel_size=6, padding=1, stride=2),
            nn.ConvTranspose2d(8, 1, kernel_size=7, padding=2),
            # nn.Sigmoid(),
        )

    def postprocess(self, outputs: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(outputs)

    def loss(self, targets: torch.Tensor, outputs: torch.Tensor) -> torch.Tensor:
        loss_encode = F.l1_loss(outputs, targets, reduction="sum")
        loss_hinge = torch.sum(  # Hinge loss -  try to make values more crisp
            0.01
            * F.relu(
                1.0
                - ((torch.sigmoid(outputs) - 0.5) * 2.0) * (((targets - 0.5) > 0) * 1.0)
            )
        )
        return loss_encode  # + loss_hinge

    def get_dataset_composite(self) -> Type[HydraDatasetHeadComposite]:
        return DecoderHydraDatasetHeadComposite

    def log_metrics(
        self, targets: torch.Tensor, outputs: torch.Tensor, logger: CometLogger
    ) -> None:
        # log only first few images each epoch
        targets = targets[:8]
        outputs = outputs[:8]
        for idx, (target, output) in enumerate(zip(targets, outputs)):
            combined = torch.cat((target.T, output.T), dim=0).squeeze().T.numpy()
            logger.experiment.log_image(combined, name=f"{idx}_{type(self).__name__}")

    def log_final_metrics(
        self, targets: torch.Tensor, outputs: torch.Tensor, logger: CometLogger
    ) -> None:
        # log a larger number of images
        for idx, (target, output) in enumerate(zip(targets[:16], outputs[:16])):
            combined = torch.cat((target.T, output.T), dim=0).squeeze().T.numpy()
            logger.experiment.log_image(combined, name=f"{idx}_{type(self).__name__}")


class ClassifierHydraDatasetHeadComposite(HydraDatasetHeadComposite):
    def __init__(self, parent: HydraDataset) -> None:
        super().__init__(parent)

    def __getitem__(self, idx) -> torch.Tensor:
        _, label = self.parent.event_dataset[idx]
        return label - 1


class HydraClassifier(HydraHead):
    def __init__(self, num_classes: int = 2, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.num_classes = num_classes
        self.model = nn.Sequential(
            nn.Linear(4 * 4 * 4, 32),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(16, num_classes),
            # nn.Softmax(dim=1),
        )

    def postprocess(self, outputs: torch.Tensor) -> torch.Tensor:
        return F.softmax(outputs, dim=-1)

    def loss(self, targets: torch.Tensor, outputs: torch.Tensor) -> torch.Tensor:
        targets = targets.long()
        # logoutputs = F.log_softmax(outputs, dim=-1)
        # loss = F.nll_loss(logoutputs, targets, reduction="sum")

        loss = F.cross_entropy(outputs, targets, reduction="sum")
        return loss

    def get_dataset_composite(self) -> Type[HydraDatasetHeadComposite]:
        return ClassifierHydraDatasetHeadComposite

    def log_metrics(
        self, targets: torch.Tensor, outputs: torch.Tensor, logger: CometLogger
    ) -> None:
        targets = targets.numpy()
        outputs = outputs.numpy().argmax(axis=1)
        logger.experiment.log_confusion_matrix(
            targets, outputs, file_name=f"{type(self).__name__}_matrix.json"
        )
        logger.experiment.log_metric(
            f"{type(self).__name__}_J_score",
            balanced_accuracy_score(targets, outputs, adjusted=True),
        )
        logger.experiment.log_histogram_3d(targets, f"{type(self).__name__}_targets")
        logger.experiment.log_histogram_3d(outputs, f"{type(self).__name__}_outputs")

    def log_final_metrics(
        self, targets: torch.Tensor, outputs: torch.Tensor, logger: CometLogger
    ) -> None:
        combined = torch.cat(
            (targets.unsqueeze(-1), outputs, outputs.argmax(dim=-1).unsqueeze(-1)),
            dim=-1,
        ).numpy()
        logger.experiment.log_table(
            f"{type(self).__name__}.tsv",
            combined,
            headers=[
                "TARGET",
                *[f"PRED_{i}" for i in range(outputs.shape[-1])],
                "PREDICT",
            ],
        )


class HydraClassifierConv(HydraClassifier):
    def __init__(self, num_classes: int = 2, *args, **kwargs) -> None:
        super().__init__(num_classes, *args, **kwargs)
        self.model = nn.Sequential(
            nn.Unflatten(dim=1, unflattened_size=(4, 4, 4)),
            nn.Conv2d(4, 4, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(4),
            nn.ReLU(),
            nn.Flatten(start_dim=1),
            nn.Linear(4 * 4 * 4, 16),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(16, num_classes),
            # nn.Softmax(dim=1),
        )


from sklearn.utils import resample
from sklearn.decomposition import KernelPCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


class ClustererHydraDatasetHeadComposite(HydraDatasetHeadComposite):
    num_cluster = 8
    num_pca = 8

    def __init__(self, parent: HydraDataset) -> None:
        super().__init__(parent)
        self.clusters = torch.zeros(len(self.parent), dtype=int)
        self._kpca: KernelPCA = KernelPCA(
            n_components=self.num_pca,
            kernel="cosine",
            n_jobs=-1,
            random_state=1701,
        )
        self._scaler: StandardScaler = StandardScaler(with_mean=True, with_std=True)
        self._clusterer: KMeans = KMeans(n_clusters=self.num_cluster, random_state=2021)
        self.mui = 0.0

    def __getitem__(self, idx) -> torch.Tensor:
        return self.clusters[idx].item()

    def _fit(self, latent_train: torch.Tensor, use_full: bool = False) -> None:
        latent_train = latent_train.numpy()

        if use_full:
            X = latent_train
        else:
            X = resample(
                latent_train,
                replace=False,
                n_samples=min(20_000, int(0.1 * len(latent_train))),
            )

        # PCA
        X_kpca = self._kpca.fit_transform(X)
        # Whitening
        X_whitened = X_kpca  # TODO
        # Standard score
        X_norm = self._scaler.fit_transform(X_whitened)
        # kMeans
        self._clusterer.fit(X_norm)

    def _predict(self, latent: torch.Tensor):
        X_norm = self._transform(latent)
        # kMeans
        clusters = self._clusterer.predict(X_norm)
        # return cluster assignments
        return torch.from_numpy(clusters)

    def _transform(self, latent):
        X = latent.numpy()
        # PCA
        X_kpca = self._kpca.transform(X)
        # Whitening
        X_whitened = X_kpca  # TODO
        # Standard score
        X_norm = self._scaler.transform(X_whitened)
        return X_norm

    def _get_latent(self, base: HydraBase, indexes: Sequence[int]):
        index_count = len(indexes)
        latent = torch.zeros(index_count, 64, dtype=float, device="cuda")
        chunks = torch.ceil(torch.tensor(index_count / 128)).int()
        positions = torch.arange(index_count).chunk(chunks)
        indexes = torch.tensor(indexes).chunk(chunks)
        for batch_pos, batch_idx in zip(positions, indexes):
            images = torch.stack(
                [self.parent.get_base_item(idx.item()) for idx in batch_idx]
            ).to(base.device)
            latent[batch_pos] = base(images).to(base.device).type(latent.dtype)
        return latent

    def _update_clusters(self, indexes, new_clusters):
        self.clusters[
            torch.tensor(indexes, device=self.clusters.device)
        ] = new_clusters.to(self.clusters.device)

    def on_epoch_start(
        self,
        base: HydraBase,
        train_indexes: Sequence[int],
        test_indexes: Sequence[int],
    ) -> None:
        with torch.no_grad():
            latent_train = self._get_latent(base, train_indexes)
            self._fit(latent_train)
            new_clusters_train = self._predict(latent_train)
            self._update_clusters(train_indexes, new_clusters_train)

            latent_test = self._get_latent(base, test_indexes)
            new_clusters_test = self._predict(latent_test)
            old_clusters = self.clusters[test_indexes].cpu().numpy()
            self._update_clusters(test_indexes, new_clusters_test)

            new_clusters = new_clusters_test.cpu().numpy()
            latent_test = latent_test.cpu().numpy()
            self.mui = normalized_mutual_info_score(old_clusters, new_clusters)
            try:
                self.calinski_harabasz = calinski_harabasz_score(
                    latent_test, new_clusters
                )
                self.silhouette = silhouette_score(latent_test, new_clusters)
            except ValueError:
                self.calinski_harabasz = 0
                self.silhouette = -1

    def log_dataset_metric(self, logger: CometLogger) -> None:
        logger.experiment.log_metric(f"{type(self).__name__}_NMI_score", self.mui)
        logger.experiment.log_metric(
            f"{type(self).__name__}_calinski_harabasz_score",
            self.calinski_harabasz,
        )
        logger.experiment.log_metric(
            f"{type(self).__name__}_silhouette_score", self.silhouette
        )


class HydraClusterer(HydraClassifier):
    def __init__(self, num_classes: int = 8, num_pca: int = 8, *args, **kwargs) -> None:
        super().__init__(num_classes, *args, **kwargs)
        self.num_pca = num_pca

    def get_dataset_composite(self) -> Type[HydraDatasetHeadComposite]:
        composite = ClustererHydraDatasetHeadComposite
        composite.num_cluster = self.num_classes
        composite.num_pca = self.num_pca

        return composite

    def log_metrics(
        self, targets: torch.Tensor, outputs: torch.Tensor, logger: CometLogger
    ) -> None:
        return super().log_metrics(targets, outputs, logger)


# %% [markdown]
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------


# %%
def cosine_annealing_with_warmup(epoch, max_epochs, min_epochs, lr_max, lr_min, warmup):
    assert warmup <= max_epochs
    return lr_min + (lr_max - lr_min) * (
        ((epoch - min_epochs) / warmup) * (min_epochs <= epoch < (min_epochs + warmup))
        + 0.5
        * (
            1
            + COS(
                PI * (epoch - min_epochs - warmup) / (max_epochs - min_epochs - warmup)
            )
        )
        * ((min_epochs + warmup) <= epoch <= max_epochs)
    )


class Hydra(pl.LightningModule):
    def __init__(
        self,
        base: HydraBase,
        heads: List[HydraHead],
        base_weight: float = None,
        head_weights: List[float] = None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.base = base
        self.heads = nn.ModuleList(heads)
        self.train_step_idx = 0
        self.val_step_idx = 0
        self.test_step_idx = 0
        self.dataset: Optional[HydraDataset] = None
        self.base_weight = 1.0 if base_weight is None else base_weight
        if head_weights is None:
            self.head_weights = [1.0 for _ in self.heads]
        else:
            assert len(head_weights) == len(
                heads
            ), "Loss from each head must have a weight associated with it"
            self.head_weights = head_weights

    def forward(self, images: torch.Tensor):
        latent = self.base(images)

        outputs = [head(latent) for head in self.heads]
        if not self.training:
            outputs = [
                head.postprocess(output) for head, output in zip(self.heads, outputs)
            ]

        return latent, outputs

    def process_step(self, batch, batch_idx):
        inputs, targets = batch
        latent, outputs = self(inputs)
        base_loss_scale = 0.5 * (1.0 + self.current_epoch) / self.trainer.max_epochs
        base_loss = self.base.loss(latent, self.current_epoch) * base_loss_scale
        losses = [
            head.loss(target, output)
            for head, target, output in zip(self.heads, targets, outputs)
        ]
        weighted_losses = [
            loss * weight for loss, weight in zip(losses, self.head_weights)
        ]

        loss = sum(weighted_losses) + (self.base_weight * base_loss)

        if self.training:
            outputs = [
                head.postprocess(output) for head, output in zip(self.heads, outputs)
            ]

        self.log_losses(loss, base_loss, losses)

        return loss, targets, latent, outputs

    def on_train_epoch_start(self) -> None:
        assert not (self.dataset is None), "Dataset must be available during training"

        self.dataset.on_epoch_start(self.base)
        self.dataset.log_dataset_metric(self.logger)

    def training_step(self, batch, batch_idx):
        self.logger.experiment.set_epoch(self.current_epoch)
        self.train_step_idx += 1
        self.logger.experiment.set_step(self.train_step_idx)

        with self.logger.experiment.train():
            loss, _, _, _ = self.process_step(batch, batch_idx)

        return {"loss": loss}

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        self.logger.experiment.set_epoch(self.current_epoch)
        self.val_step_idx += 1
        self.logger.experiment.set_step(self.val_step_idx)

        with self.logger.experiment.validate():
            _, targets, latent, outputs = self.process_step(batch, batch_idx)

        targets = [val.detach().cpu() for val in targets]
        latent = [val.detach().cpu() for val in latent]
        outputs = [val.detach().cpu() for val in outputs]

        return targets, latent, outputs

    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        self.logger.experiment.set_epoch(self.trainer.max_epochs)
        self.test_step_idx += 1
        self.logger.experiment.set_step(self.val_step_idx + self.test_step_idx)

        with self.logger.experiment.test():
            _, targets, latent, outputs = self.process_step(batch, batch_idx)

        targets = [val.detach().cpu() for val in targets]
        latent = [val.detach().cpu() for val in latent]
        outputs = [val.detach().cpu() for val in outputs]

        return targets, latent, outputs

    def configure_optimizers(self):
        self.torch_optimizer = torch.optim.Adamax(
            filter(lambda p: p.requires_grad, self.parameters()),
            lr=1.0,
            weight_decay=0.0004,
        )
        self.torch_scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.torch_optimizer,
            lr_lambda=lambda epoch: cosine_annealing_with_warmup(
                epoch,
                max_epochs=self.trainer.max_epochs,
                min_epochs=0,
                lr_max=0.025,
                lr_min=0.01,
                warmup=max(1, self.trainer.max_epochs // 10),
            ),
        )
        return [self.torch_optimizer], [self.torch_scheduler]

    def update_children_location(self):
        self.base.device = self.device
        for head in self.heads:
            head.device = self.device

    @torch.no_grad()
    def training_epoch_end(self, _):
        self.logger.experiment.log_metric(
            "learning_rate", self.torch_scheduler.get_last_lr()
        )

    @torch.no_grad()
    def validation_epoch_end(self, step_outputs: List[List[torch.Tensor]]) -> None:
        targets, latent, outputs = self._reshape_outputs(step_outputs)

        with self.logger.experiment.validate():
            self.base.log_metrics(latent, self.logger)
            for head, target, predict in zip(self.heads, targets, outputs):
                head.log_metrics(target, predict, self.logger)

    @torch.no_grad()
    def test_epoch_end(self, step_outputs: List[List[torch.Tensor]]) -> None:
        targets, latent, outputs = self._reshape_outputs(step_outputs)

        with self.logger.experiment.test():
            self.base.log_final_metrics(latent, self.logger)
            for head, target, predict in zip(self.heads, targets, outputs):
                head.log_final_metrics(target, predict, self.logger)

    @torch.no_grad()
    def _reshape_outputs(self, step_outputs):
        targets, latent, outputs = zip(*step_outputs)

        targets = zip(*targets)
        outputs = zip(*outputs)

        targets = [torch.cat(val) for val in targets]
        outputs = [torch.cat(val) for val in outputs]

        latent = torch.stack([val for batch in latent for val in batch])
        return targets, latent, outputs

    @torch.no_grad()
    def log_losses(
        self,
        loss: torch.Tensor,
        base_loss: torch.Tensor,
        head_losses: List[torch.Tensor],
    ):
        self.logger.experiment.log_metric("loss", loss.detach().cpu().item())
        self.logger.experiment.log_metric(
            f"{type(self.base).__name__}_loss", base_loss.detach().cpu().item()
        )
        for head, loss in zip(self.heads, head_losses):
            self.logger.experiment.log_metric(
                f"{type(head).__name__}_loss", loss.detach().cpu().item()
            )


class ReferenceHead(nn.Module):
    def __init__(
        self, window_len: int = 3, num_classes: int = 2, *args, **kwargs
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.window_len = window_len
        self.model = nn.Sequential(
            nn.Flatten(1),
            nn.Linear(window_len * 4 * 4 * 4, 32),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(16, num_classes),
            # nn.Softmax(dim=1),
        )

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        return self.model(latent)

    def postprocess(self, outputs: torch.Tensor) -> torch.Tensor:
        return F.softmax(outputs, dim=-1)

    def loss(self, targets: torch.Tensor, outputs: torch.Tensor) -> torch.Tensor:
        targets = targets.long()
        # logoutputs = F.log_softmax(outputs, dim=-1)
        # loss = F.nll_loss(logoutputs, targets, reduction="sum")

        loss = F.cross_entropy(outputs, targets, reduction="sum")
        return loss


class ReferenceHeadConv(ReferenceHead):
    def __init__(
        self, window_len: int = 3, num_classes: int = 2, *args, **kwargs
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.window_len = window_len
        self.model = nn.Sequential(
            nn.Conv1d(
                window_len,
                1,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            ),
            nn.BatchNorm1d(1),
            nn.ReLU(),
            nn.Flatten(1),
            nn.Linear(4 * 4 * 4, 32),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(16, num_classes),
            # nn.Softmax(dim=1),
        )


# %% [markdown]
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------

# %%
from itertools import chain, combinations


def powerset(iterable: Iterable):
    """
    powerset([1,2,3]) --> (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)
    from: https://docs.python.org/3/library/itertools.html#itertools-recipes
    Modified to avoid empty sets
    """
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(1, len(s) + 1))


def prepare_dataset():
    event_dataset = ViratEventClassificationDataset(
        path="/datasets/VIRAT/VIRAT_squares_VGA_32"
    )
    return HydraDataset(event_dataset)


# %%
def run_experiment(
    base_cls: Type[HydraBase],
    heads_cls: List[Type[HydraHead]],
    heads_configs: List[Dict[str, Any]],
    head_weights: List[float],
    dataset: HydraDataset,
    epochs: int = 20,
    hydra_cls: Type[Hydra] = Hydra,
) -> Tuple[comet_ml.Experiment, Hydra]:
    comet_logger = CometLogger(project_name="MISEL_CNN_PyTorch")

    # log exact git version
    with open(".git/refs/heads/master", "r") as git_ref:
        comet_logger.experiment.log_parameter("git_id", git_ref.readline())

    # Configure model
    base = base_cls()
    heads = [head(**config) for head, config in zip(heads_cls, heads_configs)]
    model = hydra_cls(base, heads, base_weight=0.05, head_weights=head_weights)
    model.cuda()  # works in-place
    model.update_children_location()

    # Log model structure
    from torchinfo import summary

    summ = summary(model, (1, 1, 32, 32), device="cuda", depth=5)
    comet_logger.experiment.set_model_graph(f"{model.__repr__()}\n{summ}")

    total_params = 0
    for _, para in model.named_parameters():
        total_params += torch.numel(para.data)
    print(f"Total parameters : {total_params}")
    comet_logger.experiment.log_parameter("ParameterCount", total_params)

    # Configure dataset
    for head in heads:
        dataset.add_composite(head.get_dataset_composite())
    trainset, testset = dataset.train_test_split()
    model.dataset = dataset

    workers = min(os.cpu_count(), 16)
    batch_size = 64
    comet_logger.experiment.log_parameter("batch_size", batch_size)
    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=True,
        num_workers=workers,
    )
    comet_logger.experiment.log_parameter("train_size", len(trainset))
    # if any([issubclass(head_cls, HydraClassifier) for head_cls in heads_cls]):
    #     comet_logger.experiment.log_text(
    #         "TRAIN: "
    #         + str(
    #             sorted(
    #                 Counter(
    #                     torch.tensor([label for image, label in trainset])
    #                     .flatten()
    #                     .tolist()
    #                 ).items()
    #             )
    #         )
    #     )

    testloader = torch.utils.data.DataLoader(
        testset,
        batch_size=batch_size,
        pin_memory=True,
        # shuffle=True,
        num_workers=workers,
    )
    comet_logger.experiment.log_parameter("test_size", len(testset))
    # if any([issubclass(head_cls, HydraClassifier) for head_cls in heads_cls]):
    #     comet_logger.experiment.log_text(
    #         "TEST: "
    #         + str(
    #             sorted(
    #                 Counter(
    #                     torch.tensor([label for image, label in testset]).flatten().tolist()
    #                 ).items()
    #             )
    #         )
    #     )

    # Run experiment
    trainer = pl.Trainer(
        gpus=[0],
        max_epochs=epochs,
        logger=comet_logger,
        precision=16,
        num_sanity_val_steps=0,
        reload_dataloaders_every_n_epochs=1,
    )

    trainer.fit(model, trainloader, testloader)
    model.eval()
    trainer.test(model, testloader)

    # Log model
    model_name = f"{type(model).__name__}_{comet_logger.experiment.get_key()}"
    torch.save(model.state_dict(), f"{model_name}.pth")
    comet_logger.experiment.log_model(model_name, f"./{model_name}.pth")
    os.remove(f"./{model_name}.pth")

    return comet_logger.experiment, model


# %%
from sklearn.manifold import TSNE

import plotly.express as px
import pandas as pd


def evaluate_clustering(
    experiment: comet_ml.Experiment, model: Hydra, dataset: HydraDataset
) -> ClustererHydraDatasetHeadComposite:
    _, testset = dataset.train_test_split()

    dataset.on_epoch_start(model.base)

    model.eval()

    latent_vals = []
    label_vals = []
    cluster_vals = []
    for image, (label, cluster) in tqdm(testset):
        if len(image.shape) == 3:
            image = image.unsqueeze(0)
        image = image.to(model.device)
        latent_vals.append(model.base.forward(image).flatten())
        label_vals.append(label)
        cluster_vals.append(cluster)
    latent = torch.stack(latent_vals).cpu().numpy()
    labels = torch.tensor(label_vals).cpu().numpy()
    clusters = torch.tensor(cluster_vals).cpu().numpy()

    print(
        "Clustering done "
        f"unlabelled: {(clusters == -1).sum()}, "
        f"labelled: {(clusters != -1).sum()}"
    )

    X_kpca = dataset.head_datasets[-1]._kpca.transform(latent)
    tsne = TSNE(
        n_components=3,
        metric="correlation",
        n_jobs=-1,
        square_distances=True,
        random_state=1701,
        init="pca",
        learning_rate="auto",
    )
    X_TSNE = tsne.fit_transform(X_kpca)

    df_tsne = pd.DataFrame(
        {
            "x": X_TSNE.T[0],
            "y": X_TSNE.T[1],
            "z": X_TSNE.T[2],
            "cluster": clusters,
            "target": labels,
        }
    )

    experiment.log_table(
        "clustering.csv", df_tsne.groupby("cluster")["cluster"].count()
    )
    experiment.log_table("TSNE.csv", df_tsne)

    fig = px.scatter_3d(
        df_tsne,
        x="x",
        y="y",
        z="z",
        color="cluster",
        symbol="target",
        template="plotly_white",
    )
    fig.update(layout_coloraxis_showscale=False)
    fig.update_layout(height=800)
    experiment.log_html(fig.to_html())

    return dataset.head_datasets[-1]


from NeuralNets.HydraHeads.HyperDimensional import HDC
from Datasets.ViratSequenceClassificationDataset import (
    ViratSequenceClassificationDataset,
)
from Datasets.ViratDataset import ViratObjectType
from sklearn.metrics import roc_auc_score


def gini_impurity(labels):
    gini = 1
    for label in torch.unique(labels):
        proba = torch.sum(labels == label) / torch.numel(labels)
        gini -= proba * proba
    return gini


def prepare_sequential_dataset(window_size: int):
    dataset_raw = ViratSequenceClassificationDataset(
        path="/datasets/VIRAT/VIRAT_sequences_VGA_32",
        window_size=window_size,
    )
    dataset_size = len(dataset_raw)
    train_size = int(dataset_size * 0.8)
    test_size = dataset_size - train_size
    generator = torch.Generator().manual_seed(2021)
    dataset_train, dataset_test = torch.utils.data.random_split(
        dataset=dataset_raw,
        lengths=(train_size, test_size),
        generator=generator,
    )
    return (
        dataset_test,
        dataset_train,
        Enum(
            "ViratSubset",
            [
                (a.name, a.value)
                for a in ViratObjectType
                if (
                    a.value in (ViratObjectType.PERSON.value, ViratObjectType.CAR.value)
                )
            ],
        ),
    )


def _resample(dataset: torch.utils.data.Dataset, reduction: float):
    subset_size = int(len(dataset) * reduction)
    subset, _ = torch.utils.data.random_split(
        dataset,
        (subset_size, len(dataset) - subset_size),
        generator=torch.Generator().manual_seed(42),
    )

    return subset


def _image_to_hd(
    subset,
    encoder: HydraEncoder,
    clusterer: ClustererHydraDatasetHeadComposite,
    HD: HDC,
):
    microsequence_buffer = []
    label_buffer = []
    symbol_buffer = []
    for images, label in tqdm(subset):
        images = torch.as_tensor(images).unsqueeze(1) / 255.0
        encoded = encoder(images)
        symbol = clusterer._predict(encoded)
        hypersequence = HD.sequence(*[HD.get_vector(f"{sym}") for sym in symbol])
        microsequence_buffer.append(hypersequence)
        symbol_buffer.append(symbol)
        label_buffer.append(label)
    microsequences = torch.stack(microsequence_buffer)
    labels = torch.tensor(label_buffer, dtype=int)
    symbols = torch.stack(symbol_buffer)

    return microsequences, labels, symbols


def _image_to_hd_batch(
    subset,
    encoder: HydraEncoder,
    clusterer: ClustererHydraDatasetHeadComposite,
    HD: HDC,
):
    microsequence_buffer = []
    label_buffer = []
    symbol_buffer = []
    for image_batch, label_batch in tqdm(subset):
        image_batch = torch.as_tensor(image_batch) / 255.0
        batch_size, window_size = image_batch.shape[:2]
        image_batch = image_batch.view(-1, 1, 32, 32)
        encoded_batch = encoder(image_batch)
        symbol_batch = clusterer._predict(encoded_batch)
        symbol_batch = symbol_batch.reshape(batch_size, window_size, -1)
        for symbol, label in zip(symbol_batch, label_batch):
            hypersequence = HD.sequence(
                *[HD.get_vector(f"{sym.item()}") for sym in symbol]
            )
            microsequence_buffer.append(hypersequence)
            symbol_buffer.append(symbol)
            label_buffer.append(label)
    microsequences = torch.stack(microsequence_buffer)
    labels = torch.tensor(label_buffer, dtype=int)
    symbols = torch.stack(symbol_buffer)

    return microsequences, labels, symbols


def evaluate_hdc(
    experiment: comet_ml.Experiment,
    encoder: HydraEncoder,
    clusterer: ClustererHydraDatasetHeadComposite,
    window_size: int = 3,
    reduction: float = 0.2,
    sequential_dataset_func: Callable = prepare_sequential_dataset,
    image_sequence_encode_func: Callable = _image_to_hd_batch,
):
    experiment.log_parameter("HDC_window_size", window_size)
    assert 0 <= reduction < 1, "Reduction is a percentage"

    encoder.eval()

    dataset_test, dataset_train, dataset_classes = sequential_dataset_func(window_size)
    if reduction > 0:
        dataset_train = _resample(dataset_train, reduction)
        # dataset_test = _resample(dataset_test, reduction)

    workers = min(os.cpu_count(), 16)

    HD = HDC(width=2**13, size=clusterer.num_cluster)
    train_loader = torch.utils.data.DataLoader(
        dataset_train, batch_size=128, pin_memory=True, num_workers=workers
    )
    test_loader = torch.utils.data.DataLoader(
        dataset_test, batch_size=128, pin_memory=True, num_workers=workers
    )
    X_train, y_train, symbols_train = image_sequence_encode_func(
        train_loader, encoder, clusterer, HD
    )
    X_test, y_test, symbols_test = image_sequence_encode_func(
        test_loader, encoder, clusterer, HD
    )

    experiment.log_parameter("HDC_train_size", len(y_train))
    experiment.log_text(
        "HDC_TRAIN: " + str(sorted(Counter(y_train.flatten().tolist()).items()))
    )
    experiment.log_parameter("HDC_test_size", len(y_test))
    experiment.log_text(
        "HDC_TEST: " + str(sorted(Counter(y_test.flatten().tolist()).items()))
    )

    models = []
    for label in dataset_classes:
        model = X_train[y_train == label.value]
        model = HDC.sum(*model)
        models.append(model)

    similarities = []
    for model in models:
        similarity = HDC.cosine(X_test, model)
        similarities.append(similarity)

    similarities = torch.stack(similarities)
    dataset_class_map = torch.tensor([a.value for a in dataset_classes])
    y_pred = dataset_class_map[similarities.argmax(dim=0)]
    j_score = balanced_accuracy_score(y_test, y_pred, adjusted=True)

    experiment.log_metric("HDC_J_score", j_score)
    experiment.log_confusion_matrix(
        (y_test == dataset_class_map.unsqueeze(-1)).int().argmax(dim=0).numpy(),
        (y_pred == dataset_class_map.unsqueeze(-1)).int().argmax(dim=0).numpy(),
        file_name="HDC_matrix.json",
    )
    for similarity, name in zip(similarities, dataset_classes):
        experiment.log_histogram_3d(
            similarity.numpy().flatten(), f"HDC_{name}_similarity", step=0
        )

    experiment.log_metric(
        "HDC_AUC",
        roc_auc_score(
            (y_test == dataset_class_map.unsqueeze(-1)).int().T.numpy(),
            torch.softmax(similarities, dim=0).T.numpy(),
            average="weighted",
            multi_class="ovo",
        ),
    )

    classes = torch.cat((y_train, y_test)).repeat((window_size, 1)).T.flatten()
    clusters = torch.cat((symbols_train, symbols_test)).flatten()

    df_gini = pd.DataFrame(
        [
            (cluster.item(), gini_impurity(classes[clusters == cluster]).item())
            for cluster in torch.unique(clusters)
        ],
        columns=["symbol", "GINI"],
    )
    experiment.log_table("GINI.csv", df_gini)

    for model, name in zip(models, dataset_classes):
        HD.memorize(f"{name}", model)
    experiment.log_table(
        "HDC_dictionary.tsv", HD._vectors.numpy().T, headers=HD._symbols
    )


# %%
if __name__ == "__main__":
    # Set GPU config
    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = str(1)

    for params in powerset(
        [
            (HydraClassifier, {"num_classes": 2}, 2.0),
            (HydraDecoder, {}, 0.02),
            (HydraClusterer, {"num_classes": 8, "num_pca": 8}, 2.5),
        ]
    ):
        try:
            heads, configs, weights = zip(*params)
            experiment, model = run_experiment(
                HydraEncoder, heads, configs, weights, prepare_dataset()
            )
            experiment.add_tag("REFACTOR")
            with torch.no_grad():
                model.cpu()
                model.update_children_location()
                # ClustererHydraDatasetHeadComposite.num_cluster = 4
                experiment.add_tag(
                    f"{ClustererHydraDatasetHeadComposite.num_cluster} CLUSTER"
                )
                clusterer = evaluate_clustering(experiment, model, prepare_dataset())
                evaluate_hdc(
                    experiment,
                    model.base,
                    clusterer,
                    window_size=3,
                    reduction=0.05,
                    sequential_dataset_func=prepare_sequential_dataset,
                )
        except ValueError as err:
            print(err)
            print("Runtime failure, moving on to next configuration")
