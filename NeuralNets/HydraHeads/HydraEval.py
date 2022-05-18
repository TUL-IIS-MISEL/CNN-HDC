from NeuralNets.HydraHeads.Hydra import *
from NeuralNets.HydraHeads.HydraKaist import (
    prepare_sequential_dataset as prepare_sequential_dataset_kaist,
)
import NeuralNets.HydraHeads.HydraKaist

from comet_ml import API, APIExperiment, ExistingExperiment
from comet_ml.api import Parameter
from io import BytesIO

import torch
from sklearn.metrics import balanced_accuracy_score
from torchinfo import summary

# %%
def evaluate_reference(
    base: HydraEncoder,
    head: Type[ReferenceHead],
    experiment: comet_ml.Experiment,
    window_size: int = 3,
    sequential_dataset_func: Callable = prepare_sequential_dataset,
    epochs: int = 10,
    batch_size: int = 128,
):
    experiment.add_tag(head.__name__)
    dataset_test, dataset_train, dataset_classes = sequential_dataset_func(window_size)

    model = head(window_size, len(dataset_classes))
    model.cuda()

    summ = summary(model, (1, window_size, 4 * 4 * 4), device="cuda", depth=5)
    print(summ)
    experiment.set_model_graph(f"{model.__repr__()}\n{summ}")
    total_params = 0
    for _, para in model.named_parameters():
        total_params += torch.numel(para.data)
    print(f"Total parameters : {total_params}")
    experiment.log_parameter("ParameterCount", total_params)

    experiment.log_parameter("batch_size", batch_size)
    workers = min(os.cpu_count(), 16)
    train_loader = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=True,
        num_workers=workers,
    )
    test_loader = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=batch_size,
        pin_memory=True,
        num_workers=workers,
    )
    experiment.log_parameter("train_size", len(dataset_train))
    experiment.log_parameter("test_size", len(dataset_test))

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1.0,
        weight_decay=0.0004,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda epoch: cosine_annealing_with_warmup(
            epoch,
            max_epochs=epochs,
            min_epochs=0,
            lr_max=0.025,
            lr_min=0.01,
            warmup=max(1, epochs // 5),
        ),
    )

    def process_epoch(experiment, loader, optimizer, epoch):
        experiment.set_epoch(epoch)

        targets = []
        outputs = []
        for image_batch, label_batch in tqdm(
            loader, desc=f"{'Train'if optimizer is not None else 'Test'} {epoch}: "
        ):
            experiment.set_step(epoch)

            batch_size = label_batch.shape[0]

            image_batch = image_batch.cuda()
            label_batch = label_batch.cuda()

            label_batch = label_batch - 1
            image_batch = image_batch / 255.0
            latent_batch = base(image_batch.view(-1, 1, 32, 32)).reshape(
                batch_size, -1, 4 * 4 * 4
            )

            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)

            predict = model(latent_batch)
            loss = model.loss(label_batch, predict)

            experiment.log_metric("loss", loss.item())

            targets.append(label_batch.detach().cpu())
            outputs.append(predict.detach().cpu())

            if optimizer is not None:
                loss.backward()
                optimizer.step()

        targets = torch.cat(targets).numpy()
        outputs = torch.cat(outputs, dim=0).numpy().argmax(axis=1)
        experiment.log_confusion_matrix(
            targets, outputs, file_name=f"{type(model).__name__}_matrix.json"
        )
        experiment.log_metric(
            f"{type(model).__name__}_J_score",
            balanced_accuracy_score(targets, outputs, adjusted=True),
        )
        experiment.log_histogram_3d(
            targets, f"{type(model).__name__}_targets", step=epoch
        )
        experiment.log_histogram_3d(
            outputs, f"{type(model).__name__}_outputs", step=epoch
        )

    for epoch in range(epochs):
        with experiment.validate() as _, torch.no_grad() as _:
            process_epoch(experiment, test_loader, None, epoch)

        with experiment.train():
            process_epoch(experiment, train_loader, optimizer, epoch)

        scheduler.step()

    model.eval()
    with experiment.test() as _, torch.no_grad() as _:
        process_epoch(experiment, test_loader, None, epochs)

    # Log model
    model_name = f"{type(model).__name__}_{experiment.get_key()}"
    torch.save(model.state_dict(), f"{model_name}.pth")
    experiment.log_model(model_name, f"./{model_name}.pth")
    os.remove(f"./{model_name}.pth")


# %%
if __name__ == "__main__":
    # Set GPU config
    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = str(0)

    comet_api = API()

    for params in powerset(
        [
            # (HydraClassifierConv, {"num_classes": 2}, 2.0),
            (HydraClassifierConv, {"num_classes": 3}, 2.0),
            (HydraDecoder, {}, 0.02),
            # (HydraClusterer, {"num_classes": 8, "num_pca": 8}, 2.5),
            (HydraClusterer, {"num_classes": 16, "num_pca": 8}, 2.5),
        ]
    ):
        try:
            heads_cls, heads_configs, head_weights = zip(*params)

            base = HydraEncoder()
            heads = [head(**config) for head, config in zip(heads_cls, heads_configs)]
            model = Hydra(base, heads, base_weight=0.05, head_weights=head_weights)
            total_params = 0
            for _, para in model.named_parameters():
                total_params += torch.numel(para.data)

            exp_list = comet_api.query(
                workspace="paluczak",
                project_name="misel-cnn-pytorch",
                query=(
                    (Parameter("ParameterCount") == total_params)
                    # & (comet_ml.api.Metadata("file_name") == "Hydra.py")
                    & (comet_ml.api.Metadata("file_name") == "HydraKaist.py")
                ),
            )
            # summary(model, (1, 1, 32, 32), device="cuda", depth=5)
            for exp in exp_list:
                if not exp.get_parameters_summary("git_id"):
                    continue
                try:
                    model_asset = filter(
                        lambda asset: asset["type"] == "model-element",
                        exp.get_asset_list(),
                    ).__next__()
                except StopIteration:
                    continue
                model_weights = torch.load(
                    BytesIO(exp.get_asset(model_asset["assetId"], return_type="binary"))
                )
                base_dict = {
                    k[5:]: v for k, v in model_weights.items() if k.startswith("base")
                }
                base.load_state_dict(base_dict)
                base.cuda()
                base.eval()

                # for window_size in [3, 8]:
                for window_size in [3, 8, 12]:
                    experiment = APIExperiment(
                        workspace="paluczak",
                        project_name="MISEL_CNN_PyTorch",
                    )
                    experiment.set_filename("HydraEval.py")
                    experiment.log_parameter("parent_key", exp.key)
                    experiment.log_parameter("parent_name", exp.name)
                    experiment.log_parameter("HDC_window_size", window_size)
                    experiment = ExistingExperiment(previous_experiment=experiment.id)
                    for tag in exp.get_tags():
                        experiment.add_tag(tag)
                        if "SPECTRUM" in tag:
                            NeuralNets.HydraHeads.HydraKaist.SPECTRUM = tag.split(" ")[
                                0
                            ]
                    # experiment.add_tag(
                    #     f"{ClustererHydraDatasetHeadComposite.num_cluster} CLUSTER"
                    # )
                    experiment.add_tag("SEQUENCE")
                    evaluate_reference(
                        base,
                        ReferenceHead,
                        experiment,
                        window_size=window_size,
                        # sequential_dataset_func=prepare_sequential_dataset
                        sequential_dataset_func=prepare_sequential_dataset_kaist,
                    )

        except ValueError as err:
            print(err)
            print("Runtime failure, moving on to next configuration")
            raise err
