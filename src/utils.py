# %%
from math import pi as PI  # π
from math import cos as COS  # cos()


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
