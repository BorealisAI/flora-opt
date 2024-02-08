# Copyright (c) 2024-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


from flora_opt.optimizers.torch.flora import Flora

__all__ = {
    "flora": Flora,
}


__kwargs_mapping__ = {
    "flora": lambda kwargs: {
        "lr": kwargs["learning_rate"],
        "eps": (kwargs["eps"], kwargs["eps"]),
        "clip_threshold": kwargs["clipping_threshold"],
        "decay_rate": -kwargs["b2"],
        "beta1": kwargs["b1"],
        "weight_decay": kwargs["weight_decay"],
        "scale_parameter": kwargs["multiply_by_parameter_scale"],
        "relative_step": False,
        "rank": kwargs["tau"],
        "kappa": kwargs["kappa"],
        "factorize_second_moment": kwargs["factorized_second_moment"],
    },
}


def get_optimizer(params, **kwargs):
    try:
        name = kwargs.pop("name")
        optimizer_cls = __all__[name]
    except KeyError as e:
        raise ValueError(f"Unknown optimizer {name}") from e
    return optimizer_cls(params, **__kwargs_mapping__.get(name, lambda x: x)(kwargs))
