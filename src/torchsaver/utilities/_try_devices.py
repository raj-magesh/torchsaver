import functools
import gc
from collections.abc import Callable, Collection
from typing import ParamSpec, TypeVar

import torch
from loguru import logger

P = ParamSpec("P")
R = TypeVar("R")

DEFAULT_DEVICES: list[torch.device] = [
    torch.device(f"cuda:{gpu}") for gpu in range(torch.cuda.device_count())
] + [torch.device("cpu")]


def try_devices(
    func: Callable[P, R],
    devices: Collection[None | torch.device | str] = DEFAULT_DEVICES,
    *,
    current: bool = False,
) -> Callable[P, R]:
    """Try to run a function on any of the provided `devices`, exiting on success.

    For each device provided, the tensor-valued arguments and keyword arguments
    to `func` are copied to the device before the function is run. This allows
    us to write device-agnostic code, since the function can be run on whichever
    device is available at runtime. This function can be used as a decorator.

    Example:
    -------
    ```
    import torch

    n = 10
    x = torch.zeros(n).to("cuda:1")
    y = torch.ones(n).to("cuda:0")

    # to try running the function on all devices

    @try_devices
    def add(x: torch.Tensor, *, y: torch.Tensor) -> torch.Tensor:
        return x + y

    # to try running the function only on "cuda:1" and "cpu"

    @try_devices(devices=("cuda:1", "cpu"))
    def add(x: torch.Tensor, *, y: torch.Tensor) -> torch.Tensor:
        return x + y

    # using the function without a decorator

    z = try_devices(add)(x, y=y)
    ```

    If you need more flexibility in how the function should be applied on
    different devices (for e.g., your function takes in numpy arrays and not
    tensors as inputs), consider using the function `try_environments`.

    Args:
    ----
        func: The function that should be wrapped. devices: GPUs/CPU that the
        function should be tried on, in the order specified. Defaults to all the
        GPUs available and then the CPU (i.e., ["cuda:0", ...,
        f"cuda:{torch.cuda.device_count()}", "cpu"]) current: Whether to try
        running the function with all the tensors on their current devices,
        defaults to False.

    Returns:
    -------
        Wrapped function that can be called.

    """
    if not devices:  # handle case with empty Collection
        devices = DEFAULT_DEVICES
    else:
        devices = [
            None if device is None else torch.device(device) for device in devices
        ]

    if current:
        devices.insert(0, None)

    @functools.wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        contains_tensor_arg = any(
            [isinstance(arg, torch.Tensor) for arg in args]
            + [isinstance(kwarg, torch.Tensor) for kwarg in kwargs.values()],
        )
        if not contains_tensor_arg:
            logger.warning(
                "The function %s does not have any tensor-valued arg/kwarg: `try_devices` is redundant",
                func,
            )

        for device in devices:
            try:
                args_device = [
                    arg.to(device) if isinstance(arg, torch.Tensor) else arg
                    for arg in args
                ]
                kwargs_device = {
                    key: kwarg.to(device) if isinstance(kwarg, torch.Tensor) else kwarg
                    for key, kwarg in kwargs.items()
                }

                return func(
                    *args_device,
                    **kwargs_device,
                )
            except Exception:
                logger.exception("Could not run the function on this device")
                try:
                    del args_device
                    del kwargs_device
                    gc.collect()
                    torch.cuda.empty_cache()
                except Exception:
                    logger.exception("something broke")
                    continue

                continue
        return None

    return wrapper
