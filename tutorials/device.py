from functools import cache

import torch
from triton.runtime import driver


class DeviceProperties:
    def __init__(self, device_index) -> None:
        self._index = device_index
        self._properties = driver.active.utils.get_device_properties(device_index)

    @classmethod
    def new_for_device(cls, device):
        if device.type != "cuda":
            raise ValueError("GPU not found")

        device_index = device.index if device.index is not None else torch.cuda.current_device()
        return cls.new_for_index(device_index)

    @classmethod
    @cache
    def new_for_index(cls, device_index):
        return cls(device_index)

    def streaming_multiprocessor_count(self):
        return self._properties["multiprocessor_count"]

    def max_registers_per_multiprocessor(self):
        return self._properties["max_num_regs"]

    def max_shared_memory_bytes(self):
        return self._properties["max_shared_mem"]

    def warp_size(self):
        return self._properties["warpSize"]
