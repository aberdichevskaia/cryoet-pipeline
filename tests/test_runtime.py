from cryoet_pipeline.runtime import DevicePreference, resolve_device


class _Unavailable:
    @staticmethod
    def is_available() -> bool:
        return False


class _Available:
    @staticmethod
    def is_available() -> bool:
        return True


class _Backends:
    mps = _Unavailable()


class _TorchCpuOnly:
    cuda = _Unavailable()
    backends = _Backends()


class _TorchCuda:
    cuda = _Available()
    backends = _Backends()


class _TorchMps:
    cuda = _Unavailable()


class _MpsBackends:
    mps = _Available()


_TorchMps.backends = _MpsBackends()


def test_resolve_device_prefers_cuda_for_auto() -> None:
    assert resolve_device("auto", torch_module=_TorchCuda) == DevicePreference.CUDA


def test_resolve_device_uses_mps_when_cuda_is_unavailable() -> None:
    assert resolve_device("auto", torch_module=_TorchMps) == DevicePreference.MPS


def test_resolve_device_falls_back_to_cpu() -> None:
    assert resolve_device("auto", torch_module=_TorchCpuOnly) == DevicePreference.CPU


def test_resolve_device_keeps_explicit_choice() -> None:
    assert resolve_device("mps", torch_module=_TorchCuda) == DevicePreference.MPS
