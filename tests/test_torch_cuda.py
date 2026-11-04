import pytest


def test_torch_cuda():
    torch = pytest.importorskip(
        "torch",
        reason="torch is not installed",
    )

    print(f"torch version:       {torch.__version__}")
    print(f"cuda version:        {torch.version.cuda}")
    print(f"cuDNN version:       {torch.backends.cudnn.version()}")

    assert torch.cuda.is_available(), "CUDA is expected to be available"

    print(f"cuda available:      {torch.cuda.is_available()}")

    device_count = torch.cuda.device_count()
    print(f"device count:        {device_count}")
    assert device_count > 0

    for i in range(device_count):
        props = torch.cuda.get_device_properties(i)
        print("device i capability:\n", props)
        assert props.major >= 0
        assert props.minor >= 0

    device = torch.device("cuda:0")
    a = torch.randn(1024, 1024, device=device)
    b = torch.randn(1024, 1024, device=device)
    c = a @ b
    torch.cuda.synchronize()

    print(f"computed on: {c.device.type}")
    assert c.device.type == "cuda"
    assert c.shape == (1024, 1024)
    assert torch.isfinite(c).all()
    assert not torch.isnan(c).any()
