def main() -> None:
    print("Hello from opus-magnum-blackbox!")

    try:
        import torch  # noqa: PLC0415
    except ImportError:
        print("Error importing torch")

    print(f"torch version:       {torch.__version__}")
    print(f"cuda version:        {torch.version.cuda}")
    print(f"cuDNN version:       {torch.backends.cudnn.version()}")
    print(f"cuda available:      {torch.cuda.is_available()}")

    device_count = torch.cuda.device_count()
    print(f"device count:        {device_count}")

    for i in range(device_count):
        print(f"device {i} capability: \n", torch.cuda.get_device_properties(0))

    try:
        device = torch.device("cuda:0")
        a = torch.randn(1024, 1024, device=device)
        b = torch.randn(1024, 1024, device=device)
        c = a @ b
        torch.cuda.synchronize()
        print(f"computed on: {c.device.type}")
    except Exception as e:
        print("compute test failed: \n", e)

if __name__=="__main__":
    main()
