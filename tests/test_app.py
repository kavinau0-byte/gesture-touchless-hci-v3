from gesture_hci.app import parse_args


def test_parse_args_camera_defaults():
    args = parse_args([])
    assert args.fps == 30
    assert args.width == 640
    assert args.height == 480


def test_parse_args_camera_overrides():
    args = parse_args(["--fps", "60", "--width", "1280", "--height", "720"])
    assert args.fps == 60
    assert args.width == 1280
    assert args.height == 720
