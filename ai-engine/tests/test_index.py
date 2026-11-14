import numpy as np
import pytest

from ivai.data.index import parse_label_from_name, pixel_sha256, read_yolo_labels

CLASSES = ("crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches")


@pytest.mark.parametrize(
    "name,label",
    [("crazing_1.jpg", "crazing"), ("rolled-in_scale_300.jpg", "rolled-in_scale"), ("pitted_surface_12.jpg", "pitted_surface")],
)
def test_parse_label(name, label):
    assert parse_label_from_name(name, CLASSES) == label


def test_unknown_label_raises():
    with pytest.raises(ValueError):
        parse_label_from_name("dent_1.jpg", CLASSES)


def test_pixel_hash_depends_on_content_and_shape():
    a = np.zeros((4, 4), np.uint8)
    assert pixel_sha256(a) == pixel_sha256(a.copy())
    assert pixel_sha256(a) != pixel_sha256(np.zeros((2, 8), np.uint8))


def test_read_yolo(tmp_path):
    p = tmp_path / "x.txt"
    p.write_text("5 0.5 0.5 0.2 0.3\n\n2 0.1 0.1 0.05 0.05\n")
    assert read_yolo_labels(p) == [(5, 0.5, 0.5, 0.2, 0.3), (2, 0.1, 0.1, 0.05, 0.05)]
    assert read_yolo_labels(tmp_path / "missing.txt") == []
