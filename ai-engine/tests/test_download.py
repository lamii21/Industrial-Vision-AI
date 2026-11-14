import zipfile

import pytest

from ivai.data.download import DownloadError, extract_neu_archive, md5sum


def make_zip(path, members: dict[str, bytes]):
    with zipfile.ZipFile(path, "w") as zf:
        for name, data in members.items():
            zf.writestr(name, data)
    return path


def test_extract_flattens_and_filters(tmp_path):
    archive = make_zip(
        tmp_path / "a.zip",
        {
            "train/train/images/crazing_1.jpg": b"img1",
            "valid/valid/images/scratches_2.jpg": b"img2",
            "train/train/labels/crazing_1.txt": b"0 0.5 0.5 0.1 0.1",
            "train/train/labels.cache": b"pickle - must not be extracted",
        },
    )
    out = tmp_path / "out"
    counts = extract_neu_archive(archive, out)
    assert counts == {"images": 2, "labels": 1}
    assert (out / "images" / "scratches_2.jpg").read_bytes() == b"img2"
    assert not list(out.rglob("*.cache"))


def test_zip_slip_rejected(tmp_path):
    archive = make_zip(tmp_path / "evil.zip", {"../../evil.jpg": b"x"})
    with pytest.raises(DownloadError):
        extract_neu_archive(archive, tmp_path / "out")


def test_duplicate_names_rejected(tmp_path):
    archive = make_zip(tmp_path / "d.zip", {"a/x_1.jpg": b"1", "b/x_1.jpg": b"2"})
    with pytest.raises(DownloadError):
        extract_neu_archive(archive, tmp_path / "out")


def test_md5(tmp_path):
    p = tmp_path / "f"
    p.write_bytes(b"abc")
    assert md5sum(p) == "900150983cd24fb0d6963f7d28e17f72"
