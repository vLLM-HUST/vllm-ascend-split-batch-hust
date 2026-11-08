import argparse
import configparser
import tarfile
import zipfile
from pathlib import Path, PurePosixPath


def check(directory):
    directory = Path(directory)
    sdists = list(directory.glob("*.tar.gz"))
    wheels = list(directory.glob("*.whl"))
    if len(sdists) != 1 or len(wheels) != 1:
        raise ValueError("Expected exactly one sdist and wheel")
    with tarfile.open(sdists[0]) as archive:
        members = archive.getmembers()
        if any(not (m.isfile() or m.isdir()) for m in members):
            raise ValueError("Non-regular distribution member")
        source = {
            str(PurePosixPath(m.name).relative_to(PurePosixPath(m.name).parts[0]))
            for m in members
            if m.isfile()
        }
    with zipfile.ZipFile(wheels[0]) as archive:
        wheel = set(archive.namelist())
        entry_point_files = [
            name for name in wheel if name.endswith(".dist-info/entry_points.txt")
        ]
        if len(entry_point_files) != 1:
            raise ValueError("Missing plugin entry points")
        entry_points = configparser.ConfigParser()
        entry_points.read_string(archive.read(entry_point_files[0]).decode("utf-8"))
    for names in (source, wheel):
        for name in names:
            path = PurePosixPath(name)
            if (
                path.is_absolute()
                or ".." in path.parts
                or {"bench", "results", "evidence", "logs"}.intersection(path.parts)
                or path.suffix in {".log", ".jsonl", ".csv", ".tsv"}
            ):
                raise ValueError(f"Evidence or unsafe member in distribution: {name}")
    expected = {
        "src/" + name for name in wheel if name.startswith("vllm_ascend_split_batch/")
    }
    actual = {name for name in source if name.startswith("src/")}
    if not expected or expected != actual:
        raise ValueError("sdist/wheel runtime source inventory differs")
    bundle_group = "vllm_hust.extension_bundles"
    expected_manifests = (
        len(entry_points[bundle_group]) if entry_points.has_section(bundle_group) else 0
    )
    packaged_manifests = len(
        [n for n in wheel if n.endswith("vllm-hust-extension-v0.2.json")]
    )
    if expected_manifests == 0 or packaged_manifests != expected_manifests:
        raise ValueError("Missing extension manifests")
    print(
        "sdist and wheel: evidence-free, matching runtime inventory, manifests present"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    check(parser.parse_args().directory)
