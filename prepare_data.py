"""
After downloading the Kaggle dataset, organize images into dataset/train/0..4
so that train.py can use flow_from_directory. Handles common zip structures.
"""

import os
import shutil
from pathlib import Path


# Diabetic Retinopathy 224x224 Gaussian Filtered dataset uses these folder names
NAME_TO_INDEX = {
    "No_DR": 0,
    "Mild": 1,
    "Moderate": 2,
    "Severe": 3,
    "Proliferate_DR": 4,
}


def find_train_dirs(dataset_root, exclude_train_dir):
    """Find directory that contains class subdirs 0,1,2,3,4 or No_DR, Mild, etc. Exclude the target train dir."""
    dataset_root = Path(dataset_root)
    exclude = Path(exclude_train_dir).resolve() if exclude_train_dir else None
    for d in dataset_root.rglob("*"):
        if not d.is_dir() or (exclude and Path(d).resolve() == exclude):
            continue
        subdirs = [x.name for x in d.iterdir() if x.is_dir()]
        # Prefer named folders (source from Kaggle); skip if this is our train dir
        if set(NAME_TO_INDEX.keys()) <= set(subdirs):
            return d, NAME_TO_INDEX
        if all(str(i) in subdirs for i in range(5)):
            return d, None
    for d in dataset_root.rglob("*"):
        if not d.is_dir() or (exclude and Path(d).resolve() == exclude):
            continue
        subdirs = [x.name for x in d.iterdir() if x.is_dir()]
        if set(subdirs) >= {"0", "1", "2", "3", "4"}:
            return d, None
    return None, None


def prepare_train_structure(data_dir="dataset"):
    """Ensure dataset/train/0, dataset/train/1, ... exist and contain images."""
    data_dir = Path(data_dir)
    train_dir = data_dir / "train"
    train_dir.mkdir(parents=True, exist_ok=True)
    for i in range(5):
        (train_dir / str(i)).mkdir(exist_ok=True)

    # If train/0 already has images, we're done
    if any((train_dir / "0").iterdir()):
        return str(train_dir)

    # Find where Kaggle unzipped the data (exclude our train dir so we don't use it as source)
    found, name_to_idx = find_train_dirs(data_dir, train_dir)
    if found is None:
        # Maybe images are in dataset/0, dataset/1, ...
        if all((data_dir / str(i)).exists() for i in range(5)):
            for i in range(5):
                for f in (data_dir / str(i)).iterdir():
                    if f.suffix.lower() in (".png", ".jpg", ".jpeg"):
                        shutil.copy2(f, train_dir / str(i) / f.name)
            return str(train_dir)
        return None

    # Copy from found path to dataset/train/0..4
    if name_to_idx is not None:
        # Folder names like No_DR, Mild, Moderate, Severe, Proliferate_DR
        for name, idx in name_to_idx.items():
            src = found / name
            dst = train_dir / str(idx)
            if src.exists():
                for f in src.iterdir():
                    if f.is_file() and f.suffix.lower() in (".png", ".jpg", ".jpeg"):
                        shutil.copy2(f, dst / f.name)
    else:
        for i in range(5):
            src = found / str(i)
            dst = train_dir / str(i)
            if src.exists():
                for f in src.iterdir():
                    if f.is_file() and f.suffix.lower() in (".png", ".jpg", ".jpeg"):
                        shutil.copy2(f, dst / f.name)
    return str(train_dir)


if __name__ == "__main__":
    import sys
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "dataset"
    result = prepare_train_structure(data_dir)
    if result:
        print("Training data ready at:", result)
    else:
        print("Could not find class folders 0..4 under", data_dir)
