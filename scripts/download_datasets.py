#!/usr/bin/env python
"""
Download datasets for BiomeML.

Usage:
    python scripts/download_datasets.py --list
    python scripts/download_datasets.py --dataset cmd
    python scripts/download_datasets.py --dataset cmd --check
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.datasets import load_dataset, list_datasets, get_dataset_info


def list_available():
    print("Available Datasets:")
    print("-" * 40)
    for name in list_datasets():
        info = get_dataset_info(name)
        print(f"\n{name.upper()}: {info['name']}")
        print(f"  {info['description']}")
        if 'conditions' in info:
            print(f"  Conditions: {', '.join(str(c) for c in info['conditions'][:5])}")


def download(dataset_name: str, data_dir: str = "data"):
    config = {'data_dir': data_dir, 'dataset_name': dataset_name,
              'abundance_file': 'merged_abundance.tsv', 'metadata_file': 'merged_metadata.tsv',
              'biom_file': 'AGP.data.biom', 'phylogeny_file': 'phylogeny.nwk',
              'min_reads_per_sample': 1000, 'min_feature_prevalence': 0.001,
              'target_column': 'disease', 'case_values': ['1'], 'control_values': ['0']}
    ds = load_dataset(dataset_name, config)
    ds.download()


def check(dataset_name: str, data_dir: str = "data"):
    config = {'data_dir': data_dir, 'dataset_name': dataset_name,
              'abundance_file': 'merged_abundance.tsv', 'metadata_file': 'merged_metadata.tsv',
              'biom_file': 'AGP.data.biom', 'phylogeny_file': 'phylogeny.nwk',
              'min_reads_per_sample': 1000, 'min_feature_prevalence': 0.001,
              'target_column': 'disease', 'case_values': ['1'], 'control_values': ['0']}
    ds = load_dataset(dataset_name, config)
    paths = []
    for attr in ['biom_path', 'abundance_path', 'metadata_path', 'phylogeny_path']:
        if hasattr(ds, attr):
            p = getattr(ds, attr)
            status = "FOUND" if p.exists() else "MISSING"
            paths.append((attr, p, status))
            print(f"  {attr}: {p} [{status}]")
    if all(s == "FOUND" for _, _, s in paths):
        print("\nAll files found.")
    else:
        print("\nSome files missing. Run download instructions.")


def main():
    parser = argparse.ArgumentParser(description="BiomeML Dataset Tool")
    parser.add_argument("--list", "-l", action="store_true")
    parser.add_argument("--dataset", "-d", choices=list_datasets())
    parser.add_argument("--check", "-c", action="store_true")
    parser.add_argument("--data-dir", default="data")
    args = parser.parse_args()

    if args.list or not args.dataset:
        list_available()
        return
    if args.check:
        check(args.dataset, args.data_dir)
    else:
        download(args.dataset, args.data_dir)


if __name__ == "__main__":
    main()
