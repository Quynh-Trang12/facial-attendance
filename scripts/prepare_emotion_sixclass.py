"""Build six-class FER2013 without manual relabeling."""
import argparse
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from facial_attendance.config import load_paths
from facial_attendance.emotion.prepare_sixclass import build_dataset


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=ROOT/'configs/project.toml')
    parser.add_argument('--data-root',type=Path)
    parser.add_argument('--output',type=Path,default=ROOT/'data/processed/emotion_6class')
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--validation-fraction',type=float,default=.2)
    parser.add_argument('--workers',type=int,default=8)
    args=parser.parse_args();config=(ROOT/args.config).resolve()
    paths=load_paths(config,root=ROOT)
    source=(ROOT/args.data_root).resolve() if args.data_root else paths.emotion_data
    build_dataset(source,(ROOT/args.output).resolve(),seed=args.seed,
                  validation_fraction=args.validation_fraction,workers=args.workers,config=config)


if __name__=='__main__':main()
