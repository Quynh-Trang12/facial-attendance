"""Generate negative/neutral/positive targets from verified FER2013 image selection."""
import argparse
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from facial_attendance.emotion.aggregate import build_dataset


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--selection-root',type=Path,default=ROOT/'data/processed/emotion_6class')
    parser.add_argument('--output',type=Path,default=ROOT/'data/processed/emotion_3class')
    parser.add_argument('--workers',type=int,default=8)
    args=parser.parse_args()
    build_dataset((ROOT/args.selection_root).resolve(),(ROOT/args.output).resolve(),workers=args.workers)


if __name__=='__main__':main()
