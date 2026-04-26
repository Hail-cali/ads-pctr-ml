"""Download Criteo Display Ads dataset from Hugging Face and convert to TSV.

Uses reczoo/Criteo_x1 which has pre-encoded sparse features (integer indices).
Dense features (I1-I13) are already normalized floats.
Sparse features (C1-C26) are already integer-encoded.
"""

import argparse
import logging

from datasets import load_dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DENSE_COLS = [f"I{i}" for i in range(1, 14)]
SPARSE_COLS = [f"C{i}" for i in range(1, 27)]
OUTPUT_COLS = ["label"] + DENSE_COLS + SPARSE_COLS


def main():
    parser = argparse.ArgumentParser(description="Download Criteo dataset from Hugging Face")
    parser.add_argument("--output", default="data/raw/train.txt", help="Output TSV path")
    parser.add_argument("--max-rows", type=int, default=None, help="Max rows to download (None=all)")
    args = parser.parse_args()

    logger.info("Loading reczoo/Criteo_x1 from Hugging Face (this may take a while)...")
    ds = load_dataset("reczoo/Criteo_x1", split="train")

    if args.max_rows:
        ds = ds.select(range(min(args.max_rows, len(ds))))
        logger.info(f"Selected {len(ds)} rows")

    logger.info(f"Total rows: {len(ds)}")
    logger.info(f"Converting to TSV: {args.output}")
    df = ds.to_pandas()
    df = df[OUTPUT_COLS]
    df.to_csv(args.output, sep="\t", header=False, index=False)
    logger.info(f"Saved {len(df)} rows to {args.output}")


if __name__ == "__main__":
    main()
