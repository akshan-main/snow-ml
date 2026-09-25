# pylint: disable=C0103
"""
Profiles the snowml-model-ready S3 bucket: how many basins, which regions,
which file schema, what's undocumented relative to docs/Data_Pipeline.md.

Run standalone (needs AWS_PROFILE pointed at a bucket with read access):
    python3 -m snowML.analysis.s3_profile --bucket snowml-model-ready

Everything here is read-only. No files are modified or uploaded.
"""

import argparse
import re
from collections import Counter

import pandas as pd

MODEL_READY_RE = re.compile(r"model_ready_huc(\d+)(_ucla)?\.csv(\.csv)?$")


def list_bucket(bucket, profile=None, prefix=""):
    """Return every object key under prefix in bucket, as a flat list.

    boto3 is imported here rather than at module level so that classify_keys,
    summarize's grouping logic, and the self-check can run (and be unit
    tested) without boto3 installed or any network access.
    """
    import boto3  # pylint: disable=import-outside-toplevel
    session = boto3.Session(profile_name=profile) if profile else boto3.Session()
    s3 = session.client("s3")
    keys = []
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        keys.extend(obj["Key"] for obj in page.get("Contents", []))
    return keys


def classify_keys(keys):
    """
    Split bucket keys into model-ready basin files vs other datasets, and pull
    the HUC id and region (first two digits) out of each model-ready filename.
    """
    basins, other = [], []
    for k in keys:
        m = MODEL_READY_RE.search(k)
        if m:
            huc = m.group(1)
            basins.append({"key": k, "huc": huc, "huc_len": len(huc),
                           "region": huc[:2], "is_ucla": bool(m.group(2))})
        else:
            other.append(k)
    return pd.DataFrame(basins), other


def summarize(bucket, profile=None):
    """Full profile: basin counts by region, dataset prefixes outside
    model-ready, and a flag for HUC ids that aren't the expected 12 digits."""
    keys = list_bucket(bucket, profile=profile)
    basins, other = classify_keys(keys)

    plain = basins[~basins.is_ucla]
    unique_hucs = plain.drop_duplicates("huc")

    other_prefixes = Counter(k.split("/")[0] for k in other if "/" in k)

    return {
        "total_objects": len(keys),
        "unique_basins": len(unique_hucs),
        "basins_by_region": unique_hucs.region.value_counts().to_dict(),
        "nonstandard_huc_length": sorted(
            unique_hucs[unique_hucs.huc_len != 12].huc.tolist()),
        "other_dataset_prefixes": dict(other_prefixes),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bucket", default="snowml-model-ready")
    ap.add_argument("--profile", default=None)
    args = ap.parse_args()

    report = summarize(args.bucket, profile=args.profile)
    print(f"total objects:        {report['total_objects']}")
    print(f"unique basins:        {report['unique_basins']}")
    print(f"by region:            {report['basins_by_region']}")
    print(f"non-12-digit HUCs:    {report['nonstandard_huc_length']}")
    print(f"other dataset prefixes: {report['other_dataset_prefixes']}")


# ----------------------------------------------------------------------------
# self-check: exercises classify_keys/summarize's grouping logic on a fixed,
# fake key list, so the parsing logic is verified without any network access.

def _self_check():
    fake_keys = [
        "model_ready_huc170103020101.csv",
        "model_ready_huc170103020101_ucla.csv",
        "model_ready_huc170103020102.csv",
        "model_ready_huc160401010101.csv",
        "model_ready_huc180400090201.csv",
        "model_ready_huc1711000501.csv",          # HUC10, 10 digits
        "huc12_snow_classification_master.csv",    # not a basin file
        "snow_cover_era5_land/ERA5_Snow_2020.csv",
        "scf_time_series_partitioned/huc_id=170103020101/part-0.parquet",
        "code/snowml-pilot-123/sourcedir.tar.gz",
    ]
    basins, other = classify_keys(fake_keys)

    plain = basins[~basins.is_ucla]
    assert len(plain) == 5, f"expected 5 plain basin files, got {len(plain)}"
    assert plain.huc.nunique() == 5, "each basin should be counted once"
    assert set(plain.region) == {"17", "16", "18"}, plain.region.tolist()

    nonstandard = plain[plain.huc_len != 12]
    assert list(nonstandard.huc) == ["1711000501"], nonstandard.huc.tolist()

    assert len(other) == 4, f"expected 4 non-basin keys, got {len(other)}"
    prefixes = Counter(k.split("/")[0] for k in other if "/" in k)
    assert prefixes["snow_cover_era5_land"] == 1
    assert prefixes["scf_time_series_partitioned"] == 1
    assert prefixes["code"] == 1

    print("s3_profile self-check passed (no network access used)")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--self-check":
        _self_check()
    else:
        main()
