import logging
from pathlib import Path

import numpy as np
import pandas as pd


# ================== CONFIG ================== #

ROOT = Path(
    r"D:\work\own_projects\wind_data\dap.ceda.ac.uk"
    r"\badc\ukmo-midas-open\data\uk-mean-wind-obs\dataset-version-202507"
)

OUTPUT_CSV =  "uk_mean_wind_obs_all_qcv1.csv"
SAMPLE_CSV =  "uk_mean_wind_obs_sample_qcv1.csv"
LOG_FILE =  "build_wind_dataset.log"

CHUNKSIZE = 100_000          # rows per chunk
SAMPLE_FRAC = 0.01           # fraction of each chunk to add to sample
MAX_SAMPLE_ROWS = 100_000    # final cap on sample size

# Keep only useful columns to save space
USECOLS = [
    "ob_end_time", "id_type", "id", "ob_hour_count",
    "met_domain_name", "version_num", "src_id", "rec_st_ind",
    "mean_wind_dir", "mean_wind_speed",
    "max_gust_dir", "max_gust_speed"
]

# NA marker used by MIDAS
NA_VALUES = ["NA"]

# ============================================ #

def setup_logging():
    """Configure logging to file and console."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(LOG_FILE, mode="w", encoding="utf-8"),
            logging.StreamHandler()
        ],
    )
    logging.info("Logging initialised.")


def find_skiprows(csv_path: Path) -> int:
    """
    Count how many lines to skip so that pandas' header row is the line
    after the one that only contains 'data'.
    """
    skip = 0
    with csv_path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip().lower() == "data":
                break
            skip += 1
    # +1 to also skip the 'data' line itself
    logging.debug(f"{csv_path.name}: skiprows={skip + 1}")
    return skip + 1


def iter_data_files(root: Path):
    """
    Yield (csv_path, county, station_id, station_name) for all station files.

    Filenames look like:
    midas-open_uk-mean-wind-obs_dv-202507_antrim_01449_lisnafillan_qcv-1_1967.csv
    """
    # use qc-version-1 only; change to 'qc-version-*' if you want both
    pattern = "*/*/qc-version-*/*.csv"
    for csv_path in root.glob(pattern):
        name = csv_path.stem  # without .csv
        parts = name.split("_")

        county = None
        station_id = None
        station_name = None

        try:
            # parts: [midas-open, uk-mean-wind-obs, dv-202507, county, id, station, qcv-1, year]
            county = parts[3]
            station_id = parts[4]
            station_name = parts[5]
        except Exception as e:
            logging.warning(f"Could not parse metadata from filename {name}: {e}")

        yield csv_path, county, station_id, station_name


def build_master_and_sample():
    """Stream all qc-version-1 files into one CSV and build a random sample."""
    first_chunk = True
    sample_parts = []

    for csv_path, county, station_id, station_name in iter_data_files(ROOT):
        logging.info(
            f"Processing file: {csv_path} "
            f"(county={county}, station_id={station_id}, station_name={station_name})"
        )

        try:
            skiprows = find_skiprows(csv_path)
        except Exception as e:
            logging.error(f"Failed to determine skiprows for {csv_path}: {e}")
            continue

        try:
            chunk_iter = pd.read_csv(
                csv_path,
                skiprows=skiprows,
                chunksize=CHUNKSIZE,
                na_values=NA_VALUES,
                low_memory=False,
                on_bad_lines="warn",  # skip malformed rows but log a warning
            )
        except Exception as e:
            logging.error(f"Failed to create reader for {csv_path}: {e}")
            continue

        for chunk_idx, chunk in enumerate(chunk_iter, start=1):
            logging.info(
                f"{csv_path.name}: processing chunk {chunk_idx} with {len(chunk)} rows"
            )

            try:
                # Restrict to useful columns if specified
                if USECOLS:
                    # Some files might miss a column; handle gracefully
                    missing = [c for c in USECOLS if c not in chunk.columns]
                    if missing:
                        logging.warning(
                            f"{csv_path.name}: missing columns {missing}; "
                            "they will be filled with NaN."
                        )
                        for c in missing:
                            chunk[c] = np.nan
                    chunk = chunk[USECOLS]

                # Add metadata columns
                chunk["county"] = county
                chunk["station_id"] = station_id
                chunk["station_name"] = station_name

                # Append to big CSV on disk
                mode = "w" if first_chunk else "a"
                header = first_chunk
                chunk.to_csv(
                    OUTPUT_CSV,
                    mode=mode,
                    header=header,
                    index=False,
                )
                first_chunk = False

                # Build random sample in memory
                if SAMPLE_FRAC > 0 and len(chunk) > 0:
                    frac = min(SAMPLE_FRAC, 1.0)
                    sample_chunk = chunk.sample(
                        frac=frac,
                        replace=False,
                        random_state=None,
                    )
                    sample_parts.append(sample_chunk)

            except Exception as e:
                logging.error(f"Error while processing chunk {chunk_idx} of {csv_path}: {e}")

    # Concatenate and downsample the sample
    if sample_parts:
        sample_df = pd.concat(sample_parts, ignore_index=True)
        logging.info(f"Combined raw sample size: {len(sample_df)} rows")

        if len(sample_df) > MAX_SAMPLE_ROWS:
            sample_df = sample_df.sample(
                n=MAX_SAMPLE_ROWS,
                random_state=42
            )
            logging.info(
                f"Downsampled sample to {MAX_SAMPLE_ROWS} rows "
                f"(SAMPLE_FRAC={SAMPLE_FRAC})"
            )

        sample_df.to_csv(SAMPLE_CSV, index=False)
        logging.info(f"Saved sample to {SAMPLE_CSV}")
    else:
        logging.warning("No sample data collected – check patterns / paths.")


if __name__ == "__main__":
    setup_logging()
    logging.info("Starting master CSV + sample build.")
    build_master_and_sample()
    logging.info("Done.")
