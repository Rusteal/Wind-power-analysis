from typing import Dict, Tuple, Set
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys


# =====================
# CONFIG
# =====================
CSV_PATH = Path(r"D:\work\own_projects\wind_data\uk_mean_wind_obs_all_qcv1.csv")
OUTPUT_DIR = Path("figures")
OUTPUT_DIR.mkdir(exist_ok=True)

STATION_ID = "00235"   # example: Stansted (change this)
START_YEAR = 1949
END_YEAR = 2026

CHUNKSIZE = 1_000_000
ROLLING_DAYS = 7


def aggregate_daily_wind(
    csv_path: Path,
    station_id: str,
    start_year: int,
    end_year: int,
    chunksize: int
) -> pd.DataFrame:
    """
    Stream a large CSV and compute daily mean and variance of wind speed
    for a single station.

    Returns
    -------
    pd.DataFrame
        index: date
        columns: ['mean', 'var', 'count']
    """
    daily_stats: Dict[pd.Timestamp, list] = {}
    station_candidates: Set[str] = {station_id}

    if station_id.isdigit():
        station_id_int = str(int(station_id))
        station_candidates.add(station_id_int)
        station_candidates.add(station_id_int.zfill(len(station_id)))

    usecols = ["ob_end_time", "station_id", "mean_wind_speed"]

    for chunk in pd.read_csv(
        csv_path,
        usecols=usecols,
        chunksize=chunksize,
        parse_dates=["ob_end_time"]
    ):
        if not pd.api.types.is_datetime64_any_dtype(chunk["ob_end_time"]):
            chunk["ob_end_time"] = pd.to_datetime(
                chunk["ob_end_time"], errors="coerce"
            )
            chunk = chunk.dropna(subset=["ob_end_time"])
        chunk["station_id"] = chunk["station_id"].astype(str).str.strip()
        chunk = chunk[chunk["station_id"].isin(station_candidates)]

        if chunk.empty:
            continue

        chunk["mean_wind_speed"] = pd.to_numeric(
            chunk["mean_wind_speed"], errors="coerce"
        )
        chunk = chunk.dropna(subset=["mean_wind_speed"])

        chunk["date"] = chunk["ob_end_time"].dt.floor("D")


        for date, g in chunk.groupby("date"):
            year = pd.Timestamp(date).year
            if year < start_year or year > end_year:
                continue

            if date not in daily_stats:
                daily_stats[date] = [0.0, 0.0, 0]

            s = daily_stats[date]
            s[0] += g["mean_wind_speed"].sum()
            s[1] += (g["mean_wind_speed"] ** 2).sum()
            s[2] += len(g)

    records = []
    for date, (sum_x, sum_x2, n) in daily_stats.items():
        if n < 12:  # discard very incomplete days
            continue
        mean = sum_x / n
        var = sum_x2 / n - mean ** 2
        records.append((pd.Timestamp(date), mean, var, n))

    df = pd.DataFrame(
        records,
        columns=["date", "mean", "var", "count"]
    )

    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()

    df["mean"] = df["mean"].astype(float)
    df["var"] = df["var"].astype(float)

    
    return df


def plot_daily_mean(
    df: pd.DataFrame,
    station_id: str,
    output_dir: Path
) -> None:
    """
    Plot daily mean wind speed over time.
    """
    if df.empty:
        print("No daily data to plot (mean). Skipping.")
        return

    fig, ax = plt.subplots(figsize=(18, 5))

    ax.plot(
        df.index,
        df["mean"],
        color="steelblue",
        alpha=0.6
    )

    ax.set_title(f"Daily Mean Wind Speed — Station {station_id}")
    ax.set_ylabel("Wind speed")
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / f"daily_mean_{station_id}.png", dpi=300)
    plt.close()

def plot_daily_variance(
    df: pd.DataFrame,
    station_id: str,
    output_dir: Path
) -> None:
    """
    Plot daily variance of wind speed over time.
    """
    if df.empty:
        print("No daily data to plot (variance). Skipping.")
        return

    fig, ax = plt.subplots(figsize=(18, 5))

    ax.plot(
        df.index,
        df["var"],
        color="darkorange",
        alpha=0.6
    )

    ax.set_title(f"Daily Wind Variance — Station {station_id}")
    ax.set_ylabel("Variance")
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / f"daily_variance_{station_id}.png", dpi=300)
    plt.close()


def plot_monthly_climatology(
    df: pd.DataFrame,
    station_id: str,
    output_dir: Path
) -> None:
    """
    Plot mean daily wind speed by month-of-year.
    """
    if df.empty:
        print("No data for climatology plot. Skipping.")
        return

    monthly = df.copy()
    monthly["month"] = monthly.index.month

    stats = monthly.groupby("month")["mean"].agg(["mean", "std"])

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(stats.index, stats["mean"], marker="o")
    ax.fill_between(
        stats.index,
        stats["mean"] - stats["std"],
        stats["mean"] + stats["std"],
        alpha=0.3
    )

    ax.set_xticks(range(1, 13))
    ax.set_xlabel("Month")
    ax.set_ylabel("Wind speed")
    ax.set_title(f"Monthly Wind Climatology — Station {station_id}")
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / f"monthly_climatology_{station_id}.png", dpi=300)
    plt.close()

def plot_quarterly_mean_time_series(
    df: pd.DataFrame,
    station_id: str,
    output_dir: Path
) -> None:
    """
    Plot quarterly mean wind speed over time.
    """
    if df.empty:
        print("No data for quarterly mean plot. Skipping.")
        return

    quarterly = df["mean"].resample("QE").mean().dropna()
    if quarterly.empty:
        print("No quarterly data after resampling. Skipping.")
        return

    fig, ax = plt.subplots(figsize=(18, 5))
    ax.plot(quarterly.index, quarterly.values, color="teal", linewidth=1.2)

    ax.set_xlabel("Quarter")
    ax.set_ylabel("Quarterly mean wind speed")
    ax.set_title(f"Quarterly Mean Wind Speed (1949-2026) - Station {station_id}")
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / f"quarterly_mean_{station_id}.png", dpi=300)
    plt.close()


if __name__ == "__main__":
    df_daily = aggregate_daily_wind(
        CSV_PATH,
        STATION_ID,
        START_YEAR,
        END_YEAR,
        CHUNKSIZE
    )

    print(df_daily.head())
    print("Number of days:", len(df_daily))

    if df_daily.empty:
        print(
            "No daily data after filtering. "
            "Check STATION_ID and the year range."
        )
        sys.exit(0)

    # plot_daily_mean(df_daily, STATION_ID, OUTPUT_DIR)
    # plot_daily_variance(df_daily, STATION_ID, OUTPUT_DIR)
    # plot_monthly_climatology(df_daily, STATION_ID, OUTPUT_DIR)
    plot_quarterly_mean_time_series(df_daily, STATION_ID, OUTPUT_DIR)
