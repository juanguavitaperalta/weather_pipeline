from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import requests
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("weather_pipeline")


@dataclass(frozen=True)
class OpenMeteoConfig:
    latitude: float
    longitude: float
    timezone: str
    hourly: list[str]
    start_date: str
    end_date: str


def load_config(path: str | Path) -> tuple[OpenMeteoConfig, Path]:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    om = cfg["open_meteo"]
    raw_dir = Path(cfg["paths"]["raw_dir"])

    om_cfg = OpenMeteoConfig(
        latitude=float(om["latitude"]),
        longitude=float(om["longitude"]),
        timezone=str(om["timezone"]),
        hourly=list(om["hourly"]),
        start_date=str(om["start_date"]),
        end_date=str(om["end_date"]),
    )
    return om_cfg, raw_dir


def fetch_open_meteo(cfg: OpenMeteoConfig) -> pd.DataFrame:
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": cfg.latitude,
        "longitude": cfg.longitude,
        "hourly": ",".join(cfg.hourly),
        "start_date": cfg.start_date,
        "end_date": cfg.end_date,
        "timezone": cfg.timezone,
    }

    logger.info("Fetching Open-Meteo data from %s to %s...", cfg.start_date, cfg.end_date)
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    payload = r.json()

    hourly = payload.get("hourly", {})
    if "time" not in hourly:
        raise ValueError("Unexpected API response: missing 'hourly.time'")

    df = pd.DataFrame(hourly)
    df["time"] = pd.to_datetime(df["time"])
    return df


def save_raw(df: pd.DataFrame, raw_dir: Path) -> Path:
    raw_dir.mkdir(parents=True, exist_ok=True)
    out = raw_dir / "open_meteo_hourly.csv"
    df.to_csv(out, index=False)
    logger.info("Saved raw data to %s", out.as_posix())
    return out


def main() -> None:
    cfg, raw_dir = load_config(Path("configs/config.yaml"))
    df = fetch_open_meteo(cfg)
    save_raw(df, raw_dir)


if __name__ == "__main__":
    main()
