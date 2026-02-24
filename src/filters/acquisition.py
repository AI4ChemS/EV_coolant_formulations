
from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd
import requests

PUBCHEM_BASE = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"


# -----------------------------
# Data model (Stage 1 schema)
# -----------------------------
@dataclass(frozen=True)
class PubChemRecord:
    cid: int
    smiles: Optional[str] = None
    name: Optional[str] = None
    formula: Optional[str] = None
    molecular_weight: Optional[float] = None


# -----------------------------
# Helpers
# -----------------------------
def chunked(seq: List[int], size: int) -> Iterable[List[int]]:
    for i in range(0, len(seq), size):
        yield seq[i : i + size]


def _pubchem_get_json(
    url: str,
    session: requests.Session,
    timeout: Tuple[float, float] = (10.0, 60.0),
    max_retries: int = 6,
    base_sleep: float = 0.5,
) -> dict:
    """
    Robust GET with retry + exponential backoff.
    """
    last_err: Optional[str] = None
    for attempt in range(max_retries):
        try:
            resp = session.get(url, timeout=timeout)
        except requests.RequestException as e:
            last_err = str(e)
            time.sleep(base_sleep * (2**attempt))
            continue

        if resp.status_code == 429:
            retry_after = resp.headers.get("Retry-After")
            if retry_after is not None:
                try:
                    sleep_s = float(retry_after)
                except ValueError:
                    sleep_s = base_sleep * (2**attempt)
            else:
                sleep_s = base_sleep * (2**attempt)
            time.sleep(sleep_s)
            last_err = f"HTTP 429 rate-limited"
            continue

        if 500 <= resp.status_code < 600:
            time.sleep(base_sleep * (2**attempt))
            last_err = f"HTTP {resp.status_code} server error"
            continue

        if resp.status_code != 200:
            raise RuntimeError(f"HTTP {resp.status_code} from PubChem: {resp.text[:300]}")

        return resp.json()

    raise RuntimeError(f"Failed after {max_retries} retries. Last error: {last_err}. URL={url}")


def fetch_pubchem_properties_batch(
    cids: List[int],
    session: requests.Session,
    properties: Tuple[str, ...] = ("IsomericSMILES", "IUPACName", "MolecularFormula", "MolecularWeight"),
) -> Dict[int, PubChemRecord]:
    """
    Fetch properties for a batch of CIDs. Returns dict keyed by CID.
    PubChem may omit invalid/nonexistent CIDs, so missing keys are expected.
    """
    prop_str = ",".join(properties)
    cid_str = ",".join(map(str, cids))
    url = f"{PUBCHEM_BASE}/compound/cid/{cid_str}/property/{prop_str}/JSON"

    data = _pubchem_get_json(url, session=session)
    props = data.get("PropertyTable", {}).get("Properties", [])

    out: Dict[int, PubChemRecord] = {}
    for p in props:
        cid = p.get("CID")
        if cid is None:
            continue

        out[int(cid)] = PubChemRecord(
            cid=int(cid),
            smiles=p.get("IsomericSMILES"),
            name=p.get("IUPACName"),
            formula=p.get("MolecularFormula"),
            molecular_weight=p.get("MolecularWeight"),
        )

    return out


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def _save_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


# -----------------------------
# Main entry function
# -----------------------------
def run_acquisition(
    start_cid: int,
    end_cid: int,
    out_dir: str | Path,
    *,
    batch_size: int = 400,
    checkpoint_every_batches: int = 2500,  # 2500*400 ~ 1,000,000 attempted per checkpoint
    polite_sleep_s: float = 0.2,
    timeout: Tuple[float, float] = (10.0, 60.0),
    max_retries: int = 6,
    base_sleep: float = 0.5,
    properties: Tuple[str, ...] = ("IsomericSMILES", "IUPACName", "MolecularFormula", "MolecularWeight"),
) -> Path:
    """
    Download PubChem properties for CIDs in [start_cid, end_cid] inclusive.

    Outputs in out_dir:
      - material_bank_<start>_<end>_ckpt_batch<N>.parquet (periodic)
      - material_bank_<start>_<end>_final.parquet
      - material_bank_<start>_<end>_manifest.json

    Returns:
      Path to final parquet.
    """
    if start_cid < 1 or end_cid < start_cid:
        raise ValueError(f"Invalid CID range: start_cid={start_cid}, end_cid={end_cid}")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Manifest for reproducibility + resuming/debugging
    manifest_path = out_dir / f"{start_cid}_{end_cid}_manifest.json"
    manifest = {
        "start_cid": start_cid,
        "end_cid": end_cid,
        "batch_size": batch_size,
        "checkpoint_every_batches": checkpoint_every_batches,
        "polite_sleep_s": polite_sleep_s,
        "timeout": list(timeout),
        "max_retries": max_retries,
        "base_sleep": base_sleep,
        "properties": list(properties),
        "created_at_unix": time.time(),
        "failed_batches": [],
        "missing_cids": [],
        "rows_collected": 0,
    }
    _write_json(manifest_path, manifest)

    session = requests.Session()

    cid_list = list(range(start_cid, end_cid + 1))
    batches = list(chunked(cid_list, batch_size))

    rows: List[PubChemRecord] = []
    missing_cids: List[int] = []

    for bi, batch in enumerate(batches, start=1):
        try:
            batch_map = fetch_pubchem_properties_batch(batch, session=session, properties=properties)
        except Exception as e:
            manifest["failed_batches"].append(
                {"batch_index": bi, "first_cid": batch[0], "last_cid": batch[-1], "error": str(e)}
            )
            _write_json(manifest_path, manifest)
            if polite_sleep_s:
                time.sleep(polite_sleep_s)
            continue

        for cid in batch:
            if cid not in batch_map:
                missing_cids.append(cid)

        rows.extend(batch_map.values())

        if bi % checkpoint_every_batches == 0:
            df_ckpt = pd.DataFrame([asdict(r) for r in rows]).sort_values("cid")
            df_ckpt = df_ckpt.drop(columns=["cid"])
            ckpt_path = out_dir / f"{start_cid}_{end_cid}_ckpt_batch{bi}.parquet"
            _save_parquet(df_ckpt, ckpt_path)

            manifest["missing_cids"] = missing_cids
            manifest["rows_collected"] = int(df_ckpt.shape[0])
            _write_json(manifest_path, manifest)

            print(f"[checkpoint] {ckpt_path.name} | rows={df_ckpt.shape[0]:,} | missing={len(missing_cids):,}")

        if polite_sleep_s:
            time.sleep(polite_sleep_s)

        if bi % 50 == 0:
            print(f"[progress] batches {bi}/{len(batches)} | rows={len(rows):,} | missing={len(missing_cids):,}")

    df_final = pd.DataFrame([asdict(r) for r in rows]).sort_values("cid")
    df_final = pd.DataFrame([asdict(r) for r in rows]).sort_values("cid")
    df_final = df_final.drop(columns=["cid"])
    final_path = out_dir / f"{start_cid}_{end_cid}.parquet"
    _save_parquet(df_final, final_path)

    manifest["missing_cids"] = missing_cids
    manifest["rows_collected"] = int(df_final.shape[0])
    _write_json(manifest_path, manifest)

    print(f"[done] {final_path.name} | rows={df_final.shape[0]:,} | missing={len(missing_cids):,}")
    return final_path


# -----------------------------
# Optional: CLI wrapper
# -----------------------------
def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="PubChem Stage-1 acquisition: CID -> SMILES/name/formula/MW")
    p.add_argument("--start-cid", type=int, required=True)
    p.add_argument("--end-cid", type=int, required=True)
    p.add_argument("--out-dir", type=str, required=True)

    p.add_argument("--batch-size", type=int, default=400)
    p.add_argument("--checkpoint-every-batches", type=int, default=2500)
    p.add_argument("--polite-sleep-s", type=float, default=0.2)

    p.add_argument("--max-retries", type=int, default=6)
    p.add_argument("--base-sleep", type=float, default=0.5)

    return p


def main() -> None:
    args = _build_argparser().parse_args()
    run_acquisition(
        start_cid=args.start_cid,
        end_cid=args.end_cid,
        out_dir=args.out_dir,
        batch_size=args.batch_size,
        checkpoint_every_batches=args.checkpoint_every_batches,
        polite_sleep_s=args.polite_sleep_s,
        max_retries=args.max_retries,
        base_sleep=args.base_sleep,
    )


if __name__ == "__main__":
    main()