import sys
import json
import re
import tempfile
import subprocess
from pathlib import Path
from collections import defaultdict

from rdkit import Chem


# =========================================================
# USER CONFIG
# =========================================================
QVINA_PATH = "C:/Users/norep/Desktop/hakaton/qvina02"

RECEPTOR_PATH = "C:/Users/norep/Desktop/hakaton/1XGIprot.pdbqt"

CENTER_X = 22.12
CENTER_Y = 8.08
CENTER_Z = 10.79

SIZE_X = 15.0
SIZE_Y = 15.0
SIZE_Z = 15.0

EXHAUSTIVENESS = 4
NUM_MODES = 3
CPU = 1

PH = 7.0
MAX_CONFORMERS_PER_MOLECULE = 1  

FAILED_PREDICTION = 0.0

WRITE_DEBUG_LOG = True
DEBUG_LOG_FILE = "score_qvina_debug.log"


# =========================================================
# LOGGING
# =========================================================
def log_message(msg: str):
    if not WRITE_DEBUG_LOG:
        return
    try:
        with open(DEBUG_LOG_FILE, "a", encoding="utf-8") as f:
            f.write(msg.rstrip() + "\n")
    except Exception:
        pass


# =========================================================
# HELPERS
# =========================================================
def run_cmd(cmd, cwd=None):
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=cwd
    )


def parse_best_affinity(log_text: str):
    """
    Ищет первую строку таблицы поз:
    1   -8.2   0.000   0.000
    """
    for line in log_text.splitlines():
        m = re.match(r"^\s*1\s+(-?\d+\.\d+)\s+", line)
        if m:
            return float(m.group(1))
    return None


def prepare_input_smi(smiles_list, input_smi: Path):
    with open(input_smi, "w", encoding="utf-8") as f:
        for idx, smi in enumerate(smiles_list):
            f.write(f"{smi} {idx}\n")


def split_sdf_to_files(ligands_sdf: Path, sdf_dir: Path, max_per_source: int):
    """
    Разбивает общий SDF на отдельные sdf-файлы.
    Оставляет максимум max_per_source конформеров на исходную молекулу.
    Имя источника берётся как часть до '_i', например:
    1_i0 -> source = 1
    """
    supplier = Chem.SDMolSupplier(str(ligands_sdf), removeHs=False)
    counts = defaultdict(int)
    kept = []

    for i, mol in enumerate(supplier):
        if mol is None:
            continue

        raw_name = mol.GetProp("_Name") if mol.HasProp("_Name") else f"lig_{i}"
        source = raw_name.split("_i")[0]

        if counts[source] >= max_per_source:
            continue

        out_sdf = sdf_dir / f"{raw_name}.sdf"
        writer = Chem.SDWriter(str(out_sdf))
        writer.write(mol)
        writer.close()

        kept.append((source, raw_name, out_sdf))
        counts[source] += 1

    return kept


def convert_sdf_to_pdbqt(sdf_file: Path, pdbqt_file: Path):
    cmd = [
        "obabel",
        str(sdf_file),
        "-O", str(pdbqt_file),
        "-h",
        "--partialcharge", "gasteiger"
    ]
    result = run_cmd(cmd)
    ok = (result.returncode == 0 and pdbqt_file.exists())
    return ok, result


def dock_single_ligand(ligand_pdbqt: Path, out_pdbqt: Path):
    cmd = [
        QVINA_PATH,
        "--receptor", RECEPTOR_PATH,
        "--ligand", str(ligand_pdbqt),
        "--center_x", str(CENTER_X),
        "--center_y", str(CENTER_Y),
        "--center_z", str(CENTER_Z),
        "--size_x", str(SIZE_X),
        "--size_y", str(SIZE_Y),
        "--size_z", str(SIZE_Z),
        "--exhaustiveness", str(EXHAUSTIVENESS),
        "--num_modes", str(NUM_MODES),
        "--cpu", str(CPU),
        "--out", str(out_pdbqt)
    ]
    result = run_cmd(cmd)
    full_log = (result.stdout or "") + "\n" + (result.stderr or "")
    affinity = parse_best_affinity(full_log)
    return affinity, full_log, result.returncode


def score_smiles_batch(smiles_list):
    """
    Возвращает список сырых docking affinities.
    Чем значение более отрицательное, тем лучше.
    """
    predictions = [FAILED_PREDICTION] * len(smiles_list)

    with tempfile.TemporaryDirectory(prefix="qvina_score_") as tmp:
        workdir = Path(tmp)

        input_smi = workdir / "input_ligands.smi"
        ligands_sdf = workdir / "ligands_3d.sdf"
        sdf_dir = workdir / "ligands_sdf"
        pdbqt_dir = workdir / "ligands_pdbqt"
        dock_dir = workdir / "qvina_out"

        sdf_dir.mkdir(exist_ok=True)
        pdbqt_dir.mkdir(exist_ok=True)
        dock_dir.mkdir(exist_ok=True)

        prepare_input_smi(smiles_list, input_smi)

        cmd_scrub = [
            "scrub.py",
            str(input_smi),
            "-o", str(ligands_sdf),
            "--ph", str(PH)
        ]
        scrub_res = run_cmd(cmd_scrub)

        if scrub_res.returncode != 0 or not ligands_sdf.exists():
            log_message("SCRUB FAILED")
            log_message(scrub_res.stderr[:1000] if scrub_res.stderr else "No stderr")
            return predictions

        kept = split_sdf_to_files(
            ligands_sdf=ligands_sdf,
            sdf_dir=sdf_dir,
            max_per_source=MAX_CONFORMERS_PER_MOLECULE
        )

        if not kept:
            log_message("No valid molecules after SDF splitting.")
            return predictions

        prepared = []
        for source, raw_name, sdf_file in kept:
            pdbqt_file = pdbqt_dir / f"{raw_name}.pdbqt"
            ok, obabel_res = convert_sdf_to_pdbqt(sdf_file, pdbqt_file)
            if not ok:
                log_message(f"OBABEL FAILED for {raw_name}")
                log_message(obabel_res.stderr[:1000] if obabel_res.stderr else "No stderr")
                continue
            prepared.append((source, raw_name, pdbqt_file))

        if not prepared:
            log_message("No ligands prepared to PDBQT.")
            return predictions

        best_by_source = {}

        for source, raw_name, pdbqt_file in prepared:
            out_pdbqt = dock_dir / f"{raw_name}_out.pdbqt"
            affinity, full_log, returncode = dock_single_ligand(pdbqt_file, out_pdbqt)

            if affinity is None:
                log_message(f"DOCK FAILED for {raw_name}")
                log_message(full_log[:1500])
                continue

            if source not in best_by_source or affinity < best_by_source[source]:
                best_by_source[source] = affinity

        for idx in range(len(smiles_list)):
            source = str(idx)
            if source in best_by_source:
                predictions[idx] = float(best_by_source[source])

    return predictions


# =========================================================
# MAIN
# =========================================================
def main():
    raw = sys.stdin.read().strip()

    if not raw:
        print(json.dumps({"version": 1, "payload": {"predictions": []}}))
        return

    try:
        data = json.loads(raw)
        smiles_list = data["payload"]["smiles"]
    except Exception as e:
        log_message(f"JSON INPUT ERROR: {repr(e)}")
        print(json.dumps({"version": 1, "payload": {"predictions": []}}))
        return

    try:
        predictions = score_smiles_batch(smiles_list)
    except Exception as e:
        log_message(f"FATAL SCORING ERROR: {repr(e)}")
        predictions = [FAILED_PREDICTION] * len(smiles_list)

    out = {
        "version": 1,
        "payload": {
            "predictions": predictions
        }
    }
    print(json.dumps(out))


if __name__ == "__main__":
    main()