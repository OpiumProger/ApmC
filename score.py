from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")

import sys
import json
import math
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from rdkit import Chem, DataStructs
from rdkit.Chem import (
    Descriptors,
    rdMolDescriptors,
    Crippen,
    GraphDescriptors,
    MolSurf,
    Fragments,
    QED,
    rdFingerprintGenerator,
)
from rdkit.Chem import EState
from rdkit.Chem.EState import EState_VSA

try:
    from rdkit.Chem.SpacialScore import SPS
except ImportError:
    SPS = lambda mol: 0.0


# =========================================================
# CONFIG
# =========================================================
BUNDLE_PATH = Path("C:/Users/norep/Desktop/hakaton/catboost_ensemble_bundle.pkl")

INVALID_PRED_MEAN = 0.0
INVALID_PRED_STD = 999.0
INVALID_SIMILARITY = 0.0
INVALID_CONFIDENCE = 0.0
INVALID_FINAL_SCORE = 0.0


# =========================================================
# JSON IO
# =========================================================
def read_payload():
    raw = sys.stdin.read()
    if raw is None:
        return []

    raw = raw.strip()
    if not raw:
        return []

    try:
        data = json.loads(raw)

        if isinstance(data, dict):
            if "smiles" in data and isinstance(data["smiles"], list):
                return data["smiles"]

            if "payload" in data and isinstance(data["payload"], dict):
                payload = data["payload"]
                if "smiles" in payload and isinstance(payload["smiles"], list):
                    return payload["smiles"]

        if isinstance(data, list):
            return [str(x) for x in data]

    except json.JSONDecodeError:
        pass

    lines = [line.strip() for line in raw.splitlines() if line.strip()]
    if lines and lines[0].upper() == "SMILES":
        lines = lines[1:]

    return lines


def write_payload(predictions):
    out = {
        "version": 1,
        "payload": {
            "predictions": predictions
        }
    }
    sys.stdout.write(json.dumps(out))
    sys.stdout.flush()


# =========================================================
# RDKit HELPERS
# =========================================================
def build_descriptor_registry():
    registry = {name: func for name, func in Descriptors.descList}

    registry.update({
        "qed": QED.qed,
        "QED": QED.qed,
        "SPS": SPS,

        "MolWt": Descriptors.MolWt,
        "MolLogP": Crippen.MolLogP,
        "TPSA": rdMolDescriptors.CalcTPSA,

        "FractionCSP3": rdMolDescriptors.CalcFractionCSP3,
        "NHOHCount": rdMolDescriptors.CalcNumLipinskiHBD,
        "NumHAcceptors": rdMolDescriptors.CalcNumLipinskiHBA,
        "RingCount": rdMolDescriptors.CalcNumRings,

        "NumAliphaticCarbocycles": rdMolDescriptors.CalcNumAliphaticCarbocycles,
        "NumAliphaticHeterocycles": rdMolDescriptors.CalcNumAliphaticHeterocycles,
        "NumAliphaticRings": rdMolDescriptors.CalcNumAliphaticRings,
        "NumAromaticCarbocycles": rdMolDescriptors.CalcNumAromaticCarbocycles,
        "NumAromaticHeterocycles": rdMolDescriptors.CalcNumAromaticHeterocycles,
        "NumAromaticRings": rdMolDescriptors.CalcNumAromaticRings,
        "NumHeterocycles": rdMolDescriptors.CalcNumHeterocycles,
        "NumSaturatedCarbocycles": rdMolDescriptors.CalcNumSaturatedCarbocycles,
        "NumSaturatedHeterocycles": rdMolDescriptors.CalcNumSaturatedHeterocycles,
        "NumSaturatedRings": rdMolDescriptors.CalcNumSaturatedRings,

        "NumAtomStereoCenters": rdMolDescriptors.CalcNumAtomStereoCenters,
        "NumUnspecifiedAtomStereoCenters": rdMolDescriptors.CalcNumUnspecifiedAtomStereoCenters,
        "NumAmideBonds": rdMolDescriptors.CalcNumAmideBonds,
        "NumBridgeheadAtoms": rdMolDescriptors.CalcNumBridgeheadAtoms,
        "NumSpiroAtoms": rdMolDescriptors.CalcNumSpiroAtoms,

        "BalabanJ": GraphDescriptors.BalabanJ,
        "BertzCT": GraphDescriptors.BertzCT,
        "HallKierAlpha": GraphDescriptors.HallKierAlpha,
        "AvgIpc": lambda mol: GraphDescriptors.Ipc(mol, avg=True),

        "MaxAbsEStateIndex": EState.MaxAbsEStateIndex,
        "MinAbsEStateIndex": EState.MinAbsEStateIndex,
        "MaxEStateIndex": EState.MaxEStateIndex,
        "MinEStateIndex": EState.MinEStateIndex,

        "FpDensityMorgan1": Descriptors.FpDensityMorgan1,
    })

    for module in [MolSurf, EState_VSA, Fragments]:
        for name in dir(module):
            obj = getattr(module, name)
            if callable(obj):
                registry.setdefault(name, obj)

    return registry


DESCRIPTOR_REGISTRY = build_descriptor_registry()


def get_descriptor_function(name):
    if name in DESCRIPTOR_REGISTRY:
        return DESCRIPTOR_REGISTRY[name]
    if hasattr(Descriptors, name):
        return getattr(Descriptors, name)
    return None


def get_morgan_generator(radius, n_bits):
    return rdFingerprintGenerator.GetMorganGenerator(
        radius=radius,
        fpSize=n_bits
    )


def featurize_smiles_to_row(smiles, descriptor_columns, generator, fp_nbits, fp_prefix="fp_"):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, None, None

    row = {}

    for col in descriptor_columns:
        func = get_descriptor_function(col)
        if func is None:
            raise KeyError(
                f"Descriptor '{col}' not found in RDKit registry. "
                f"Training/inference feature mismatch."
            )
        try:
            val = func(mol)
            if val is None or not np.isfinite(val):
                val = 0.0
        except Exception:
            val = 0.0
        row[col] = float(val)

    fp = generator.GetFingerprint(mol)
    arr = np.zeros((fp_nbits,), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(fp, arr)

    for i in range(fp_nbits):
        row[f"{fp_prefix}{i}"] = int(arr[i])

    return row, fp, mol


def max_tanimoto_similarity(query_bv, train_bitvects):
    if query_bv is None:
        return 0.0

    sims = []
    for bv in train_bitvects:
        if bv is None:
            continue
        sims.append(DataStructs.TanimotoSimilarity(query_bv, bv))

    return float(max(sims)) if sims else 0.0


# =========================================================
# SCORING LOGIC
# =========================================================
def sigmoid_transform(x, low=5.5, high=8.5, k=1.0):
    mid = 0.5 * (low + high)
    scale = max((high - low) / 2.0, 1e-8)
    z = (x - mid) / (scale / max(k, 1e-8))
    z = max(min(z, 60.0), -60.0)
    return float(1.0 / (1.0 + math.exp(-z)))


def compute_uncertainty_score(pred_std):
    score = 1.0 / (1.0 + 1.5 * pred_std)
    if not np.isfinite(score):
        return 0.0
    return float(min(1.0, max(0.0, score)))


def compute_novelty_score(max_similarity):
    sim = float(max_similarity)

    if sim >= 0.85:
        return 0.10
    elif sim >= 0.70:
        t = (0.85 - sim) / (0.85 - 0.70)
        return float(0.10 + t * (0.60 - 0.10))
    elif sim >= 0.35:
        return 1.00
    elif sim >= 0.20:
        t = (sim - 0.20) / (0.35 - 0.20)
        return float(0.70 + t * (1.00 - 0.70))
    else:
        return 0.30


def compute_confidence(pred_std, max_similarity):
    uncertainty_score = compute_uncertainty_score(pred_std)
    novelty_score = compute_novelty_score(max_similarity)
    conf = uncertainty_score * novelty_score
    if not np.isfinite(conf):
        return 0.0
    return float(min(1.0, max(0.0, conf)))


def compute_qed_score(qed_value):
    if not np.isfinite(qed_value):
        return 0.2

    if qed_value >= 0.8:
        return 1.0
    elif qed_value >= 0.5:
        return 0.7 + (qed_value - 0.5) / 0.3 * 0.3
    elif qed_value >= 0.3:
        return 0.4 + (qed_value - 0.3) / 0.2 * 0.3
    else:
        return 0.2


def compute_logp_score(logp):
    if not np.isfinite(logp):
        return 0.2

    if 1.0 <= logp <= 3.5:
        return 1.0
    elif 0.0 <= logp < 1.0:
        return 0.6 + 0.4 * (logp / 1.0)
    elif 3.5 < logp <= 5.0:
        return 1.0 - 0.6 * (logp - 3.5) / 1.5
    else:
        return 0.2


def compute_final_score(pred_mean, pred_std, max_similarity, mol):
    activity_score = sigmoid_transform(pred_mean, low=5.5, high=8.5, k=1.0)
    uncertainty_score = compute_uncertainty_score(pred_std)
    novelty_score = compute_novelty_score(max_similarity)

    try:
        qed_value = QED.qed(mol)
    except Exception:
        qed_value = 0.0

    try:
        logp_value = Crippen.MolLogP(mol)
    except Exception:
        logp_value = np.nan

    qed_score = compute_qed_score(qed_value)
    logp_score = compute_logp_score(logp_value)

    base_score = activity_score * uncertainty_score * novelty_score
    score = base_score * (0.7 + 0.3 * qed_score) * (0.7 + 0.3 * logp_score)

    if not np.isfinite(score):
        return 0.0

    return float(min(1.0, max(0.0, score)))


# =========================================================
# MAIN PREDICTION
# =========================================================
def predict_with_uncertainty(smiles_list, bundle):
    descriptor_columns = bundle["descriptor_columns"]
    fp_radius = bundle["fp_radius"]
    fp_nbits = bundle["fp_nbits"]
    train_bitvects = bundle["train_bitvects"]
    ensemble_models = bundle["ensemble_models"]
    fp_prefix = bundle.get("fp_prefix", "fp_")

    generator = get_morgan_generator(fp_radius, fp_nbits)

    rows = []
    valid_indices = []
    query_bitvects = {}
    mol_map = {}

    for i, smi in enumerate(smiles_list):
        row, bv, mol = featurize_smiles_to_row(
            smi,
            descriptor_columns=descriptor_columns,
            generator=generator,
            fp_nbits=fp_nbits,
            fp_prefix=fp_prefix
        )
        if row is not None:
            rows.append(row)
            valid_indices.append(i)
            query_bitvects[i] = bv
            mol_map[i] = mol

    pred_map = {}

    if rows:
        Xq = pd.DataFrame(rows)

        all_preds = []
        for model in ensemble_models:
            preds = model.predict(Xq)
            all_preds.append(np.asarray(preds, dtype=float))

        all_preds = np.vstack(all_preds)
        pred_mean = all_preds.mean(axis=0)
        pred_std = all_preds.std(axis=0)

        for idx, mean_val, std_val in zip(valid_indices, pred_mean, pred_std):
            sim = max_tanimoto_similarity(query_bitvects[idx], train_bitvects)

            uncertainty_score = compute_uncertainty_score(std_val)
            novelty_score = compute_novelty_score(sim)
            confidence = compute_confidence(std_val, sim)
            final_score = compute_final_score(mean_val, std_val, sim, mol_map[idx])

            pred_map[idx] = {
                "pred_mean": float(mean_val),
                "pred_std": float(std_val),
                "max_similarity": float(sim),
                "novelty": float(1.0 - sim),
                "uncertainty_score": float(uncertainty_score),
                "novelty_score": float(novelty_score),
                "confidence": float(confidence),
                "final_score": float(final_score),
                "is_valid": True
            }

    results = []
    for i, smi in enumerate(smiles_list):
        if i in pred_map:
            item = {"smiles": smi}
            item.update(pred_map[i])
        else:
            item = {
                "smiles": smi,
                "pred_mean": INVALID_PRED_MEAN,
                "pred_std": INVALID_PRED_STD,
                "max_similarity": INVALID_SIMILARITY,
                "novelty": 1.0,
                "confidence": INVALID_CONFIDENCE,
                "final_score": INVALID_FINAL_SCORE,
                "is_valid": False
            }
        results.append(item)

    return results


# =========================================================
# MAIN
# =========================================================
def main():
    with open(BUNDLE_PATH, "rb") as f:
        bundle = pickle.load(f)

    smiles_list = read_payload()

    if not smiles_list:
        write_payload([])
        return

    results = predict_with_uncertainty(smiles_list, bundle)
    predictions = [float(x["final_score"]) for x in results]
    write_payload(predictions)


if __name__ == "__main__":
    main()