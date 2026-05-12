import json
import os
import sys
from typing import Any, Dict, List, Tuple

try:
    import numpy as np
except Exception:
    np = None


def describe_value(v: Any) -> Tuple[str, int, str]:
    tname = type(v).__name__
    length = -1
    shape = ""
    if isinstance(v, (list, tuple)):
        length = len(v)
        if np is not None:
            try:
                arr = np.array(v)
                shape = str(arr.shape)
            except Exception:
                shape = "unavailable"
        else:
            shape = "numpy-not-available"
    elif isinstance(v, dict):
        length = len(v)
        shape = "dict"
    else:
        length = -1
        shape = "scalar"
    return tname, length, shape


def is_numeric_list(lst) -> bool:
    if not isinstance(lst, (list, tuple)):
        return False
    if len(lst) == 0:
        return False
    if np is not None:
        try:
            arr = np.array(lst)
            return arr.dtype.kind in ("f", "i") and arr.ndim >= 1
        except Exception:
            return False
    else:
        # Fallback: check element types
        return all(isinstance(x, (int, float)) for x in lst)


def recursive_find_numeric(obj: Any, path: str = "") -> List[Tuple[str, int, str]]:
    found = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            p = f"{path}/{k}" if path else k
            found += recursive_find_numeric(v, p)
    elif isinstance(obj, list):
        # If list itself is numeric
        if is_numeric_list(obj):
            shape = ""
            if np is not None:
                try:
                    shape = str(np.array(obj).shape)
                except Exception:
                    shape = "unavailable"
            found.append((path or "<root_list>", len(obj), shape))
        else:
            # traverse elements (e.g., list of dicts)
            for idx, item in enumerate(obj[:5]):
                p = f"{path}[{idx}]" if path else f"[{idx}]"
                found += recursive_find_numeric(item, p)
    return found


def inspect_file(path: str) -> Dict[str, Dict]:
    if not os.path.exists(path):
        print(f"Missing file: {path}")
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    top = {}
    print(f"Loaded: {path}")
    print("Top-level keys:")
    for k in sorted(data.keys()):
        v = data[k]
        tname, length, shape = describe_value(v)
        if length >= 0:
            print(f" - {k}: type={tname}, len={length}, shape={shape}")
        else:
            print(f" - {k}: type={tname}, {shape}")
        top[k] = {"type": tname, "len": length, "shape": shape}

    # find numeric arrays nested anywhere
    numeric_found = recursive_find_numeric(data)
    if numeric_found:
        print("\nNumeric arrays/lists found (path, len, shape):")
        for p, l, s in numeric_found:
            print(f" - {p}: len={l}, shape={s}")
    else:
        print("\nNo numeric lists/arrays found in nested structure (limited scan).")

    # heuristic candidate keys
    candidates = []
    keywords = ["score", "scores", "anomaly", "distance", "distances", "val_", "test_", "label"]
    for k, v in data.items():
        kl = k.lower()
        if any(kw in kl for kw in keywords):
            flag = ""
            if is_numeric_list(v):
                flag = " (numeric list)"
            candidates.append((k, type(v).__name__, flag))

    if candidates:
        print("\nHeuristic candidate keys (likely score/label arrays):")
        for k, t, flag in candidates:
            print(f" - {k}: {t}{flag}")
    else:
        print("\nNo obvious candidate keys found by heuristic.")

    return {"top": top, "numeric_found": numeric_found, "candidates": candidates, "data": data}


def save_fold4_structure(out_path: str, info: Dict[str, Dict]):
    lines = []
    top = info.get("top", {})
    lines.append("fold_4 structure:\n")
    for k, meta in sorted(top.items()):
        lines.append(f"{k}: type={meta['type']}, len={meta['len']}, shape={meta['shape']}\n")

    if info.get("numeric_found"):
        lines.append("\nNumeric entries found (path, len, shape):\n")
        for p, l, s in info["numeric_found"]:
            lines.append(f"{p}: len={l}, shape={s}\n")

    with open(out_path, "w", encoding="utf-8") as f:
        f.writelines(lines)

    print(f"Saved fold_4 structure to: {out_path}")


def main():
    base = os.path.join(os.getcwd(), "experiments_multimodal_kaggle", "multimodal_hyperbolic_prototype_v1", "hyperbolic")
    fold4 = os.path.join(base, "fold_4", "fold_results.json")

    print("\n== Inspecting fold_4/fold_results.json ==\n")
    info = inspect_file(fold4)
    out_file = os.path.join(os.getcwd(), "fold4_structure.txt")
    save_fold4_structure(out_file, info)

    print("\n== Inspecting all folds (0..4) top-level keys ==\n")
    for i in range(5):
        p = os.path.join(base, f"fold_{i}", "fold_results.json")
        print(f"\n-- Fold {i} --")
        if not os.path.exists(p):
            print(f" Missing: {p}")
            continue
        try:
            with open(p, "r", encoding="utf-8") as f:
                d = json.load(f)
            print(f" Top-level keys: {sorted(list(d.keys()))}")
        except Exception as e:
            print(f" Error loading fold_{i}: {e}")


if __name__ == "__main__":
    main()
