import os
import pickle
import numpy as np
import pandas as pd

# Numărul de jucători sintetici per poziție
N_SYNTHETIC_PER_POSITION = 75

# ID-uri sintetice — negative ca să nu se confunde cu ID-uri reale Wyscout
SYNTHETIC_ID_START = -1


def generate_synthetic_players(
    df_real: pd.DataFrame,
    n: int,
    position: str,
    id_start: int,
) -> pd.DataFrame:
    """
    Generează n jucători sintetici pentru o poziție.

    Atributele sunt uniform distribuite între
    min și max din datele reale pentru fiecare feature.

    Parameters:
        df_real   — DataFrame normalizat real pentru poziție
        n         — numărul de jucători de generat
        position  — numele poziției (pentru name)
        id_start  — ID-ul de start pentru jucătorii sintetici

    Returns:
        DataFrame cu n rânduri sintetice
    """
    features = [c for c in df_real.columns if c not in ["playerId", "name", "is_synthetic"]]

    synthetic_rows = []
    for i in range(n):
        row = {
            "playerId":     id_start - i,
            "name":         f"Synthetic_{position}_{i+1}",
            "is_synthetic": True,
        }

        for feat in features:
            col_min = float(df_real[feat].min())
            col_max = float(df_real[feat].max())

            # Uniform între min și max din datele reale
            if col_max > col_min:
                row[feat] = round(np.random.uniform(col_min, col_max), 4)
            else:
                row[feat] = col_min

        synthetic_rows.append(row)

    return pd.DataFrame(synthetic_rows)


def augment_all_positions(
    saved_data_path: str = "saved_data",
    n_per_position: int = N_SYNTHETIC_PER_POSITION,
    seed: int = 42,
) -> dict:
    """
    Generează și salvează date sintetice pentru toate pozițiile.

    Returns dict cu statistici: {pozitie: n_generat}
    """
    np.random.seed(seed)

    stats = {}
    id_counter = SYNTHETIC_ID_START

    for filename in sorted(os.listdir(saved_data_path)):
        if not filename.startswith("normalized_") or not filename.endswith(".pkl"):
            continue

        pozitie = filename.replace("normalized_", "").replace(".pkl", "")
        real_path = os.path.join(saved_data_path, filename)
        aug_path  = os.path.join(saved_data_path, f"augmented_{pozitie}.pkl")

        # Încarcă datele reale
        df_real = pd.read_pickle(real_path)

        # Adaugă flag is_synthetic=False la datele reale dacă nu există
        if "is_synthetic" not in df_real.columns:
            df_real["is_synthetic"] = False

        # Generează sintetice
        df_synthetic = generate_synthetic_players(
            df_real=df_real,
            n=n_per_position,
            position=pozitie,
            id_start=id_counter,
        )

        # Salvează augmented pkl (real + sintetic combinat)
        df_augmented = pd.concat([df_real, df_synthetic], ignore_index=True)
        df_augmented.to_pickle(aug_path)

        id_counter -= n_per_position
        stats[pozitie] = {
            "real":      len(df_real),
            "synthetic": n_per_position,
            "total":     len(df_augmented),
        }

        print(f"[OK] {pozitie}: {len(df_real)} reali + {n_per_position} sintetici = {len(df_augmented)} total")

    return stats


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    saved_data_path = os.path.join(base_dir, "saved_data")

    print("Generând date sintetice...\n")
    stats = augment_all_positions(saved_data_path=saved_data_path)

    total_synthetic = sum(s["synthetic"] for s in stats.values())
    total_all       = sum(s["total"] for s in stats.values())

    print(f"\n{'='*50}")
    print(f"Total jucători sintetici generați: {total_synthetic}")
    print(f"Total jucători (real + sintetic):  {total_all}")
    print(f"{'='*50}")
    print("\nFișiere salvate în saved_data/augmented_*.pkl")
    print("Pentru a reveni la date reale: șterge fișierele augmented_*.pkl")