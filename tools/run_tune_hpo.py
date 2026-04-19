import optuna
from pathlib import Path
from setup import *
from main import main


# =========================
# 1. Level mapping
# =========================

LEVEL = {
    "pop":   [50, 100, 150, 200],
    "cx":    [0.6, 0.7, 0.8, 0.9],
    "mut":   [0.01, 0.03, 0.05, 0.1],
    "rho":   [0.1, 0.3, 0.5, 0.7],
    "alpha": [0.5, 1, 2, 3],
    "beta":  [1, 4, 7, 10],
}


# =========================
# 2. L32 matrix
# =========================

L32 = [
    [1,1,1,1,1,1,1,1,1],
    [1,2,2,2,2,2,2,2,2],
    [1,3,3,3,3,3,3,3,3],
    [1,4,4,4,4,4,4,4,4],

    [2,1,2,3,4,3,4,2,3],
    [2,2,3,4,1,4,1,3,4],
    [2,3,4,1,2,1,2,4,1],
    [2,4,1,2,3,2,3,1,2],

    [3,1,3,4,2,4,3,4,2],
    [3,2,4,1,3,1,4,1,3],
    [3,3,1,2,4,2,1,2,4],
    [3,4,2,3,1,3,2,3,1],

    [4,1,4,2,3,2,2,3,4],
    [4,2,1,3,4,3,3,4,1],
    [4,3,2,4,1,4,4,1,2],
    [4,4,3,1,2,1,1,2,3],

    [1,1,2,4,3,2,1,4,3],
    [1,2,3,1,4,3,2,1,4],
    [1,3,4,2,1,4,3,2,1],
    [1,4,1,3,2,1,4,3,2],

    [2,1,3,2,1,4,4,3,2],
    [2,2,4,3,2,1,1,4,3],
    [2,3,1,4,3,2,2,1,4],
    [2,4,2,1,4,3,3,2,1],

    [3,1,4,3,2,1,2,4,4],
    [3,2,1,4,3,2,3,1,1],
    [3,3,2,1,4,3,4,2,2],
    [3,4,3,2,1,4,1,3,3],

    [4,1,1,4,3,2,4,3,1],
    [4,2,2,1,4,3,1,4,2],
    [4,3,3,2,1,4,2,1,3],
    [4,4,4,3,2,1,3,2,4],
]


# =========================
# 3. decode
# =========================

def decode(row):
    return {
        "ex_pop": LEVEL["pop"][row[0]-1],
        "ex_cx":  LEVEL["cx"][row[1]-1],
        "ex_mut": LEVEL["mut"][row[2]-1],

        "al_pop": LEVEL["pop"][row[3]-1],
        "al_cx":  LEVEL["cx"][row[4]-1],
        "al_mut": LEVEL["mut"][row[5]-1],

        "rho":    LEVEL["rho"][row[6]-1],
        "alpha":  LEVEL["alpha"][row[7]-1],
        "beta":   LEVEL["beta"][row[8]-1],
    }


# =========================
# 4. objective
# =========================

def objective(trial, data_name, seed=0):

    run_id = trial.suggest_int("run_id", 0, 31)

    params = decode(L32[run_id])

    return main(
        file_date=data_name,
        random_seed=seed,
        hyper_params=params
    )


# =========================
# 5. run optuna
# =========================

def run(data_name, seed=0):
    # optuna-dashboard sqlite:///C:\Users\User\Documents\ntut\115_NTUT_Thesis\output\optuna\optuna_20221203.db
    output_dir = Path(r"C:\Users\User\Documents\ntut\115_NTUT_Thesis\output\optuna")
    output_dir.mkdir(parents=True, exist_ok=True)

    db_path = output_dir / f"optuna_{data_name}.db"
    storage = f"sqlite:///{db_path}"

    print(f"DB saved at: {db_path}")

    study_name_base = f"taguchi_l32_{data_name}_{seed}"
    study_name = study_name_base

    search_space = {"run_id": list(range(len(L32)))}
    study = optuna.create_study(
        direction="minimize",
        study_name=study_name,
        storage=storage,
        load_if_exists=False,
        sampler=optuna.samplers.GridSampler(search_space)
    )

    study.optimize(lambda t: objective(t, data_name, seed=seed), n_trials=32)

    print("\n===== BEST RESULT =====")
    print("Best value:", study.best_value)
    print("Best params:", study.best_params)

    return study


# =========================
# 6. main
# =========================

if __name__ == "__main__":
    run("20221203", seed=0)