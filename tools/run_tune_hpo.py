import optuna
from pathlib import Path
from setup import *
from main import main


LEVEL = {
    "pop":   [50, 100, 150, 200],
    "cx":    [0.6, 0.7, 0.8, 0.9],
    "mut":   [0.01, 0.03, 0.05, 0.1],
    "rho":   [0.1, 0.3, 0.5, 0.7],
    "alpha": [0.5, 1, 2, 3],
    "beta":  [1, 4, 7, 10],
}

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


def objective(trial, data_name, seed):
    ex_pop   = trial.suggest_categorical("ex_pop",  LEVEL["pop"])
    ex_cx    = trial.suggest_categorical("ex_cx",   LEVEL["cx"])
    ex_mut   = trial.suggest_categorical("ex_mut",  LEVEL["mut"])
    al_pop   = trial.suggest_categorical("al_pop",  LEVEL["pop"])
    al_cx    = trial.suggest_categorical("al_cx",   LEVEL["cx"])
    al_mut   = trial.suggest_categorical("al_mut",  LEVEL["mut"])
    rho      = trial.suggest_categorical("rho",     LEVEL["rho"])
    alpha    = trial.suggest_categorical("alpha",   LEVEL["alpha"])
    beta     = trial.suggest_categorical("beta",    LEVEL["beta"])

    params = {
        "ex_pop": ex_pop, "ex_cx": ex_cx, "ex_mut": ex_mut,
        "al_pop": al_pop, "al_cx": al_cx, "al_mut": al_mut,
        "rho": rho, "alpha": alpha, "beta": beta,
    }

    return main(
        file_date=data_name,
        random_seed=seed,
        hyper_params=params
    )


def run(data_name, seed=0):
    output_dir = Path(r"C:\Users\User\Documents\ntut\115_NTUT_Thesis\output\optuna")
    output_dir.mkdir(parents=True, exist_ok=True)

    db_path = output_dir / f"optuna_{data_name}.db"

    study = optuna.create_study(
        direction="minimize",
        study_name=f"taguchi_l32_{data_name}_{seed}",
        storage=f"sqlite:///{db_path}",
        load_if_exists=False
    )

    for i in range(len(L32)):
        study.enqueue_trial(decode(L32[i]))

    study.optimize(lambda t: objective(t, data_name, seed), n_trials=32)

    print("Best value:", study.best_value)
    print("Best params:", study.best_params)

    return study


if __name__ == "__main__":
    run("20221203", seed=0)