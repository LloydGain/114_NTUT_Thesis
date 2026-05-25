import optuna
import argparse
import pandas as pd
from pathlib import Path
from setup import *
from main import main


LEVEL = {
    "ex_pop":   [5, 10, 15, 20],
    "ex_cx":    [0.6, 0.7, 0.8, 0.9],
    "ex_mut":   [0.01, 0.03, 0.05, 0.1],
    "al_ants":  [5, 10, 25, 50],
    "al_beta":  [1, 3, 5, 7],
    "al_rho":   [0.1, 0.3, 0.5, 0.7],
    "ants":     [5, 10, 25, 50],
    "beta":     [1, 3, 5, 7],
    "rho":      [0.1, 0.3, 0.5, 0.7],
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
        "ex_pop": LEVEL["ex_pop"][row[0]-1],
        "ex_cx":  LEVEL["ex_cx"][row[1]-1],
        "ex_mut": LEVEL["ex_mut"][row[2]-1],
        "al_ants": LEVEL["al_ants"][row[3]-1],
        "al_beta": LEVEL["al_beta"][row[4]-1],
        "al_rho":  LEVEL["al_rho"][row[5]-1],
        "ants":    LEVEL["ants"][row[6]-1],
        "beta":    LEVEL["beta"][row[7]-1],
        "rho":     LEVEL["rho"][row[8]-1],
    }


def objective(trial, data_name, seed):
    ex_pop   = trial.suggest_categorical("ex_pop",  LEVEL["ex_pop"])
    ex_cx    = trial.suggest_categorical("ex_cx",   LEVEL["ex_cx"])
    ex_mut   = trial.suggest_categorical("ex_mut",  LEVEL["ex_mut"])
    al_ants  = trial.suggest_categorical("al_ants",  LEVEL["al_ants"])
    al_beta  = trial.suggest_categorical("al_beta",  LEVEL["al_beta"])
    al_rho   = trial.suggest_categorical("al_rho",   LEVEL["al_rho"])
    ants     = trial.suggest_categorical("ants",     LEVEL["ants"])
    beta     = trial.suggest_categorical("beta",     LEVEL["beta"])
    rho      = trial.suggest_categorical("rho",      LEVEL["rho"])

    params = {
        "ex_pop": ex_pop, "ex_cx": ex_cx, "ex_mut": ex_mut,
        "al_ants": al_ants, "al_beta": al_beta, "al_rho": al_rho,
        "ants": ants, "beta": beta, "rho": rho,
    }

    result = main(
        file_date=data_name,
        random_seed=seed,
        hyper_params=params
    )
    return result["cost"]


def save_results(study, output_dir, data_name, seed):
    records = []

    for t in study.trials:
        if t.value is None:
            continue

        row = {"trial": t.number}
        row.update(t.params)
        row["value"] = t.value
        records.append(row)

    if not records:
        return

    df = pd.DataFrame(records)

    min_val = df["value"].min()
    max_val = df["value"].max()

    if max_val == min_val:
        df["norm_value"] = 1.0
    else:
        df["norm_value"] = (max_val - df["value"]) / (max_val - min_val)

    excel_path = output_dir / f"optuna_{data_name}_seed{seed}.xlsx"
    df.to_excel(excel_path, index=False)

    print(f"Saved Excel to: {excel_path}")


def run(data_name, seed=0):
    output_dir = ROOT / "output" / "optuna"
    db_dir = ROOT / "output" / "optuna" / "db"
    trial_dir = ROOT / "output" / "optuna" / "trial"
    output_dir.mkdir(parents=True, exist_ok=True)
    db_dir.mkdir(parents=True, exist_ok=True)
    trial_dir.mkdir(parents=True, exist_ok=True)

    db_path = db_dir / f"optuna_{data_name}.db"

    study = optuna.create_study(
        direction="minimize",
        study_name=f"taguchi_l32_{data_name}_{seed}",
        storage=f"sqlite:///{db_path}",
        load_if_exists=True
    )

    existing_trials = [
        t.params for t in study.trials
        if t.state == optuna.trial.TrialState.COMPLETE
    ]

    for row in L32:
        params = decode(row)

        if params not in existing_trials:
            study.enqueue_trial(params)

    remaining = 32 - len(existing_trials)

    study.optimize(
        lambda t: objective(t, data_name, seed),
        n_trials=remaining,
        gc_after_trial=True,
        callbacks=[lambda study, trial: save_results(study, trial_dir, data_name, seed)]
    )


    print("Best value:", study.best_value)
    print("Best params:", study.best_params)

    save_results(study, trial_dir, data_name, seed)

    return study


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_date", type=str, required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    run(args.file_date, args.seed)