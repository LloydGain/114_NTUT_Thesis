from asyncio import timeout
import optuna
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from optuna.samplers import TPESampler
from setup import *
from main import main

def objective_extract(trial, args):
    """
    Notes:
        Objective function for GA store extraction hyperparameter tuning with Optuna.

    Args:
        trial (optuna.trial.Trial): Optuna trial for suggesting hyperparameters.
        args (tuple): Additional arguments for `main()` (e.g., file_date, seed).

    Returns:
        float: Cost to be minimized.
    """
    h_params = {
        "population_size": trial.suggest_int("extraction_pop", 30, 200, step=10),
        "cross_rate": trial.suggest_float("extraction_cross", 0.5, 1.0, step=0.05),
        "mutation_rate": trial.suggest_float("extraction_mutate", 0.001, 0.05, step=0.001),
    }
    return main(args.file_date, args.seed, hyper_params=h_params)


def objective_allocate(trial, args):
    """
    Notes:
        Objective function for GA store allocation hyperparameter tuning with Optuna.

    Args:
        trial (optuna.trial.Trial): Optuna trial for suggesting hyperparameters.
        args (tuple): Additional arguments for `main()` (e.g., file_date, seed).

    Returns:
        float: Cost to be minimized.
    """
    h_params = {
        "population_size": trial.suggest_int("allocation_pop", 30, 200, step=10),
        "cross_rate": trial.suggest_float("allocation_cross", 0.5, 1.0, step=0.05),
        "mutation_rate": trial.suggest_float("allocation_mutate", 0.001, 0.05, step=0.001),
    }
    return main(args.file_date, args.seed, hyper_params=h_params)


def objective_aco(trial, args):
    """
    Notes:
        Objective function for ACO support line planning hyperparameter tuning with Optuna.

    Args:
        trial (optuna.trial.Trial): Optuna trial for suggesting hyperparameters.
        args (tuple): Additional arguments for `main()` (e.g., file_date, seed).

    Returns:
        float: Cost to be minimized.
    """
    h_params = {
        "alpha": trial.suggest_float("aco_alpha", 0.5, 3.0, step=0.5),
        "beta": trial.suggest_float("aco_beta", 1.0, 10.0, step=1),
        "rho": trial.suggest_float("aco_rho", 0.1, 0.7, step=0.05),
    }
    return main(args.file_date, args.seed, hyper_params=h_params)


def run_bayesian_optimization(objective_fn, args, study_name, n_trials=50, direction="minimize", storage=None):
    """
    Notes:
        Run a Bayesian optimization using Optuna.
    """
    sampler = TPESampler()
    study = optuna.create_study(
        study_name=study_name,
        sampler=sampler,
        direction=direction,
        storage=storage,
        load_if_exists=True
    )
    study.optimize(lambda trial: objective_fn(trial, args), n_trials=n_trials)
    return study


def run_full_tuning(args):
    """
    View dashboard:
        optuna-dashboard sqlite:///path
    """
    root_dir = Path(__file__).resolve().parent.parent
    db_time = datetime.today().strftime("%Y-%m-%d_%H-%M-%S")
    # db_time = "2026-03-18_16-26-43"

    output_dir = root_dir / "output" / "optuna"
    output_dir.mkdir(parents=True, exist_ok=True)

    DB = f"sqlite:///{output_dir / f'optuna_{db_time}.db'}"
    print("==== Bayesian Optimization: store extraction GA ====")
    run_bayesian_optimization(objective_extract, args, "extract_ga", n_trials=50, storage=DB)

    # print("==== Bayesian Optimization: store allocation GA ====")
    # run_bayesian_optimization(objective_allocate, args, "allocate_ga", n_trials=50, storage=DB)

    # print("==== Bayesian Optimization: support line ACO ====")
    # run_bayesian_optimization(objective_aco, args, "support_line_aco", n_trials=50, storage=DB)

    print("==== All Tuning Complete! ====")


if __name__ == "__main__":
    args = SimpleNamespace(
        file_date = "20221223",
        seed = 0
    )
    run_full_tuning(args)