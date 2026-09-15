"""Standing parameters from assignment 1, Section 0.1. SID4 = last four of SJSU ID."""

SID4 = 1605
SEED = SID4
SLICE = SID4 % 1000
HP_ID = SID4 % 6
CLS_A = SID4 % 10
CLS_B = (CLS_A + 1 + ((SID4 // 10) % 9)) % 10

STUDENT = "Rajesh Paruchuri"
COURSE = "DATA 266"
ASSIGNMENT = "HW2.5"

# HW2.5 does not define an HP_ID mapping. The assignment explicitly allows
# skipping the second model with a different hyperparameter configuration.
HP_NOTE = (
    "HP_ID is reported only, from Section 0.1. HW2.5 has no HP_ID mapping and "
    "no second-model requirement, so it is not trained here."
)


def seed_everything(seed: int = SEED) -> None:
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def banner() -> str:
    return (
        f"{COURSE} {ASSIGNMENT} — {STUDENT}\n"
        f"SID4={SID4:04d}  SEED={SEED}  SLICE={SLICE}  "
        f"HP_ID={HP_ID}  CLS_A={CLS_A}  CLS_B={CLS_B}\n"
        f"{HP_NOTE}"
    )
