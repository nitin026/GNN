"""Fixed whole-case splits (no leakage). Used by every training / evaluation script.

Test cases are never used for training, model selection or threshold tuning.
amphan_replay is kept out of training: it is the bridge to the real Amphan evaluation.
"""
TRAIN = ["cyc_01", "cyc_02", "heat_01", "heat_02", "cold_01", "cold_02"]
VAL = ["cyc_03", "heat_03", "cold_03"]
TEST = ["cyc_04", "heat_04", "cold_04", "amphan_replay"]
ALL = TRAIN + VAL + TEST


def split_of(case_id):
    for name, ids in (("train", TRAIN), ("val", VAL), ("test", TEST)):
        if case_id in ids:
            return name
    return "extra_train" if case_id.startswith("gen_") else None
