# data.py
import numpy as np
import pandas as pd
import random
from collections import defaultdict

EVENT_TYPES = ["DIAG_diabetes", "DIAG_obesity", "DIAG_hf",
               "MED_metformin", "MED_statins", "LAB_hba1c", "LAB_glucose", "VITAL_bp", "VITAL_bmi"]

def gen_patient_sequence(max_visits=8, seed=None):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    visits = []
    t = 0.0
    for i in range(random.randint(3, max_visits)):
        # random delta-days between visits (0.5 - 60 days)
        dt = float(np.abs(np.random.exponential(scale=10.0))) + 0.5
        t += dt
        # choose 1-4 events per visit
        k = random.randint(1, 4)
        events = list(np.random.choice(EVENT_TYPES, size=k))
        # numeric measurements (example)
        vitals = {
            "glucose": float(90 + np.random.randn()*15 + (i*0.5)),
            "bmi": float(25 + np.random.randn()*2 + (i*0.02)),
            "bp": float(120 + np.random.randn()*8)
        }
        meds_adherence = float(max(0.0, min(1.0, 1 - np.random.beta(2, 20))))  # fraction adhered
        visits.append({
            "time": t,
            "events": events,
            "vitals": vitals,
            "meds_adherence": meds_adherence
        })
    return visits

def build_dataset(num_patients=500, max_visits=10, seed=42):
    patients = []
    for pid in range(num_patients):
        seq = gen_patient_sequence(max_visits=max_visits, seed=seed+pid)
        # make a simple label: deterioration within 90 days if last glucose spike + missed meds
        last_glucose = seq[-1]["vitals"]["glucose"]
        avg_missed = 1 - np.mean([v["meds_adherence"] for v in seq])
        deterioration_prob = (max(0, (last_glucose - 120)/80) * 0.6) + (avg_missed * 0.4)
        label = 1 if (deterioration_prob + np.random.rand()*0.2) > 0.4 else 0
        patients.append({"patient_id": f"P{pid+1}", "visits": seq, "label": label})
    return patients

# helpers to flatten into model inputs
def prepare_sequences_for_model(patients, vocab=None, max_seq_len=120, pad_token="<PAD>"):
    # Create vocabulary from event tokens if not provided
    if vocab is None:
        vocab = set()
        for p in patients:
            for v in p["visits"]:
                for e in v["events"]:
                    vocab.add(e)
        vocab = {tok: idx for idx, tok in enumerate(sorted(vocab))}
    sequences = []
    labels = []
    times = []
    extra_feats = []  # e.g., aggregated vitals per visit
    pids = []
    for p in patients:
        events_seq = []
        time_seq = []
        feats_seq = []
        for v in p["visits"]:
            # aggregate event tokens per visit as space-separated "words" for FastText
            events_seq.append(" ".join(v["events"]))
            time_seq.append(v["time"])
            feats_seq.append([v["vitals"]["glucose"], v["vitals"]["bmi"], v["vitals"]["bp"], v["meds_adherence"]])
        sequences.append(events_seq)
        times.append(time_seq)
        extra_feats.append(feats_seq)
        labels.append(p["label"])
        pids.append(p["patient_id"])
    return sequences, times, extra_feats, labels, pids, vocab
