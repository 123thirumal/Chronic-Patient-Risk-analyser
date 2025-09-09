# train.py
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from model import FastTextWrapper, FT1LSTM
from data import build_dataset, prepare_sequences_for_model
from tqdm import tqdm
import shap

class EHRDataset(Dataset):
    def __init__(self, sequences, times, feats, labels, fasttext, max_seq=20):
        # We'll convert visits -> vector embeddings (fasttext mean) + numeric feats
        self.X = []
        self.DT = []  # delta times per visit
        self.y = labels
        for seq_events, seq_times, seq_feats in zip(sequences, times, feats):
            # embed per visit with FastText by averaging token vectors
            visit_vecs = []
            for ev in seq_events:
                tokens = ev.split()
                # get mean vector from gensim
                vec = np.mean([fasttext.get_vector(t) for t in tokens], axis=0)
                # append numerical features
                numf = np.array(seq_feats[len(visit_vecs)], dtype=np.float32)
                visit_vecs.append(np.concatenate([vec, numf], axis=0))
            # pad/truncate
            if len(visit_vecs) > max_seq:
                visit_vecs = visit_vecs[-max_seq:]
                seq_times = seq_times[-max_seq:]
            seq_len = len(visit_vecs)
            # compute delta times (difference between visits)
            dts = [0.0] + [seq_times[i]-seq_times[i-1] for i in range(1, seq_len)]
            # stack as seq_len x dim
            self.X.append(np.stack(visit_vecs, axis=0).astype(np.float32))
            self.DT.append(np.array(dts, dtype=np.float32))
        self.max_seq=max_seq
    def __len__(self): return len(self.y)
    def __getitem__(self, idx):
        x = self.X[idx]
        dt = self.DT[idx]
        y = self.y[idx]
        seq_len = x.shape[0]
        # pad to max_seq with zeros
        pad_len = self.max_seq - seq_len
        if pad_len>0:
            x = np.vstack([np.zeros((pad_len, x.shape[1]), dtype=np.float32), x])
            dt = np.concatenate([np.zeros(pad_len, dtype=np.float32), dt])
        return x, dt, np.float32(y)

def collate(batch):
    xs, dts, ys = zip(*batch)
    xs = np.stack(xs, axis=1)  # seq, batch, dim
    dts = np.stack(dts, axis=1)  # seq, batch
    ys = np.array(ys, dtype=np.float32)
    return torch.tensor(xs), torch.tensor(dts), torch.tensor(ys)

def train_main(device="cpu"):
    # 1) build data
    patients = build_dataset(num_patients=800)
    sequences, times, feats, labels, pids, vocab = prepare_sequences_for_model(patients)
    sentences = [ " ".join(s) for s in sequences ]
    ft = FastTextWrapper(sentences, embed_dim=64, epochs=10)
    # create dataset
    dataset = EHRDataset(sequences, times, feats, labels, fasttext=ft, max_seq=20)
    # simple split
    n = len(dataset)
    idx = np.arange(n)
    np.random.shuffle(idx)
    train_idx = idx[:int(0.8*n)]
    val_idx = idx[int(0.8*n):]
    from torch.utils.data import Subset
    train_loader = DataLoader(Subset(dataset, train_idx), batch_size=32, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(Subset(dataset, val_idx), batch_size=64, shuffle=False, collate_fn=collate)

    input_dim = dataset[0][0].shape[1]  # emb_dim + numeric
    model = FT1LSTM(input_dim, hidden_size=128).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.BCELoss()

    for epoch in range(6):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        for xs, dts, ys in pbar:
            xs = xs.to(device)
            dts = dts.to(device)
            ys = ys.to(device)
            seq_len, batch, dim = xs.shape
            opt.zero_grad()
            preds = model(xs, dts)
            loss = loss_fn(preds, ys)
            loss.backward()
            opt.step()
            pbar.set_postfix(loss=float(loss.detach().cpu()))
        # validate
        model.eval()
        all_y = []
        all_p = []
        with torch.no_grad():
            for xs, dts, ys in val_loader:
                xs = xs.to(device); dts = dts.to(device)
                preds = model(xs, dts)
                all_y.append(ys.numpy())
                all_p.append(preds.cpu().numpy())
        all_y = np.concatenate(all_y)
        all_p = np.concatenate(all_p)
        print("Val AUROC:", roc_auc_score(all_y, all_p), "AUPRC:", average_precision_score(all_y, all_p))

    # SHAP explanation for a single patient (using KernelExplainer on small sample)
    # convert model to a wrapper function for shap
    def model_predict(np_x):
        # np_x: (n_samples, seq, dim)
        arr = torch.tensor(np_x.transpose(1,0,2), dtype=torch.float32)  # seq,batch,dim
        # compute delta times approximate zeros for explainer simplicity
        seq_len, batch, dim = arr.shape
        dt = torch.zeros(seq_len, batch)
        with torch.no_grad():
            out = model(arr, dt)
        return out.cpu().numpy()

    # pick background: 20 random samples
    bg = xs.cpu().numpy().transpose(1,0,2)[:20]
    explainer = shap.KernelExplainer(model_predict, bg)
    # explain first validation example
    sample = xs.cpu().numpy().transpose(1,0,2)[0:1]
    shap_vals = explainer.shap_values(sample, nsamples=50)
    print("SHAP run complete (example).")
    # Save model
    torch.save(model.state_dict(), "ft1lstm.pt")
    return model, ft, dataset

if __name__ == "__main__":
    train_main(device="cpu")
