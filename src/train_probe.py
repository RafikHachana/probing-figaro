from models.probe import ProbeClassificationTwoLayer


"""
Simple training loop; Boilerplate that could apply to any arbitrary neural network,
so nothing in this file really has anything to do with GPT specifically.
"""
import os
import math
import logging
import glob

from tqdm import tqdm
import numpy as np
import json
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data.dataloader import DataLoader
from matplotlib import pyplot as plt
from models.probe import ProbeClassificationTwoLayer
from vocab import DescriptionVocab, Tokens
from torch.utils.data import Dataset, random_split

from constants import (
  DEFAULT_NOTE_DENSITY_BINS,
  DEFAULT_MEAN_VELOCITY_BINS,
  DEFAULT_MEAN_PITCH_BINS,
  DEFAULT_MEAN_DURATION_BINS
)


from constants import (

  NOTE_DENSITY_KEY,
  MEAN_PITCH_KEY,
  MEAN_VELOCITY_KEY,
  MEAN_DURATION_KEY,
)

logger = logging.getLogger(__name__)

class ProbingDataset(Dataset):
    def __init__(self, act, y):
        assert len(act) == len(y)
        print(f"{len(act)} pairs loaded...")
        self.act = act
        self.y = y
        # print(np.sum(np.array(y)==0), np.sum(np.array(y)==1), np.sum(np.array(y)==2))
    def __len__(self, ):
        return len(self.y)
    def __getitem__(self, idx):
        return self.act[idx], torch.tensor(self.y[idx]).to(torch.long)

class TrainerConfig:
    # optimization parameters
    max_epochs = 100
    batch_size = 64
    learning_rate = 3e-3
    betas = (0.9, 0.95)
    grad_norm_clip = 1.0
    weight_decay = 0.1 # only applied on matmul weights
    # learning rate decay params: linear warmup followed by cosine decay to 10% of original
    lr_decay = False
    warmup_tokens = 375e6 # these two numbers come from the GPT-3 paper, but may not be good defaults elsewhere
    final_tokens = 260e9 # (at what point we reach 10% of original LR)
    # checkpoint settings
    ckpt_path = "."
    num_workers = 0 # for DataLoader

    def __init__(self, **kwargs):
        for k,v in kwargs.items():
            setattr(self, k, v)

DATASET_PATH = os.getenv('DATASET_PATH', ".")
TOKEN_CLASS_OF_INTEREST = os.getenv('TOKEN_CLASS_OF_INTEREST', "chords")


class Trainer:
    def __init__(self, model, train_dataset, test_dataset, config):
        self.model = model
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.config = config

        # take over whatever gpus are on the system
        self.device = 'cpu'
        if torch.cuda.is_available():
            self.device = torch.cuda.current_device()
            self.model = torch.nn.DataParallel(self.model).to(self.device)
            
        # log something for plotting
        self.train_loss_cont = []
        self.test_loss_cont = []
        self.train_acc_cont = []
        self.test_acc_cont = []
        # would be a list of T-long, each is a lits of 60-long, for stratified accuracies        
        self.train_strat_acc_cont = []
        self.test_strat_acc_cont = []
        
    def flush_plot(self, ):
        # plt.close()
        fig, axs = plt.subplots(1, 2, figsize=(20, 10), dpi= 80, facecolor='w', edgecolor='k')
        axs = axs.flat
        axs[0].plot(self.train_loss_cont, label="train")
        axs[0].plot(self.test_loss_cont, label="test")
        axs[0].set_title("Loss")
        axs[0].legend()
        axs[1].plot(self.train_acc_cont, label="train")
        axs[1].plot(self.test_acc_cont, label="test")
        axs[1].set_title("Accuracy")
        axs[1].legend()
        plt.show()

    def save_traces(self, ):
        tbd = {
            "train_loss_cont": self.train_loss_cont, "test_loss_cont" :self.test_loss_cont, 
            "train_acc_cont": self.train_acc_cont, "test_acc_cont": self.test_acc_cont, 
            "train_strat_acc_cont": self.train_strat_acc_cont, "test_strat_acc_cont": self.test_strat_acc_cont, 
        }
        with open(os.path.join(self.config.ckpt_path, "tensorboard.txt"), "w") as f:
            f.write(json.dumps(tbd) + "\n")

    def save_checkpoint(self):
        # DataParallel wrappers keep raw model object in .module attribute
        raw_model = self.model.module if hasattr(self.model, "module") else self.model
        if not os.path.exists(self.config.ckpt_path):
            os.makedirs(self.config.ckpt_path)
        torch.save(raw_model.state_dict(), os.path.join(self.config.ckpt_path, "probe_checkpoint.ckpt"))

    def train(self, prt=True):
        model, config = self.model, self.config
        raw_model = model.module if hasattr(self.model, "module") else model
        optimizer, scheduler = raw_model.configure_optimizers(config)

        def run_epoch(split):
            is_train = split == 'train'
            model.train(is_train)
            data = self.train_dataset if is_train else self.test_dataset
            loader = DataLoader(data, shuffle=True, pin_memory=True,
                                batch_size=config.batch_size,
                                num_workers=config.num_workers)

            losses = []
            # totals_epoch = np.zeros(60, dtype=float)  # np.array of shape [60], for positions of age 0 to 59
            # hits_epoch = np.zeros(60, dtype=float)  # np.array of shape [60], for positions of age 0 to 59
            pbar = tqdm(enumerate(loader), total=len(loader), disable=not prt) if is_train else enumerate(loader)
            total_hits = 0
            total_samples = 0
            for it, (x, y) in pbar:
                x = x.to(self.device)  # [B, f]
                y = y.to(self.device)  # [B, #task=64] 

                with torch.set_grad_enabled(is_train):
                    logits, loss = model(x, y)
                    loss = loss.mean() # collapse all losses if they are scattered on multiple gpus
                    losses.append(loss.item())
                    # totals_epoch += np.array([torch.sum(age == i).item() for i in range(60)]).astype(float)
                    y_hat = torch.argmax(logits, dim=-1, keepdim=False)  # [B, #task]
                    hits = y_hat == y  # [B, #task]
                    # hits_epoch += np.array([torch.sum(hits * (age == i)).item() for i in range(60)]).astype(float)
                    total_hits += torch.sum(hits).item()
                    total_samples += y.shape[0]*y.shape[1]
                    
                if is_train:
                    # backprop and update the parameters
                    model.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_norm_clip)
                    optimizer.step()
                    mean_loss = float(np.mean(losses))
                    mean_acc = total_hits / total_samples
                    lr = optimizer.param_groups[0]['lr']
                    pbar.set_description(f"epoch {epoch+1}: train loss {mean_loss:.5f}; lr {lr:.2e}; train acc {mean_acc*100:.2f}%")
            if is_train:
                self.train_loss_cont.append(mean_loss)
                self.train_acc_cont.append(mean_acc)
                # self.train_strat_acc_cont.append((hits_epoch / totals_epoch).tolist())

            if not is_train:
                test_loss = float(np.mean(losses))
                scheduler.step(test_loss)
                test_acc = total_hits / total_samples
                if prt: 
                    print(f"test loss {test_loss:.5f}; test acc {test_acc*100:.2f}%")
                self.test_loss_cont.append(test_loss)
                self.test_acc_cont.append(test_acc)
                # self.test_strat_acc_cont.append((hits_epoch / totals_epoch).tolist())
                return test_loss

        best_loss = float('inf')
        self.tokens = 0  # counter used for learning rate decay
        
        for epoch in range(config.max_epochs):
            print("TEST DATASET", self.test_dataset)
            run_epoch('train')
            if self.test_dataset is not None:
                test_loss = run_epoch('test')
                if test_loss < best_loss:
                    best_loss = test_loss
                    self.save_checkpoint()


def get_all_uuids(files):
    uuids = set()
    for f in files:
        uuids.add(f.split("_")[0])

    return list(uuids)

def filter_description(descriptions, tokens_to_include: list, max_length=10):
    result = []
    for description in descriptions:
        tmp = []
        for val in description:
            if val.item() in tokens_to_include:
                tmp.append(tokens_to_include.index(val.item()))
        result.append(tmp[:max_length])
    return torch.tensor(result)

if __name__ == "__main__":
    dataset_files = os.listdir(DATASET_PATH)
    print(dataset_files)

    ids = get_all_uuids(dataset_files)
    print(ids)

    device = torch.cuda.current_device()

    vocab = DescriptionVocab()
    max_length = 10
    if TOKEN_CLASS_OF_INTEREST == 'chords':
        filtering_tokens = vocab.encode(Tokens.get_chord_tokens())
    elif TOKEN_CLASS_OF_INTEREST == "instruments":
        filtering_tokens = vocab.encode(Tokens.get_instrument_tokens())
    elif TOKEN_CLASS_OF_INTEREST == "density":
        filtering_tokens = vocab.encode([f'{NOTE_DENSITY_KEY}_{i}' for i in range(len(DEFAULT_NOTE_DENSITY_BINS))])
    elif TOKEN_CLASS_OF_INTEREST == "velocity":
        filtering_tokens = vocab.encode([f'{MEAN_VELOCITY_KEY}_{i}' for i in range(len(DEFAULT_MEAN_VELOCITY_BINS))])
    elif TOKEN_CLASS_OF_INTEREST == "pitch":
        filtering_tokens = vocab.encode([f'{MEAN_PITCH_KEY}_{i}' for i in range(len(DEFAULT_MEAN_PITCH_BINS))])
    elif TOKEN_CLASS_OF_INTEREST == "duration":
        filtering_tokens = vocab.encode([f'{MEAN_DURATION_KEY}_{i}' for i in range(len(DEFAULT_MEAN_DURATION_BINS))])
    else:
        filtering_tokens = vocab.encode(Tokens.get_time_signature_tokens())
    
    # Load the dataset
    descriptions = None
    hidden_states = None
    for file_id in tqdm(ids):
        try:
            desc_batch = filter_description(torch.load(os.path.join(DATASET_PATH, f"{file_id}_desc.pt")), tokens_to_include=filtering_tokens, max_length=max_length)
            hidden_batch = torch.load(os.path.join(DATASET_PATH, f"{file_id}_hidden.pt"))

            descriptions = torch.cat([descriptions, desc_batch]) if descriptions is not None else desc_batch
            hidden_states = torch.cat([hidden_states, torch.flatten(hidden_batch, start_dim=1)]) if hidden_states is not None else torch.flatten(hidden_batch, start_dim=1)
        except:
            continue

    print("HIDDEN ALL SHAPE", hidden_states.shape)
    print("DESCRIPTION ALL SHAPE", descriptions.shape)

    dataset = ProbingDataset(hidden_states, descriptions)

    train_dataset, test_dataset = random_split(dataset, [int(0.8 * len(dataset)), len(dataset) - int(0.8 * len(dataset))])

    model = ProbeClassificationTwoLayer(
        device,
        probe_class=len(filtering_tokens),
        num_task=max_length,
        mid_dim=256,
        input_dim=256*512
    )
    trainer = Trainer(model, train_dataset, test_dataset, TrainerConfig())

    trainer.train()

