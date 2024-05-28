
import os
import glob
import time
import torch
import random
from torch.utils.data import DataLoader

# from models.vae import VqVaeModule
from models.seq2seq import Seq2SeqModule
from my_datasets import MidiDataset, SeqCollator
from utils import description_control_iterator, medley_iterator
from input_representation import remi2midi
import uuid

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("DEVICE", device)

MODEL = os.getenv('MODEL', '')

ROOT_DIR = os.getenv('ROOT_DIR', './lmd_full')
OUTPUT_DIR = os.getenv('OUTPUT_DIR', './samples')
MAX_N_FILES = int(float(os.getenv('MAX_N_FILES', -1)))
MAX_ITER = int(os.getenv('MAX_ITER', 16_000))
MAX_BARS = int(os.getenv('MAX_BARS', 32))

MAKE_MEDLEYS = os.getenv('MAKE_MEDLEYS', 'False') == 'True'
N_MEDLEY_PIECES = int(os.getenv('N_MEDLEY_PIECES', 2))
N_MEDLEY_BARS = int(os.getenv('N_MEDLEY_BARS', 16))
  
CHECKPOINT = os.getenv('CHECKPOINT', None)
VAE_CHECKPOINT = os.getenv('VAE_CHECKPOINT', None)
BATCH_SIZE = int(os.getenv('BATCH_SIZE', 1))
VERBOSE = int(os.getenv('VERBOSE', 2))
DATASET_PATH = os.getenv('DATASET_PATH', ".")

# Added for our experiment
ALTER_DESCRIPTION = os.getenv('ALTER_DESCRIPTION', 'False') == 'True'


def save_dataset_element(model, batch, 
  initial_context=1, 
  output_dir=None, 
  max_iter=-1, 
  max_bars=-1,
  verbose=0,
):
  batch_size, seq_len = batch['input_ids'].shape[:2]

  batch_ = { key: batch[key][:, :initial_context].to(device) for key in ['input_ids', 'bar_ids', 'position_ids'] }
  if model.description_flavor in ['description', 'both']:
    batch_['description'] = batch['description'].to(device)
    batch_['desc_bar_ids'] = batch['desc_bar_ids'].to(device)
  if model.description_flavor in ['latent', 'both']:
    batch_['latents'] = batch['latents'].to(device)

  max_len = seq_len + 1024
  if max_iter > 0:
    max_len = min(max_len, initial_context + max_iter)
  if verbose:
    print(f"Generating sequence ({initial_context} initial / {max_len} max length / {max_bars} max bars / {batch_size} batch size)")
  hidden_state, desc = model.get_first_encoded_state(batch_, max_length=max_len, max_bars=max_bars, verbose=verbose//2)

  sample_id = str(uuid.uuid4())

  print("HIDDEN STATE SHAPE", hidden_state.shape)
  print("DESCRIPTION SHAPE", batch_['description'].shape)


  torch.save(hidden_state.cpu(), os.path.join(DATASET_PATH, f"{sample_id}_hidden.pt"))
  torch.save(desc.cpu(), os.path.join(DATASET_PATH, f"{sample_id}_desc.pt"))

  


def main():
  if MAKE_MEDLEYS:
    max_bars = N_MEDLEY_PIECES * N_MEDLEY_BARS
  else:
    max_bars = MAX_BARS

  if OUTPUT_DIR:
    params = []
    if MAKE_MEDLEYS:
      params.append(f"n_pieces={N_MEDLEY_PIECES}")
      params.append(f"n_bars={N_MEDLEY_BARS}")
    if MAX_ITER > 0:
      params.append(f"max_iter={MAX_ITER}")
    if MAX_BARS > 0:
      params.append(f"max_bars={MAX_BARS}")
    output_dir = os.path.join(OUTPUT_DIR, MODEL, ','.join(params))
  else:
    raise ValueError("OUTPUT_DIR must be specified.")

  print(f"Saving generated files to: {output_dir}")

  if VAE_CHECKPOINT:
    print("Loading VAE Model from checkpoint ...")
    vae_module = VqVaeModule.load_from_checkpoint(VAE_CHECKPOINT)
    vae_module.cpu()
  else:
    vae_module = None

  print("Loading Seq2Seq Model from checkpoint ...")
  model = Seq2SeqModule.load_from_checkpoint(CHECKPOINT)
  
  model.to(device)
  model.freeze()
  model.eval()


  midi_files = glob.glob(os.path.join(ROOT_DIR, '**/*.mid'), recursive=True)
  
  dm = model.get_datamodule(midi_files, vae_module=vae_module)
  dm.setup('test')
  # TODO: Get back to this
  # midi_files = dm.test_ds.files
  random.shuffle(midi_files)

  if MAX_N_FILES > 0:
    midi_files = midi_files[:MAX_N_FILES]


  description_options = None
  if MODEL in ['figaro-no-inst', 'figaro-no-chord', 'figaro-no-meta']:
    description_options = model.description_options

  dataset = MidiDataset(
    midi_files,
    max_len=-1,
    description_flavor=model.description_flavor,
    description_options=description_options,
    max_bars=model.context_size,
    vae_module=vae_module
  )


  start_time = time.time()
  coll = SeqCollator(context_size=-1)
  dl = DataLoader(dataset, batch_size=BATCH_SIZE, collate_fn=coll)

  if MAKE_MEDLEYS:
    dl = medley_iterator(dl, 
      n_pieces=N_MEDLEY_BARS, 
      n_bars=N_MEDLEY_BARS, 
      description_flavor=model.description_flavor
    )

  if ALTER_DESCRIPTION:
    dl = description_control_iterator(dl)
  
  with torch.no_grad():
    for batch in dl:
      save_dataset_element(model, batch, 
        output_dir=output_dir, 
        max_iter=MAX_ITER, 
        max_bars=max_bars,
        verbose=VERBOSE,
      )

if __name__ == '__main__':
  main()
