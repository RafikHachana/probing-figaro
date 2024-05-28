import torch
from torch.nn.utils.rnn import pad_sequence
from copy import deepcopy
import alter_description
import constants
import numpy as np

def combine_batches(batches, bars_per_sequence=8, description_flavor='none', device=None):
  if device is None:
    device = batches[0]['input_ids'].device

  batch_size = batches[0]['input_ids'].size(0)

  zero = torch.zeros(1, device=device, dtype=torch.int)

  contexts = []
  batch_ = {}

  for i in range(batch_size):
    curr_bar = 0
    ctx = {
      'input_ids': [],
      'bar_ids': [],
      'position_ids': [],
      'slices': [],
      'description': [],
      'desc_bar_ids': [],
      'desc_slices': [],
      'latents': [],
      'latent_slices': [],
      'files': [],
    }

    for batch in batches:
      if i >= batch['input_ids'].size(0):
        continue

      curr = curr_bar

      bar_ids = batch['bar_ids'][i]
      starts = (bar_ids >= curr).nonzero()
      ends = (bar_ids >= max(1, curr) + bars_per_sequence).nonzero()
      if starts.size(0) == 0:
        continue
      start = starts[0, 0]

      if ends.size(0) == 0:
        end = bar_ids.size(0)
        curr_bar = bar_ids[-1] + 1
      else:
        end = ends[0, 0]
        curr_bar = bar_ids[end]

      if description_flavor in ['description', 'both']:
        desc_bar_ids = batch['desc_bar_ids'][i]
        desc_start = (desc_bar_ids >= curr).nonzero()[0, 0]
        desc_ends = (desc_bar_ids >= max(1, curr) + bars_per_sequence).nonzero()

        if desc_ends.size(0) == 0:
          desc_end = desc_bar_ids.size(0)
        else:
          desc_end = desc_ends[0, 0]

      if description_flavor in ['latent', 'both']:
        latent_start = curr
        latent_end = max(1, curr) + bars_per_sequence


      ctx['input_ids'].append(batch['input_ids'][i, start:end])
      ctx['bar_ids'].append(batch['bar_ids'][i, start:end])
      ctx['position_ids'].append(batch['position_ids'][i, start:end])
      ctx['slices'].append((start, end))
      if description_flavor in ['description', 'both']:
        ctx['description'].append(batch['description'][i, desc_start:desc_end])
        ctx['desc_bar_ids'].append(batch['desc_bar_ids'][i, desc_start:desc_end])
        ctx['desc_slices'].append((desc_start, desc_end))
      if description_flavor in ['latent', 'both']:
        ctx['latents'].append(batch['latents'][i, latent_start:latent_end])
        ctx['latent_slices'].append((latent_start, latent_end))
      ctx['files'].append(batch['files'][i])

    if len(ctx['files']) <= 1:
      continue
  
    keys = ['input_ids', 'bar_ids', 'position_ids', 'description', 'desc_bar_ids', 'latents']
    for key in keys:
      if key in ctx and len(ctx[key]) > 0:
        ctx[key] = torch.cat(ctx[key])
    ctx['labels'] = torch.cat([ctx['input_ids'][1:], zero])
    ctx['files'] = '__'.join(ctx['files']).replace('.mid', '') + '.mid'

    contexts.append(ctx)

  batch_['files'] = [ctx['files'] for ctx in contexts]

  for key in ['input_ids', 'bar_ids', 'position_ids', 'description', 'desc_bar_ids', 'latents', 'labels']:
    xs = [ctx[key] for ctx in contexts if isinstance(ctx[key], torch.Tensor)]
    if len(xs) > 0:
      xs = pad_sequence(xs, batch_first=True, padding_value=0)
      if not key in ['latents']:
        xs = xs.long()
      batch_[key] = xs

  return batch_


def medley_iterator(dl, n_pieces=2, n_bars=8, description_flavor='none'):
  dl_iter = iter(dl)
  try:
    while True:
      batches = [next(dl_iter) for _ in range(n_pieces)]
      batch = combine_batches(batches, 
        bars_per_sequence=n_bars, 
        description_flavor=description_flavor
      )
      yield batch
  except StopIteration:
    return
  
def generate_controlled_ordinal_batches(description):
  attribute_keys = {
    "mean_pitch": constants.MEAN_PITCH_KEY,
    "mean_duration": constants.MEAN_DURATION_KEY,
    "mean_velocity": constants.MEAN_VELOCITY_KEY,
    "note_density": constants.NOTE_DENSITY_KEY
  }

  BINS = 33
  result = []
  for k, v in attribute_keys.items():
    tmp = deepcopy(description)
    # delta = np.random.randint(0, BINS)
    tmp['description'], deltas = alter_description.control_ordinal_attributes_batch_randomize(tmp['description'], attribute_key=v, n_bins=BINS)
    tmp['files'] = [x[:-4] + f'__altered_{k}_({d}).mid' for x, d in zip(tmp['files'], deltas)]

    yield tmp

  return

def generate_controlled_batches(batch):

  # Remove a random instrument
  tmp = deepcopy(batch)
  tmp['description'], instr_names = alter_description.remove_random_instrument_batch(tmp['description'])
  tmp['files'] = [x[:-4] + f'__remove_rand_inst_({instr_name}).mid' for x, instr_name in zip(tmp['files'], instr_names)]

  yield tmp

  # # Transpose the chord progression
  # tmp = deepcopy(batch)
  # # delta = np.random.randint(0, 12)
  # tmp['description'], deltas = alter_description.transpose_the_chord_progression_batch_randomize(description=tmp['description'])
  # tmp['files'] = [x[:-4] + f'__transposed_chords_({d}).mid' for x, d in zip(tmp['files'], deltas)]
  # yield tmp


  # yield from generate_controlled_ordinal_batches(batch)
  return

  
def description_control_iterator(dl):
  dl_iter = iter(dl)
  try:
    while True:
      batch = next(dl_iter)
      # yield batch
      # for b in altered_batches:
      # yield b
      yield from generate_controlled_batches(batch)
  except StopIteration:
    return
