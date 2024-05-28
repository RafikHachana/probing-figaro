import constants
from vocab import DescriptionVocab
import torch
import numpy as np

def control_ordinal_attributes(description, delta=1, attribute_key=constants.MEAN_PITCH_KEY, n_bins=33):
    desc_vocab = DescriptionVocab()
    description = desc_vocab.decode(description[0])
    # print("Given description", description)
    result = []
    for token in description:
        tmp = token
        if len(token.split('_')) == 2 and token.split('_')[0] == attribute_key:
            index = int(token.split('_')[1])
            new_index = index + delta

            # Clip the values
            new_index = min(n_bins-1, max(0, new_index))

            tmp = f'{attribute_key}_{new_index}'

        result.append(tmp)

    # print("Altered description", result)
    
    result = desc_vocab.encode(result)
    return torch.Tensor([result])

def _alter_description_batch(description, func, **kwargs):
    result = []
    for i in range(len(description)):
        result.append(func(description[i:i+1], **kwargs))
    return torch.cat(result)

def control_ordinal_attributes_batch(description, delta=1, attribute_key=constants.MEAN_PITCH_KEY, n_bins=33):
    return _alter_description_batch(description, control_ordinal_attributes, delta=delta, attribute_key=attribute_key, n_bins=n_bins)

def transpose_the_chord_progression_batch(description, delta=1):
    return _alter_description_batch(description, transpose_the_chord_progression, delta=delta)

def remove_most_common_instrument_batch(description):
    return _alter_description_batch(description, remove_most_common_instrument)

def remove_random_instrument_batch(description):
    result = []
    removed_instruments = []
    for i in range(len(description)):
        # result.append(func(description[i:i+1], **kwargs))
        tmp, removed_instrument = remove_random_instrument(description[i:i+1])
        result.append(tmp)
        removed_instruments.append(removed_instrument)
    return torch.cat(result), removed_instruments
    # return _alter_description_batch(description, remove_random_instrument)

def _alter_description_batch_random_batch(description, func, **kwargs):
    result = []
    deltas = []
    for i in range(len(description)):
        delta = 0
        while delta == 0:
            delta = np.random.randint(-33, 33)
        deltas.append(delta)
        result.append(func(description[i:i+1], delta=delta, **kwargs))
    return torch.cat(result), deltas

def control_ordinal_attributes_batch_randomize(description, attribute_key=constants.MEAN_PITCH_KEY, n_bins=33):
    return _alter_description_batch_random_batch(description, control_ordinal_attributes, attribute_key=attribute_key, n_bins=n_bins)

def transpose_the_chord_progression_batch_randomize(description):
    return _alter_description_batch_random_batch(description, transpose_the_chord_progression)

# TODO: Fix repetitive code
def transpose_the_chord_progression(description, delta=1):
    pitch_classes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

    desc_vocab = DescriptionVocab()
    description = desc_vocab.decode(description[0])
    # print("Given description", description)
    result = []
    for token in description:
        tmp = token
        if len(token.split('_')) == 2 and token.split('_')[0] == constants.CHORD_KEY and token.split('_')[1] != "N:N":
            pitch, quality = token.split('_')[1].split(":")
            pitch_index = None
            for i, p in enumerate(pitch_classes):
                if p == pitch:
                    pitch_index = i
                    break

            new_pitch_index = (pitch_index + delta)%len(pitch_classes)

            tmp = f'{constants.CHORD_KEY}_{pitch_classes[new_pitch_index]}:{quality}'

        result.append(tmp)

    # print("Altered description", result)
    
    result = desc_vocab.encode(result)
    return torch.Tensor([result])


def remove_most_common_instrument(description):
    instrument_counts = {}

    desc_vocab = DescriptionVocab()
    description = desc_vocab.decode(description[0])
    # print("Given description", description)
    result = []
    for token in description:
        # tmp = token
        if len(token.split('_')) == 2 and token.split('_')[0] == constants.INSTRUMENT_KEY:
            if token not in instrument_counts:
                instrument_counts[token] = 0
            instrument_counts[token] += 1

        # result.append(tmp)

    most_common_instrument_token = None
    count = 0
    for k, v in instrument_counts.items():
        if v > count:
            count = v
            most_common_instrument_token = k

    # print("Most common instrument", most_common_instrument_token)

    # Now we reconstruct the description and skip the most common instrument in the sequence
    for token in description:
        if token == most_common_instrument_token:
            continue

        result.append(token)

    # print("Altered description", result)
    
    result = desc_vocab.encode(result)
    return torch.Tensor([result])


def remove_random_instrument(description):
    instrument_counts = {}

    desc_vocab = DescriptionVocab()
    description = desc_vocab.decode(description[0])
    # print("Given description", description)
    result = []
    for token in description:
        if len(token.split('_')) == 2 and token.split('_')[0] == constants.INSTRUMENT_KEY:
            if token not in instrument_counts:
                instrument_counts[token] = 0
            instrument_counts[token] += 1


    ind = np.random.randint(0, len(instrument_counts))

    most_common_instrument_token = list(instrument_counts.items())[ind][0]

    # print("Most common instrument", most_common_instrument_token)

    # Now we reconstruct the description and skip the most common instrument in the sequence
    skipped_instruments = 0
    for token in description:
        if token == most_common_instrument_token:
            skipped_instruments += 1
            continue

        result.append(token)

    # Re-pad the sequence to preserve length
    for i in range(skipped_instruments):
        result.append(constants.PAD_TOKEN)

    # print("Altered description", result)
    
    result = desc_vocab.encode(result)
    return torch.Tensor([result]), most_common_instrument_token



# FIXME: Only supports BATCH_SIZE=1
def change_mean_pitch(description, delta=1):
    desc_vocab = DescriptionVocab()
    description = desc_vocab.decode(description[0])
    # print("Given description", description)
    result = []
    for token in description:
        tmp = token
        if len(token.split('_')) == 2 and token.split('_')[0] == constants.MEAN_PITCH_KEY:
            index = int(token.split('_')[1])
            new_index = index + delta

            # Clip the values
            new_index = min(32, max(0, new_index))

            tmp = f'{constants.MEAN_PITCH_KEY}_{new_index}'

        result.append(tmp)

    # print("Altered description", result)
    
    result = desc_vocab.encode(result)
    return torch.Tensor([result])


def change_mean_duration(description, delta=1):
    pass

def change_mean_velocity(description, delta=1):
    pass

def change_note_density(description, delta=1):
    pass
