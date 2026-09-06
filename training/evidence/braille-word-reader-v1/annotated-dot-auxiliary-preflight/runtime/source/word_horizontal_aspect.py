"""Opt-in per-record horizontal aspect factors from the historical camera range."""
import numpy as np

HORIZONTAL_ASPECT_RANGE=(.55,1.25)
HORIZONTAL_STREAM_ID=0x48415350

def horizontal_aspect_factors(seed,dataset_epoch,record_indices):
    for value in (seed,dataset_epoch):
        if isinstance(value,(bool,np.bool_)) or not isinstance(value,(int,np.integer)) or value<0:
            raise ValueError('Seed and dataset epoch must be nonnegative integers')
    indices=np.asarray(record_indices)
    if indices.ndim!=1 or (len(indices) and (not np.issubdtype(indices.dtype,np.integer) or np.any(indices<0))):
        raise ValueError('Record indices must be a one-dimensional nonnegative integer sequence')
    return np.asarray([np.random.default_rng(np.random.SeedSequence([int(seed),int(dataset_epoch),int(index),HORIZONTAL_STREAM_ID])).uniform(*HORIZONTAL_ASPECT_RANGE)
                       for index in indices],dtype=np.float64)
