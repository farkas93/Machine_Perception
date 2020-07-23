"""Copyright (c) 2020 AIT Lab, ETH Zurich

Students and holders of copies of this code, accompanying datasets,
and documentation, are not allowed to copy, distribute or modify
any of the mentioned materials beyond the scope and duration of the
Machine Perception course projects.

That is, no partial/full copy nor modification of this code and
accompanying data should be made publicly or privately available to
current/future students or other parties.
"""

"""HDF5 data source for gaze estimation."""
from threading import Lock
from typing import List

import cv2 as cv
import h5py
import numpy as np
import tensorflow as tf

from core import BaseDataSource

import random
from random import randint

class HDF5Source(BaseDataSource):
    """HDF5 data loading class (using h5py)."""

    def __init__(self,
                 tensorflow_session: tf.Session,
                 batch_size: int,
                 hdf_path: str,
                 use_colour: bool=False,
                 keys_to_use: List[str]=None,
                 entries_to_use: List[str]=None,
                 testing=False,
                 augmentation=False,
                 brightness=(0,0),
                 saturation=(0,0),
                 **kwargs):
        """Create queues and threads to read and preprocess data from specified keys."""
        hdf5 = h5py.File(hdf_path, 'r')
        self._short_name = 'HDF:%s' % '/'.join(hdf_path.split('/')[-2:])
        if testing:
            self._short_name += ':test'

        # Cache other settings
        self._use_colour = use_colour

        # Create global index over all specified keys
        if keys_to_use is None:  # use all available keys if not specified
            keys_to_use = list(hdf5.keys())
        self._index_to_key = {}
        index_counter = 0
        for key in keys_to_use:
            n = next(iter(hdf5[key].values())).shape[0]
            for i in range(n):
                self._index_to_key[index_counter] = (key, i)
                index_counter += 1
        self._num_entries = index_counter

        if entries_to_use is None:  # use all available input data if not specified
            entries_to_use = list(next(iter(hdf5.values())).keys())
        self.entries_to_use = entries_to_use

        self._hdf5 = hdf5
        self._mutex = Lock()
        self._current_index = 0
        self._use_data_augmentation = augmentation
        self._brightness = brightness
        self._saturation = saturation
        super().__init__(tensorflow_session, batch_size=batch_size, testing=testing, **kwargs)

        # Set index to 0 again as base class constructor called HDF5Source::entry_generator once to
        # get preprocessed sample.
        self._current_index = 0
        random.seed(420)            # Fixed random seed to make it reproducible

    @property
    def num_entries(self):
        """Number of entries in this data source."""
        return self._num_entries

    @property
    def short_name(self):
        """Short name specifying source HDF5."""
        return self._short_name

    def cleanup(self):
        """Close HDF5 file before running base class cleanup routine."""
        super().cleanup()

    def reset(self):
        """Reset index."""
        with self._mutex:
            super().reset()
            self._current_index = 0

    def entry_generator(self, yield_just_one=False):
        """Read entry from HDF5."""
        try:
            while range(1) if yield_just_one else True:
                with self._mutex:
                    if self._current_index >= self.num_entries:
                        if self.testing:
                            break
                        else:
                            self._current_index = 0
                    current_index = self._current_index
                    self._current_index += 1

                key, index = self._index_to_key[current_index]
                data = self._hdf5[key]
                entry = {}
                for name in self.entries_to_use:
                    entry[name] = data[name][index, :]
                yield entry
        finally:
            # Execute any cleanup operations as necessary
            pass

    def preprocess_entry(self, entry):
        """Normalize image intensities."""
        for k, v in entry.items():
            if v.ndim == 3:  # We get histogram-normalized BGR inputs
                if self._use_data_augmentation:
                    # brightness change for image augmentation
                    hsv = cv.cvtColor(v, cv.COLOR_BGR2HSV)
                        
                    h, s, val = cv.split(hsv)
                    if self._brightness is not (0,0):
                        update = randint(self._brightness[0], self._brightness[1])     # (-100, 200)
                        if update >= 0:
                            limit = 255 - update
                            val[val > limit] = 255
                            val[val <= limit] += update
                        else:
                            update = -1 * update
                            limit = 0 + update
                            val[val <= limit] = 0
                            val[val > limit] -= update

                    if self._saturation is not (0,0):
                        update = randint(self._saturation[0], self._saturation[1])     # less then 255 to make it less extreme (maybe -100 to darken image could be too much)
                        if update >= 0:
                            limit = 255 - update
                            s[s > limit] = 255
                            s[s <= limit] += update
                        else:
                            update = -1 * update
                            limit = 0 + update
                            s[s <= limit] = 0
                            s[s > limit] -= update
                    hsv = cv.merge((h, s, val))
                    v = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)                

                if not self._use_colour:
                    v = cv.cvtColor(v, cv.COLOR_BGR2GRAY)
                v = v.astype(np.float32)
                v *= 2.0 / 255.0
                v -= 1.0
                if self._use_colour and self.data_format == 'NCHW':
                    v = np.transpose(v, [2, 0, 1])
                elif not self._use_colour:
                    v = np.expand_dims(v, axis=0 if self.data_format == 'NCHW' else -1)
            
                if k == 'eye-region':
                    v = cv.resize(v, (448,120))
                entry[k] = v

        # Ensure all values in an entry are 4-byte floating point numbers
        for key, value in entry.items():
            entry[key] = value.astype(np.float32)

        return entry
