"""Copyright (c) 2020 AIT Lab, ETH Zurich

Students and holders of copies of this code, accompanying datasets,
and documentation, are not allowed to copy, distribute or modify
any of the mentioned materials beyond the scope and duration of the
Machine Perception course projects.

That is, no partial/full copy nor modification of this code and
accompanying data should be made publicly or privately available to
current/future students or other parties.
"""

"""Data-source definitions (one class per file)."""
from .hdf5 import HDF5Source
from .unityeyes import UnityEyes
__all__ = ('HDF5Source', 'UnityEyes')
