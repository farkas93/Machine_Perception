#!/usr/bin/env python3
"""Copyright (c) 2020 AIT Lab, ETH Zurich, Seonwook Park

Students and holders of copies of this code, accompanying datasets,
and documentation, are not allowed to copy, distribute or modify
any of the mentioned materials beyond the scope and duration of the
Machine Perception course projects.

That is, no partial/full copy nor modification of this code and
accompanying data should be made publicly or privately available to
current/future students or other parties."""

"""Main script for training a model for gaze estimation."""
import argparse
import multiprocessing
import os

import coloredlogs
import tensorflow as tf

import logging
logger = logging.getLogger(__name__)

from configs.data_location import dataconfig

from configs.gaga_config import gaga_config

if __name__ == '__main__':

    # Set global log level
    parser = argparse.ArgumentParser(description='Train a gaze estimation model.')
    parser.add_argument('-v', type=str, help='logging level', default='info',
                        choices=['debug', 'info', 'warning', 'error', 'critical'])
    parser.add_argument('--restore', type=str, help='Restore weights from this folder')
    args = parser.parse_args()
    coloredlogs.install(
        datefmt='%d/%m %H:%M',
        fmt='%(asctime)s %(levelname)s %(message)s',
        level=args.v.upper(),
    )

    # Set identifier to that provided, and restore weights from there
    identifier = None
    if dataconfig['output_local']:
        path = dataconfig['outputs_dir'] = 'K:/MLData/outputs/'
    else:
        path = os.path.abspath(os.path.dirname(__file__)) + '/../outputs'

    if args.restore is not None:
        if dataconfig['output_local']:
            identifier = args.restore
        else:
            identifier = os.path.relpath(
                args.restore,
                start=path,
            )
        logger.info('Manually selected folder to restore from: %s' % identifier)

    # Initialize Tensorflow session
    tf.logging.set_verbosity(tf.logging.INFO)
    session_config = tf.ConfigProto(
        inter_op_parallelism_threads=0,
        intra_op_parallelism_threads=multiprocessing.cpu_count(),
        allow_soft_placement=True,
        gpu_options=tf.GPUOptions(allow_growth=True),
    )
    with tf.Session(config=session_config) as session:

        # Declare some parameters
        batch_size = gaga_config['batch_size']
        learning_rate = gaga_config['learning_rate']
        data_to_retrieve = ['left-eye', 'right-eye', 'gaze', 'head']  # Available are: left-eye
                                                         #                right-eye
                                                         #                eye-region
                                                         #                face
                                                         #                gaze (required)
                                                         #                head
                                                         #                face-landmarks

        # Define model
        from datasources import HDF5Source
        from models.gaganet import GaGaJ, GaGaZs
        model = GaGaZs(
            # Tensorflow session
            # Note: The same session must be used for the model and the data sources.
            session,

            # The learning schedule describes in which order which loss term to optimize
            # at which learning rate.
            #
            # A standard network would have one entry (dict) in this argument where all model
            # parameters are optimized. To do this, you must specify which variables must be
            # optimized and this is done by specifying which prefixes to look for.
            # The prefixes are defined by using `tf.variable_scope`.
            #
            # The loss terms which can be specified depends on model specifications, specifically
            # the `loss_terms` output of `BaseModel::build_model`.
            learning_schedule=[
                {
                    'loss_terms_to_optimize': gaga_config['loss_terms'],
                    'metrics': gaga_config['metrics'],
                    'learning_rate': learning_rate,
                },
            ],

            test_losses_or_metrics=[gaga_config['loss_terms'][0], gaga_config['metrics'][0]],

            # Data sources for training and testing.
            train_data={
                'float32': HDF5Source(
                    session,
                    batch_size,
                    hdf_path=dataconfig['train_data'],
                    entries_to_use=data_to_retrieve,
                    min_after_dequeue=2000,
                    data_format='NCHW',
                    shuffle=True,
                    staging=True,
                    augmentation=gaga_config['use_augmentation'],
                    brightness=gaga_config['brightness_range'],
                    saturation=gaga_config['saturation_range']
                ),
            },
            test_data={
                'float32': HDF5Source(
                    session,
                    batch_size,
                    hdf_path=dataconfig['val_data'],
                    entries_to_use=data_to_retrieve,
                    testing=True,
                    num_threads=2,
                    data_format='NCHW',
                ),
            },

            # We may use a user-defined identifier to resume previous training or
            # perform predictions on a pre-trained model.
            identifier=identifier,
        )

        if args.restore is None:
            # Train this model for a set number of epochs
            model.start_training()

        # Evaluate for submission
        model.evaluate_for_submission(
            HDF5Source(
                session,
                batch_size,
                hdf_path=dataconfig['test_data'],
                entries_to_use=[k for k in data_to_retrieve if k != 'gaze'],
                testing=True,
                num_threads=1,
                data_format='NCHW',
            )
        )

        del model
