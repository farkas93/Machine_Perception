# Eye Gaze Estimation Skeleton Code
Visit [here](https://ait.ethz.ch/teaching/courses/2020-SS-Machine-Perception/) for more information about the Machine Perception course.

All questions should first be directed to [our course Piazza](https://piazza.com/class/k6jbn40blq36yx) before being sent to my [e-mail address](mailto:spark@inf.ethz.ch).

## Setup

The following two steps will prepare your environment to begin training and evaluating models.

### Necessary datasets

The required training datasets are located on the Leonhard cluster at `/cluster/project/infk/hilliges/lectures/mp20/project3`.

Please create a symbolic link via commands similar to the following:
```
    cd datasets/
    ln -s /cluster/project/infk/hilliges/lectures/mp20/project3/mp20_train.h5
    ln -s /cluster/project/infk/hilliges/lectures/mp20/project3/mp20_validation.h5
    ln -s /cluster/project/infk/hilliges/lectures/mp20/project3/mp20_test_students.h5
```

### Installing dependencies

Run (with `sudo` appended if necessary),
```
pip install -r requirements.txt
```

Note that this can be done within a [virtual environment](https://docs.python.org/3/tutorial/venv.html). In this case, the sequence of commands would be similar to:
```
    mkvirtualenv -p $(which python3) myenv
    pip install -r requirements.txt
```

when using [virtualenvwrapper](https://virtualenvwrapper.readthedocs.io/en/latest/).

## Structure

* `datasets/` - all data sources required for training/validation/testing.
* `outputs/` - any output for a model will be placed here, including logs, summaries, checkpoints, and submission `.txt.gz` files.
* `src/` - all source code.
    * `core/` - base classes
    * `datasources/` - routines for reading and preprocessing entries for training and testing
    * `models/` - neural network definitions
    * `util/` - utility methods
    * `train_example.py` - example training script

## Creating your own model
### Model definition
To create your own neural network, do the following:
1. Make a copy of `src/models/example.py`. For the purpose of this documentation, let's call the new file `newmodel.py` and the class within `NewModel`.
2. Now edit `src/models/__init__.py` and insert the new model by making it look like:

```
from .example import ExampleNet
from .newmodel import NewModel
__all__ = ('ExampleNet', 'NewModel')
```

3. Lastly, make a copy or edit `src/train_example.py` such that it imports and uses class `NewModel` instead of `ExampleNet`.

### Training the model
If your training script is called `train_example.py`, simply `cd` into the `src/` directory and run
```
python3 train_example.py
```

### Outputs
When your model has completed training, it will perform a full evaluation on the test set. For class `ExampleNet`, this output can be found in the folder `outputs/ExampleNet/` as `predictions_to_upload_XXXXXXXXX.txt.gz`.

Submit this `txt.gz` file to our page on [our competition website](https://machine-perception.ait.ethz.ch/).
