# Helloworld ML App

Template for a new Python ML project.

## Dev environment setup

Python 3 is the main language used in this codebase.
We strongly encourage the use of Python [virtual environments](http://docs.python-guide.org/en/latest/dev/virtualenvs/):

    virtualenv venv -p /usr/bin/python3
    source venv/bin/activate

After which, you can install the required Python modules via

    pip install -r requirements.txt

## Data

All processed data for training, evaluation, etc can be found in the [`data`](data/) folder.
See the [README](data/README.md) for information about the different datasets available.

## Model training and evaluation

### Step 1: Convert instances to sparse vectors (Featurization)

The [`ycml.scripts.featurize`](../ycml/scripts/featurize.py) script will perform the necessary steps to convert JSON instance files into featurized `numpy` arrays.

Example:

    python -m ycml.scripts.featurize --settings settings/development.settings.yaml -i ./data/train.json.gz --fit models/development.featurizer.gz -o data/train.features.npz
    python -m ycml.scripts.featurize --settings settings/development.settings.yaml -i ./data/evaluate.json.gz --featurize models/development.featurizer.gz -o data/evaluate.features.npz

You can set parameters for the featurizer through the settings file directly (or use the default).

The following featurizers are available:

- `helloworld.featurizers.HelloWorldFeaturizer`: Generates a bunch of random features.

### Step 2: Train/Evaluate model

Fitting a classifier:

    python -m ycml.scripts.train --settings settings/development.settings.yaml -f data/train.features.npz -o models/development.classifier.gz

Evaluating a classifier:

    python -m ycml.scripts.evaluate --settings settings/development.settings.yaml -c models/development.classifier.gz -f data/evaluate.features.npz

There are various options to save threshold values to file, save and load probabilities from file, and generate PR curves for analysis.

Making predictions:

    python -m ycml.scripts.predict --settings settings/development.settings.yaml data/evaluate.json.gz
