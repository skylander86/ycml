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

The [`helloworld.featurize`](featurize.py) script will perform the necessary steps to convert JSON instance files into featurized `numpy` arrays.

```
(venv) $ python -m helloworld.featurize -h
usage: featurize.py [-h] [-i [<instances> [<instances> ...]]]
                    [-o <features_uri>] [-s <settings_uri>] [--n-jobs <N>]
                    [--log-level <log_level>]
                    (-f <featurizer> | -t <featurizer> | -z <featurizer> [<featurizer> ...] | -x <features_uri> [<features_uri> ...] | -v <featurizer> <features_uri>)
                    [<featurizer_type>]

Featurize instances for ML classification.

positional arguments:
  <featurizer_type>     Name of featurizer model to use.

optional arguments:
  -h, --help            show this help message and exit
  -i [<instances> [<instances> ...]], --instances [<instances> [<instances> ...]]
                        List of instance files to featurize.
  -o <features_uri>, --output <features_uri>
                        Save featurized instances here.
  -s <settings_uri>, --settings <settings_uri>
                        Settings file to configure models.
  --n-jobs <N>          No. of processes to use during featurization.
  --log-level <log_level>
                        Set log level of logger.
  -f <featurizer>, --fit <featurizer>
                        Fit instances and save featurizer model file here.
  -t <featurizer>, --featurize <featurizer>
                        Use this featurizer to transform instances.
  -z <featurizer> [<featurizer> ...], --featurizer-info <featurizer> [<featurizer> ...]
                        Display information about featurizer model.
  -x <features_uri> [<features_uri> ...], --features-info <features_uri> [<features_uri> ...]
                        Display information about featurized instance file.
  -v <featurizer> <features_uri>, --verify <featurizer> <features_uri>
                        Verify that the featurized instance file came from the
                        same featurizer model.
```

Example:

    python -m helloworld.featurize --settings settings/development.settings.yaml -i ./data/train.json.gz --fit models/development.featurizer.gz -o data/development.features.npz

You can set parameters for the featurizer through the settings file directly (or use the default).
We store existing settings file in [`settings/`](settings/) using the naming convention of `<environment>.settings.yaml`, where settings for multiple apps are stored in a single YAML file.

The following featurizers are available:

- `HelloWorld`: Description

### Step 2: Train/Evaluate model

The [`helloworld.classify`](classify.py) script will perform the necessary steps fit/evaluate/use a model from featurized instances.

```
(venv) $ python -m helloworld.classify -h
usage: classify.py [-h] [--log-level <log_level>] [-s <settings_file>]
                   [-c <classifier_file>] [--n-jobs <N>]
                   <mode> ...

Classify instances using ML classifier.

optional arguments:
  -h, --help            show this help message and exit
  --log-level <log_level>
                        Set log level of logger.
  -s <settings_file>, --settings <settings_file>
                        Settings file to configure models.
  -c <classifier_file>, --classifier-info <classifier_file>
                        Display information about classifier.
  --n-jobs <N>          No. of processor cores to use.

Different classifier modes for fitting, evaluating, and prediction.:
  <mode>
    fit                 Fit a classifier.
    evaluate            Evaluate a classifier.
    predict             Predict using a classifier.
    info                Display information regarding classifier.
```

#### Fitting a classifier

The `fit` sub-command will fit a classifier model to training data.

```
(venv) $ python -m helloworld.classify fit -h
usage: classify.py fit [-h] [-f <featurized>] -o <classifier_file>
                       <classifier_type>

positional arguments:
  <classifier_type>     Type of classifier model to fit.

optional arguments:
  -h, --help            show this help message and exit
  -f <featurized>, --featurized <featurized>
                        Fit model on featurized instances.
  -o <classifier_file>, --output <classifier_file>
                        Save trained classifier model here.
```

Example:

    python -m helloworld.classify --settings settings/development.settings.yaml fit -f data/development.features.npz -o models/development.classifier.gz

#### Evaluating a classifier

The `evaluate` sub-command will evaluate instances using a trained model.

```
(venv) $ python -m helloworld.classify evaluate -h
usage: classify.py evaluate [-h] [-t <thresholds>] [-p <probabilities_file>]
                            <classifier_file> <featurized_file>

positional arguments:
  <classifier_file>     Model file to use for evaluation.
  <featurized_file>     Evaluate model on featurized instances.

optional arguments:
  -h, --help            show this help message and exit
  -t <thresholds>, --thresholds <thresholds>
                        Threshold file to use for prediction.
  -p <probabilities_file>, --save-probabilities <probabilities_file>
                        Save evaluation probabilities; useful for calibration.
```

Example:

    python -m helloworld.classify --settings settings/development.settings.yaml evaluate models/development.classifier.gz data/development.features.npz --save-probabilities data/development.evaluation_probabilities.npz

#### Making predictions

The `predict` sub-command will make predictions using a trained model.

```
(venv) $ python -m helloworld.classify predict -h
usage: classify.py predict [-h] [-t <thresholds>] [-o <prediction_file>]
                           [-f <format>] [-p]
                           <classifier_file> <featurized_file>

positional arguments:
  <classifier_file>     Model file to use for prediction.
  <featurized_file>     Predict labels of featurized instances.

optional arguments:
  -h, --help            show this help message and exit
  -t <thresholds>, --thresholds <thresholds>
                        Threshold file to use for prediction.
  -o <prediction_file>, --output <prediction_file>
                        Save results of prediction here.
  -f <format>, --format <format>
                        Save results of prediction using this format (defaults
                        to file extension).
  -p, --probs           Also save prediction probabilities.
```

Example:

    python -m helloworld.classify --settings settings/development.settings.yaml predict models/development.classifier.gz data/development.features.npz -p
