# utopia-mlops-assignment
Assignment for the MLOps position at Utopia Music.


## Overview
This repo contains the code to run a pipeline that comprises the several steps:

* load raw audio data from a given folder;
* process the audio files into mel-spectrograms, and save each as `.npy` file;
* create a `tf.data.Dataset` with the processed data and the automatically-extracted labels;
* build a ML model based on the user's specifications;
* train, evaluate and save the model within an `mlflow` experiment run.

## Project structure
The main files and folders of this repo are:

* `src.pipeline.classes`. The folder contains the main classes and interface
that are being used in the pipeline: 
  * `MNISTAudioDataset` provides an API to build the dataset starting from a list of files.
  The class assumes each file is an element of the dataset.
  * `Extractor` provides an API for extracting audio features from an input
  in form of `np.array`. The different audio feature extractors can be implemented
  inheriting from this abstract class.
  * `CNNModel` provides an API for building and compiling a CNN model. Some of the arguments
  required at class initialisation are `InputShape`, which defines the input dimensions
  for the model, and `ConvSpec`, which define how a `tf.Keras.layers.Conv2D` layer
  shall be built. The model can also be customised in terms of no. of hidden layers, 
  name and no. of output classes.
  * `Processor` provides an API for the data preprocessing step. In this code, an instance of
  `Processor` will be passed an instance of `Extractor` as input argument, i.e. it has an
  efferent coupling with the processor.
* `steps.py`. The script contains the different steps the pipeline is made of.
Each step represents a _task_ in Airflow terminology. The main idea underlying the script
is to model a task in a single function in a way such that it should be possible
to build a `airflow.DAG` that pipes all the functions wrapped in `PythonOperator`
classes.
* `script.py` provides an entrypoint for running the whole pipeline. When invoked, it
creates an `mlflow` experiment and a model artefact. The model is saved inside the local
`model` folder, whereas the experiment tracking logs are stored inside the local
`mlruns` folder and can be further accessed through a browser window when running
`mlflow ui`.

## Run
To run the pipeline, use the snippet below. It is assumed that a `recordings/` folder lives inside
the project folder and you can provide a `config.YAML` with all the necessary specs to configure
the pipeline. For further details on what parameters the pipeline needs for it to run, you can
inspect the examples inside the `config` folder.

> The `config.yaml` is **required**, otherwise the script will fail to run.

```shell
cd/to/repo
pip install -r requirements.txt
python script.py --config <your-config-file>
```

To load the model, you can run the following snippet inside the PyCharm Python Console:
```python
import tensorflow as tf

model_folder = "model"
model = tf.keras.models.load_model(model_folder)
```

## Discussion

The pipeline structure and the code are yet to be a fully, production-ready application due to different factors.
I would like to share some observations and ideas about possible future enhancement alongside some pain points I have 
encountered during the assignment.

### Observations on the coding part

First and foremost, I spent too much time trying to rework the training part given the change in output format for
the spectrogram i.e. from `.png` to `.npy`. That is, building the dataset in a `tf.data.Dataset` format took much 
time as the original version was using some image-specific utils. I considered using `tfio` and therefore computing
audio spectrograms within the TensorFlow environment - thus mimicking what Valerio did in his videos when using
`torchaudio`. However, I was not familiar with this library and I found the documentation not optimal overall.
Furthermore, I considered using a more _traditional_ approach for the dataset based on `np.array` structure for
features and labels - though I really though exploring the Dataset API was worth it.

Secondly, I decided **not** to focus on the transformations on the dataset i.e. flipping and rotating images, as
I found them not relevant anymore since the change in feature domain. For further exploration, I would consider the 
techniques shown in [SpecAugment](https://www.tensorflow.org/io/tutorials/audio#specaugment). All in all, my initial 
idea was to pick a framework and try to stick with it for the most part, thus the idea of leaning towards `tfio.audio`. 
Nevertheless, I still believe that `librosa` provides an easy and safe framework for all audio things.

Finally, I have not been able to rid the code of all hard-coded stuff and ad-hoc implementation e.g.

* when [shuffling the dataset](https://github.com/inspiralpatterns/utopia-mlops-assignment/blob/e1be99eed4c568a4f857daab414a8ea0b8d3120c/script.py#L22) the code 
still depends on the hard-coded size of the full dataset;
* the [input shape](https://github.com/inspiralpatterns/utopia-mlops-assignment/blob/e1be99eed4c568a4f857daab414a8ea0b8d3120c/src/pipeline/steps.py#L73) for the model 
is coupled to the shape of each processed audio file, that the model will fail is e.g. the audio file duration changes.
* most of the IO functions rely on the use of `Path`, and generators of them. 

Ideally, we would like to impute such dynamic values as input shape, dataset size etc. during the preprocessing task 
and store them as metadata so that they could be used in downstream steps. 

With respect to the ad-hoc implementations, althoughI believe the use of `Path` and friends for some sort of _lazy 
loading_ is a good approach, there should be an alternative
e.g. such functions could take either a path, or a string or a list of them, and the function could handle each case accordingly.

### Observation on MLOps tools

`mlflow` is being use for experiment tracking, as it is quite easy to set up and to include inside the pipeline. 
However, a couple of thoughts on its use:

* `mlflow.start_run()` is part of the pipeline in the example script. One idea could be to use the decorator pattern,
in which the context manager is set inside a decorator function. The decorator could then be applied to the 
training function only.
* as far as I understood, `mlflow` creates a folder called `mlruns` inside the folder where the library is imported. 
I haven't investigated on whether there could be a general folder, and where to define it (I assume there is a 
config file for the general Mlflow settings). Nevertheless, if the experiments are not saved inside such a folder, 
as in the case of the Airflow execution, nothing would show up when running `mlflow ui`. (See below for Airflow.)

