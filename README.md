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

To load the model, you can run the following snippet inside the REPL:
```python
import tensorflow as tf

model_folder = "model"
model = tf.keras.models.load_model(model_folder)
```

## Server run
You will find the same code under `/home/ubuntu/utopia-mlops-assignment`. There, you will have:

* `mlruns` folder -where experiment logs are stored. Each run has its own `uuid` and subfolder with e.g. params, 
metrics etc. As an example, to inspect accuracy for experiment 1
```shell
ubuntu@ml-mattia-patern-wattmsam:~/utopia-mlops-assignment$ cat mlruns/1/622979359eed417fb83ff3be279c4ed5/metrics/accuracy
1646494486641 0.20000000298023224 0
1646494486881 0.3499999940395355 1
1646494487026 0.5 2
1646494487121 0.699999988079071 3
1646494487263 0.7749999761581421 4
1646494487408 0.699999988079071 5
1646494487550 0.8500000238418579 6
1646494487695 0.8500000238418579 7
1646494487836 0.8999999761581421 8
1646494487982 0.9750000238418579 9
```

* `model` folder - where trained models are saved (see above for how to load the model);
* two logs file with format `experiment-name.log` where Python stdout is piped in. This gives information such as
model summary, training and evaluation stats per epoch and IO specs.

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

Finally, I have not been able to rid the code of all hard-coded stuff and ad-hoc implementation e.g. in random and 
non-exhaustive order:

* when [shuffling the dataset](https://github.com/inspiralpatterns/utopia-mlops-assignment/blob/e1be99eed4c568a4f857daab414a8ea0b8d3120c/script.py#L22) the code 
still depends on the hard-coded size of the full dataset;
* the [input shape](https://github.com/inspiralpatterns/utopia-mlops-assignment/blob/e1be99eed4c568a4f857daab414a8ea0b8d3120c/src/pipeline/steps.py#L73) for the model 
is coupled to the shape of each processed audio file, that the model will fail is e.g. the audio file duration changes.
* most of the IO functions rely on the use of `Path`, and generators of them;
* there should be a more general definition of _dataset_;
* there could be a more general `Model` interface, from which `CNNModel` could inherit;
* tests are missing for all classes so that input and output are validated;
* error handling is not in place, that is, there are no custom exception thrown by the classes in case of failure 
and no process to cope with such failure (such as `Either` types);
* docstrings are missing for all classes.

Ideally, we would like to impute such dynamic values as input shape, dataset size etc. during the preprocessing task 
and store them as metadata so that they could be used in downstream steps. 

With respect to the ad-hoc implementations, althoughI believe the use of `Path` and friends for some sort of _lazy 
loading_ is a good approach, there should be an alternative
e.g. such functions could take either a path, or a string or a list of them, and the function could handle each case accordingly.

### Observation on MLOps tools

#### Mlflow

`mlflow` is being use for experiment tracking, as it is quite easy to set up and to include inside the pipeline. 
However, a couple of thoughts on its use:

* `mlflow.start_run()` is part of the pipeline in the example script. One idea could be to use the decorator pattern,
in which the context manager is set inside a decorator function. The decorator could then be applied to the 
training function only.
* as far as I understood, `mlflow` creates a folder called `mlruns` inside the folder where the library is imported. 
I haven't investigated on whether there could be a general folder, and where to define it (I assume there is a 
config file for the general Mlflow settings). Nevertheless, if the experiments are not saved inside such a folder, 
as in the case of the Airflow execution, nothing would show up when running `mlflow ui`. (See below for Airflow.)

#### Airflow

As said above, the idea underlying the functions in `steps.py` is to have a function _per task_. That is,
they can all be combined inside a DAG: thus, the choice of Airflow. Moreover, I would expect that not all
pipelines require e.g. to set up the folder structure for the dataset. With that said, my initial idea was to:

* build an Airflow `DAG` for the pipeline;
* have optional tasks as described above;
* pass on the output of one task as input for the next one, as in the case of setting `InputShape` for the model;
* have the `mlflow` experiment run from within airflow.

However, there were some troubles that did not let me successfully get to the desired outcome. Furthermore, I had 
some doubts in how to organise the code and what strategy to pursue. I will give a few examples for both below.

> The examples refer to experiments with my local Airflow setup for convenience - I could access the UI.

Initially, I thought of invoking the Python script as a test that everything was working properly in the Airflow 
scheduler. 
To do so, I used the Airflow `BashOperator` which allows to execute a Shell command. I used `BashOperator` because I wanted to keep the scripts outside of the local `dags` 
folder, and using `PythonOperator` to run the python callable would result in some ad-hoc `sys.append()`, which I 
didn't want the DAG script to pollute with. However, this failed at first because of 
required Python dependencies. There can be several possible solutions, which I haven't fully explored:

* using `PythonVirtualenvOperator` to set up a `venv` for each task given a list of requirements;
* using `DockerOperator` to self-contain the task in its Docker image;
* packaging the pipeline and use it as dependency;
* move the code inside the `dags` folder so to avoid appending paths for the interpreter;
* installing the required dependencies in the target system.

Eventually, I went for the last option as it is the easiest and the dependencies are few. However, I do **not** 
believe this could be any good for a production setup and I think that using Docker images together with some image 
repository could be favourable in order to keep the system tidy.

Moving on, I realised that it is not that easy to have data transit between tasks. Airflow documentation suggests to 
use XCom for short messages and remote storage for bigger data, such as S3. That means I would have to change the 
way the functions in `script.py`, i.e. the tasks, fetch or read the parameters they need.

Lastly, I did try to run the dag from within Airflow. Although the execution was successful, I could not see any 
result when checking for the experiment run. As I found out from logging some of the `Path().resolve()` code, 
Airflow has its own internal setup where temporary files and/or folder can be created and/or accessed. As an example,
the absolute path of one of the logging was `/private/var/...`, which did not reflect the idea that I was having, 
namely the absolute path would be the one where the DAG script lives. This had some unexpected consequences:

* I could no longer use dynamic path resolution for the YAML config file, nor the recordings folder as the 
* resolution would be wrong when running the code from within Airflow;
* All paths should be static, i.e. dataset path, model folder path etc. That is, the code should be changed to 
* reflect this and to properly fetch and load data in its right place;
* I could not find the automatically generated `mlruns` folder and therefore access the experiment run logs.

### Conclusion

The MLOps setup I had in mind consisted of:

* **Mlflow** for experiment tracking and logging;
* **Airflow** for pipeline orchestration and scheduling.

The scheduler would run the pipeline as a number of tasks defined inside the codebase. Since the codebase makes use 
of `mlflow` as its dependency, the pipeline is assigned an experiment name and a run ID. The experiment can then be 
accessed from MLflow UI.

However, because of Airflow's own way of running scheduled jobs, the logs about the experiment and the Mlflow 
autogenerated `mlruns` folder that contains them cannot be found at the expected location. Therefore, it is not 
possible to inspect experiment logs, nor the model artefact with the current setup.

Nevertheless, the pipeline can be run manually using `script.py` and providing a YAML config file. Doing so, the 
experiment tracking logs can be found in project folder.