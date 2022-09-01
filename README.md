# PCWI Mediation Curation Pipeline

This repo contains the code to replicate the transformer-based prediction pipeline used for identifying government-separatist group mediation events in newspaper articles.

## Hardware Requirements

The pretrained model checkpoints are available for download from the Box account if you only want to replicate the results or reuse them. If, however, you need to fine-tune the models, then it is highly recommended you have access to a [Cuda-enabled GPU](https://developer.nvidia.com/cuda-gpus) preferably with more than 11 GB pf RAM (the more the better, but anything less than this will seriously constrain batch size and slow down training). In addition to a cuda-enabled GPU, cuda toolkit is required to performing training on the GPU. An installation guide can be found [here](https://docs.nvidia.com/cuda/index.html). The venv as specified in `requirements.txt` is written to work with cuda toolkit verison 10.2. More recent versions may not work correctly with the virtual environment. Training can be performed on a CPU in theory, but in practice this will be prohibitively slow on any dataset of meaningful size.

## Software Requirements

### External services

The code relies on two external services:

- UMD's Box cloud storage service
- Nexis Uni's [Data-as-a-Service API](https://www.lexisnexis.com/en-us/professional/data-as-a-service/CORE/API-data-delivery.page)

Interactive access to the UMD Box account requires generating a developer token for the umd-pcwi-mediation project. This can be done from the [Box Developer Console](https://umd.app.box.com/developers/console) by navigating to the `umd-pcwi-mediation` project > `Configuration` > clicking `Generate Developer Token`. This will generate an access token and client secret which can be stored in a `pipeline.conf` file in the root directory (see `example-pipeline.conf`) for an example. The developer token is only valid for 60 minutes after creation. Alternatively, for a more stable connection method using client credentials grant, store your
user ID (found under `General Settings`) in the `pipeline.conf` and use the Box SDK's [`CGAuth` Class](2PF6uPBklvuyx1ucg8VDemJRO4RoeWUj).

Documentation for Box's Python SDK can be found [here](https://github.com/box/box-python-sdk).

Access to the NU API requires an [API key](https://www.lexisnexis.com/en-us/professional/data-as-a-service/daas.page). In particular, the code requires access to their Batch News collection. The key used in the original nightly data pull is now out-of-service. Replicating the original data pull would require renewing the API key or purchasing access to a new one. Moreover, the original data pull represents a snapshot in time and, as such, is not deterministic. Future data pulls from the same data collection would not produce an exact replica.

### Python Modules

Before proceeding, users should create a python virtual environment (venv) for all module dependencies. The exact procedure for doing so varies by operating system. Instructions can be found [here](https://docs.python.org/3/library/venv.html). On Unix OSes, the following command works:

```console
$ python3 -m venv venv
```

After creating a virtual environment, run the following (on UNIX systems) in the base directory to install all required dependencies:

```console
$ source venv/bin/activate
(venv) $ pip install -r requirements.txt
```

## Project Structure

Below is the project directory tree. The `data`, `logs`, and `models` directories are largely empty but will be populated as you run the scripts.

The `notebooks` directory contains jupyter notebooks used for experimenting, debugging, developing and benchmarking the pipeline and classification models. These need not be used unless you want to gain more insight into the development process. Otherwise, all the code for reproducing the pipeline can be found in `scripts`.

```bash
├── data
├── figs
├── logs
├── models
├── notebooks
│   ├── benchmark_pipelines.ipynb
│   ├── create_human_validation_output.ipynb
│   ├── create_training_dataset.ipynb
│   ├── data_transform.ipynb
│   ├── debug.ipynb
│   ├── examine-failed-pipelines.ipynb
│   ├── experiment-with-transform.ipynb
│   ├── refetch_completes.ipynb
│   ├── reorder_searches.ipynb
│   └── train_classifiers.ipynb
├── scripts
│   ├── nu-api-batch-fetch.sh
│   ├── nu-api-data-fetch-aws-glue.py
│   ├── nu-api-data-predict-transform.py
│   ├── nu-api-data-predict.py
│   ├── nu-api-data-transform.py
│   ├── train_classifiers.py
│   └── update_mediation_seed_dataset.py
├── README.md
├── example-pipeline.conf
└── requirements.txt
```

## Retraining Models

If you intend to retrain the models from scratch, modify the training code, or fine-tune them on additional data, please read this section carefully. Otherwise, you may skip to the [Data Pipeline](##Data-Pipeline) section.

Once you have installed and activated the virtual environment, all you need to do is run the `train_classifiers.py` script. For example, to run the script as a background process on linux and log the output to a log file:

```bash
$ nohup python3 -u scripts/train_classifiers.py > logs/trains_classifiers.log &
```

## Data Pipeline

All the scripts used in the curation pipeline can be found in `scripts`. Currently, they are manually stitched together, though they certainly could be better productionized by using a pipeline orchestration tool like Apache Airflow.

For now, the sequencing is as follows:

1. Use the `nu-api-batch-fetch.sh` shell scripts to set up a nightly cron job.
2. Run `nu-api-data-transform.py` (intermittently or in one batch once the data is collected) to transform it into a format compatible with the prediction models
3. Run `nu-api-data-predict.py` to generate predictions
4. Run `nu-api-data-predict-transform.py` to convert the predictions into the final outputs for coders

### Setting up the cron job

The NU api endpoint is rate limited to 1000 requests in a single 24 hour period. Therefore, large data collections can take a long time to gather. The easiest way to automate this process is by setting up a cron job (a task your computer will run on a regular, scheduled basis). On linux, setting up cron jobs is easy. A nice guide can be found [here](https://phoenixnap.com/kb/set-up-cron-job-linux).

Cron jobs can be very finicky and not work well unless you use absolute (as opposed to relative paths) for specifying the location of files and directories. To ensure the job runs properly, first edit the `nu-api-batch-fetch.sh` file with the proper path to the current project directory (see the comments in the file for more details). Second, make sure when you add the job to the crontab file, you use absolute paths to the script and output (log file) location. Below is an example entry in the crontab file. You would replace [full-path-to-project-directory] with the full path to wherever you cloned this repo.

```bash
30 4    * * *   evan    /bin/bash [full-path-to-project-directory]/scripts/nu-api-batch-fetch.sh > [full-path-to-project-directory]/logs/fetch-dispute-data.log 2>&1
```

The above job will run the collection script every day at 4:30 AM. The job will finish when it hits the rate limit for a 24h period. Information about the job is logged to `logs/fetch-dispute-daata.log`. Data files are written to the `data/nu-api-data-raw` directory.

Note that the `nu-api-batch-fetch.sh` shell script just acts as a wrapper around the `nu-api-data-fetch-aws-glue.py` script so that the latter can be run as a cron job. All of the actual logic for fetching the data is in the latter file.

### Transforming raw data

### Predicting mediation events

### Transforming predictions for coders

## Model Development
