# Create hugging face dataset
from datasets import load_dataset, load_from_disk
from transformers import AutoTokenizer, AutoModel
from torch.quantization import quantize_dynamic
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    confusion_matrix,
    f1_score,
    brier_score_loss,
)
from sklearn.dummy import DummyClassifier
from joblib import dump
from imblearn.combine import SMOTEENN
from itertools import product
from collections import Counter
from tqdm import tqdm
import torch
import torch.nn as nn
import numpy as np
import os
import matplotlib.pyplot as plt
import configparser

#
# Constants
#
n_jobs = 20
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
models = ["distilbert-base-uncased", "allenai/longformer-base-4096"]
quantize = [False]  # Add false to retrain unquantized models
force = True  # Whether to force a retrain

scores = {}

#
# Initiate box client
#
parser = configparser.ConfigParser()
parser.read("pipeline.conf")

auth = OAuth2(
    client_id=parser.get("umd_box_credentials", "CLIENT_ID"),
    client_secret=parser.get("umd_box_credentials", "CLIENT_SECRET"),
    access_token=parser.get("umd_box_credentials", "ACCESS_TOKEN"),
)
client = Client(auth)

# Retrieve mediation project folder info
folder_id = parser.get("umd_box_credentials", "OTHER_DOCUMENTS_FOLDER_ID")
parent_folder = client.folder(folder_id=folder_id).get()

#
# Functions
#
def load_training_data(base_path="data/mediation_search_results"):
    dataset_paths = glob.glob(base_path + '*csv')
    # Download datasets if they aren't already downloaded
    if len(dataset_paths) < 2:
        data_id = [item.id for item in curation_pipeline.item_collection['entries'] if item.name == 'Data'][0]
        data_folder = client.folder(data_id).get()
        
        for item in data_folder.get().item_collection['entries']:
            if 'mediation_search_results' in item.name:
                with open(os.path.join('../data', item.name), 'wb') as output_file:
                    client.file(item.id).download_to(output_file)

    # Load datasets as Hugging Face Datasets
    data_files = {"train": f"{base_path}-train.csv", "test": f"{base_path}-test.csv"}
    return load_dataset("csv", data_files=data_files)


def tokenize(batch, tokenizer):
    """Tokenizer utility."""
    return tokenizer(batch["text"], padding=True, truncation=True)


def extract_hidden_states(batch, tokenizer, model, device):
    """Feature Embedding Extraction Utility."""
    # Place model inputs on the GPU
    inputs = {
        k: v.to(device) for k, v in batch.items() if k in tokenizer.model_input_names
    }
    # Extract last hideen states
    with torch.no_grad():
        last_hidden_state = model(**inputs).last_hidden_state
    # Return vector for [CLS] token
    return {"hidden_state": last_hidden_state[:, 0].cpu().numpy()}


def create_train_test_splits(dataset):
    # Create splits
    X_train = np.array(dataset["train"]["hidden_state"])
    X_valid = np.array(dataset["test"]["hidden_state"])
    y_train = np.array(dataset["train"]["label"])
    y_valid = np.array(dataset["test"]["label"])
    return X_train, y_train, X_valid, y_valid


def smoteen_resample(X, y, seed=123, n_jobs=20):
    print(f"Starting SMOTEEN resample with {n_jobs} cpu cores.")
    smote_enn = SMOTEENN(random_state=seed, n_jobs=n_jobs)
    X_re, y_re = smote_enn.fit_resample(X, y)

    print("Resample complete. New class sizes:")
    print(sorted(Counter(y_re).items()))

    return X_re, y_re


def score_model(model, X, y):
    y_preds = model.predict(X)
    y_proba = model.predict_proba(X)
    model_scores = {}
    model_scores["score"] = model.score(X, y)
    model_scores["brier"] = brier_score_loss(y, y_proba[:, 1])
    model_scores["f1"] = f1_score(y, y_preds)
    return model_scores, y_predsprint(
        f'File "{uploaded_file.name}" uploaded to Box with file ID {uploaded_file.id}'
    )


def plot_confusion_matrix(y_preds, y_true, labels, path, subtitle=None):
    cm = confusion_matrix(y_true, y_preds, normalize="true")
    fig, ax = plt.subplots(figsize=(6, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap="Blues", values_format=".2f", ax=ax, colorbar=False)
    plt.title(f"Normalized confusion matrix\n{subtitle}")
    plt.savefig(path)


def get_or_create_curation_folder(parent_folder):
    already_created = "curation-pipeline" in [
        folder.name for folder in parent_folder.item_collection["entries"]
    ]
    if not already_created:
        curation_pipeline = parent_folder.create_subfolder("curation-pipeline")
        print(f"Created project subfolder with ID {curation_pipeline.id}")
    else:
        folder_id = [
            item.id
            for item in parent_folder.item_collection["entries"]
            if item.name == "curation-pipeline"
        ][0]
        curation_pipeline = client.folder(folder_id).get()
    return curation_pipeline


def get_or_create_models_folder(parent_folder):
    already_created = "models" in [
        folder.name for folder in parent_folder.item_collection["entries"]
    ]
    if not already_created:
        models_folder = parent_folder.create_subfolder("models")
        print(f"Created models subfolder with ID {models_folder.id}")
    else:
        folder_id = [
            item.id
            for item in parent_folder.item_collection["entries"]
            if item.name == "models"
        ][0]
        models_folder = client.folder(folder_id).get()
    return models_folder


def main():
    # Setup output directories on box
    curation_pipeline_folder = get_or_create_curation_folder(
        parent_folder=parent_folder
    )
    models_folder = get_or_create_models_folder(parent_folder=curation_pipeline_folder)

    # Load training data
    mediations = load_training_data()

    # Instantiate models
    for model_name, is_quantized in tqdm(product(models, quantize)):
        # Create paths
        model_name_condensed = (
            f"{model_name.split('/').pop()}{'-quantized' if is_quantized else ''}"
        )
        dataset_path = f"models/{model_name}-feature-embeddings{'-quantized' if is_quantized else ''}"
        model_path = (
            f"models/{model_name_condensed}-mediations-logistic-classifier.joblib"
        )
        fig_path = f"figs/{model_name_condensed}-confusion-matrix.png"

        # Skip training model if it's already trained
        # and not forcing a retrain
        if os.path.exists(model_path) and not force:
            continue

        print(f"Training {model_name_condensed}...")

        # Create tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        if is_quantized:
            model = model.to("cpu")
            model = quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)
        else:
            model = model.to(device)

        # Create or load feature embeddinggs
        if os.path.exists(dataset_path):
            mediations_hidden = load_from_disk(dataset_path)
        else:
            # Tokenize data according to model and generate embeddings
            mediations_encoded = mediations.map(
                tokenize,
                batched=True,
                batch_size=None,
                fn_kwargs={"tokenizer": tokenizer},
            )
            mediations_encoded.set_format(
                "torch", columns=["input_ids", "attention_mask", "label"]
            )
            batch_size = 16 if model_name == "distilbert-base-uncased" else 8
            mediations_hidden = mediations_encoded.map(
                extract_hidden_states,
                batched=True,
                batch_size=batch_size,
                fn_kwargs={
                    "model": model,
                    "tokenizer": tokenizer,
                    "device": "cpu" if is_quantized else device,
                },
            )

            # Save to disk
            mediations_hidden.save_to_disk(dataset_path)

        # Prep samples
        X_train, y_train, X_valid, y_valid = create_train_test_splits(mediations_hidden)
        X_train_re, y_train_re = smoteen_resample(X=X_train, y=y_train)

        # Training logistic regression
        lr_clf = LogisticRegression(
            max_iter=3000, n_jobs=n_jobs
        )  # Increase max iter to ensure convergence
        lr_clf.fit(X_train_re, y_train_re)

        # Score model
        model_scores, y_preds = score_model(lr_clf, X_valid, y_valid)
        scores[model_name_condensed] = model_scores
        print(model_scores)

        # Plot confusion matrix
        labs = ["Non-mediation", "Mediation"]
        plot_confusion_matrix(
            y_preds, y_valid, labs, subtitle=model_name_condensed, path=fig_path
        )

        # Save model locally
        with open(model_path, "wb") as f:
            dump(lr_clf, f)

        # Save to box
        uploaded_file = client.folder(models_folder.id).upload(model_path)
        print(
            f'File "{uploaded_file.name}" uploaded to Box with file ID {uploaded_file.id}'
        )

    # Output scores
    print("Scores:")
    print(scores)


if __name__ == "__main__":
    main()
