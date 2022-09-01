# Create hugging face dataset
from datasets import load_dataset, load_from_disk
from transformers import AutoTokenizer, AutoModel
from joblib import load
import torch
import numpy as np
import os
import re
import logging

# Configure logger
formatter = logging.Formatter("%(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
ch.setFormatter(formatter)
logger.addHandler(ch)

#
# Constants
#
N_JOBS = 20
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_NAME = "allenai/longformer-base-4096"
LR_PATH = "models/longformer-base-4096-mediations-logistic-classifier.joblib"

#
# Functions
#
def load_training_data(file, base_path="data/dispute-dataframes/"):
    # TODO: convert to s3
    data_files = {"train": f"{base_path}{file}"}
    return load_dataset("json", data_files=data_files)


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


def huggingface_dataset_to_array(dataset):
    return np.array(dataset["train"]["hidden_state"])


def generate_preds(model, X):
    y_preds = model.predict(X)
    y_proba = model.predict_proba(X)
    return y_preds, y_proba


def prediction_pipeline(dataset, tokenizer, model, lr, batch_size=8):
    """
    Pipeline of steps to generate prediction labels for dispute data

    Params
    ------
    dataset: Huggingface Dataset, dispute text
    tokenizer: tokenizer, Huggingface tokenizer associated with model
    model: model, Huggingfacae transformer to use for generating encodings
    lr: pretrained logistic regression model, used as classifier head
    batch_size: int, number of documents to process in each batch

    Returns
    -------
    Dataframe with predictions for whether a dispute is a mediation case
    """

    # Create feature embeddings
    encoded = dataset.map(
        tokenize, batched=True, batch_size=None, fn_kwargs={"tokenizer": tokenizer}
    )
    encoded.set_format("torch", columns=["input_ids", "attention_mask"])
    hidden = encoded.map(
        extract_hidden_states,
        batched=True,
        batch_size=batch_size,
        fn_kwargs={"model": model, "tokenizer": tokenizer, "device": DEVICE},
    )

    # Generate predictions
    X = huggingface_dataset_to_array(dataset=hidden)
    return generate_preds(model=lr, X=X)


def main():
    logger.info(f"Retrieving dataset batch for prediction")
    completed = [f for f in os.listdir("data/dispute-predictions/")]
    input_files = [
        f for f in os.listdir("data/dispute-dataframes/") if f not in completed
    ]

    logger.info(f"Loading pre-trained prediction head {LR_PATH}")
    lr = load(LR_PATH)

    logger.info(f"Instantiating HF Transformer {MODEL_NAME} using Device {DEVICE}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME)
    model.to(DEVICE)

    failed_files = []
    for f in input_files:
        logger.info(f"Loading dataset: {f}")
        dataset = load_training_data(file=f)

        logger.info("Passing data to prediction pipeline")
        try:
            y_pred, y_proba = prediction_pipeline(
                dataset=dataset, tokenizer=tokenizer, model=model, lr=lr
            )
        except Exception as err:
            logger.error(
                f"Prediction pipeline failed for dataset: {f}\nwith exception {err}"
            )
            failed_files.append(f)
            continue

        logger.info("Updating dataset with predictions")
        dataset["train"] = dataset["train"].add_column(name="label", column=y_pred)
        dataset["train"] = dataset["train"].add_column(
            name="probability", column=y_proba[:, 1]
        )

        # Save
        outfile = f.replace(".json", "")
        outpath = os.path.join("data/prediction-cache", outfile)
        logger.info(f"Saving Huggingface Dataset to: {outpath}")
        dataset.save_to_disk(outpath)

        outpath = os.path.join("data/dispute-predictions", f)
        logger.info(f"Saving predicions dataframe to: {outpath}")
        dataset["train"].to_json(path_or_buf=outpath, orient="records", num_proc=N_JOBS)

    logger.info("The predictions failed for the following files:")
    for f in failed_files:
        logger.info(f)


if __name__ == "__main__":
    main()
