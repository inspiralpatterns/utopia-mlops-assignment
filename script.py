import argparse

import mlflow

from src.utils.io_utils import load_config
from steps import dataset_dir_structure, process_audio, create_mnist_audio_dataset, build_mnist_audio_model, \
    split_dataset, train_and_evaluate


# Main flow
def main(cfgs: dict):
    dataset_path = dataset_dir_structure(cfgs.get("dataset_structure"))
    # Process audio files into mel-spectrograms
    process_audio(cfgs.get("processor"), dataset_path)
    with mlflow.start_run():
        # Build MNIST audio dataset
        tf_dataset = create_mnist_audio_dataset(dataset_path)
        # Build train and validation datasets
        validation_split = cfgs.get("model").get("training").get("validation_split")
        training_dataset, validation_dataset = split_dataset(
            tf_dataset.shuffle(buffer_size=50, seed=42),
            validation_split
        )
        # Create and compile model
        mnist_model = build_mnist_audio_model(cfgs.get("model"))
        # Train and evaluate
        _, evaluation = train_and_evaluate(
            model=mnist_model,
            train_dataset=training_dataset,
            validation_dataset=validation_dataset,
            train_cfg=cfgs.get("model").get("training")
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run the audio preprocessing pipeline for the input directory.")
    parser.add_argument("--config", type=str, required=True, help="YAML config file")

    args = parser.parse_args()
    config = load_config(args.config)

    mlflow.tensorflow.autolog()

    mlflow.create_experiment(config.get("experiment_name"))
    mlflow.set_experiment(config.get("experiment_name"))

    main(config)
