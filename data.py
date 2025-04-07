import os

from datasets import load_dataset


def download_and_save(dataset_name, save_path):
    """
    Downloads the dataset from Hugging Face and saves it locally.

    Args:
        dataset_name (str): The identifier of the dataset on Hugging Face.
        save_path (str): The directory where the dataset will be saved.
    """
    # Load the dataset from Hugging Face
    print(f"Downloading {dataset_name} dataset...")
    dataset = load_dataset(dataset_name)

    # Create the save directory if it does not exist
    os.makedirs(save_path, exist_ok=True)

    # Save the dataset to disk
    dataset.save_to_disk(save_path)
    print(f"{dataset_name} dataset saved successfully at {save_path}.\n")


def main():
    data_folder = "./data/"
    datasets_info = {
        "openai/openai_humaneval": f"{data_folder}humaneval_dataset",
        "microsoft/orca-math-word-problems-200k": f"{data_folder}orca_math_dataset",
    }

    for dataset_name, save_path in datasets_info.items():
        try:
            download_and_save(dataset_name, save_path)
        except Exception as e:
            print(f"An error occurred while downloading {dataset_name}: {e}")


if __name__ == "__main__":
    main()
