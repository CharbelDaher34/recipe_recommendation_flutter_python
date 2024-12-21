import pandas as pd
import requests

df = pd.read_csv("../api/data.csv")


def get_embeddings(strings_list, api_url="http://localhost:8000/encode"):
    """
    Get embeddings for a list of strings (text or base64-encoded images)

    Args:
        strings_list (list): List of strings (text or base64-encoded images)
        api_url (str): URL of the embedding API

    Returns:
        list: List of embeddings
    """
    embeddings_list = []

    for string in strings_list:
        try:
            response = requests.post(api_url, json={"content": string})
            response.raise_for_status()
            result = response.json()

            if result["status"] == "success":
                embeddings_list.append(
                    {
                        "input": (
                            string[:50] + "..." if len(string) > 50 else string
                        ),  # Truncate long strings in log
                        "type": result["type"],
                        "embeddings": result["embeddings"],
                    }
                )
            else:
                print(
                    f"Error processing string: {result.get('message', 'Unknown error')}"
                )
                embeddings_list.append(
                    {
                        "input": string[:50] + "...",
                        "type": "error",
                        "error": result.get("message", "Unknown error"),
                    }
                )

        except Exception as e:
            print(f"Error in API call: {str(e)}")
            embeddings_list.append(
                {"input": string[:50] + "...", "type": "error", "error": str(e)}
            )

    return embeddings_list


import tqdm
import torch
import json

device = "cpu"


def calculate_embeddings(file_name="./embeddings.json", sample_size=50000):
    """
    - Calculate embeddings for titles and images in the DataFrame and save them to a JSON file.
      :param file_name: Path to the JSON file where embeddings will be saved.
      :param sample_size: Number of samples for which embeddings will be calculated.
    - This method calculates embeddings for the titles and images in the dataset using the pre-trained model.
    - These embeddings are then stored in a JSON file, which will be used later for matching and recommending recipes.
    """
    sample_df = df  # .head(sample_size)  # Use the first 10k samples
    embeddings = {}
    for index, row in tqdm.tqdm(sample_df.iterrows(), total=len(sample_df)):
        title_text = row["Title"]
        # Convert the text to a PyTorch tensor and move it to the device
        title_inputs = torch.tensor(get_embeddings([title_text])[0]["embeddings"]).to(
            device
        )
        title_embedding = title_inputs.cpu().numpy().tolist()
        # Handle image embedding if the image is available
        if not pd.isna(row["Image"]):
            try:
                print("Processing image for ID:", row["ID"])
                base64_string = row["Image"]["bytes"]
                image_inputs = torch.tensor(
                    get_embeddings([base64_string])[0]["embeddings"]
                ).to(device)
                image_embedding = image_inputs.cpu().numpy().tolist()
                final_embedding = [
                    (x + y) / 2 for x, y in zip(title_embedding, image_embedding)
                ]
            except Exception as e:
                print(f"Error processing image for ID {row['ID']}: {e}")
                final_embedding = title_embedding
        else:
            final_embedding = title_embedding
        # Save with ID as the key
        embeddings[row["ID"]] = final_embedding
    # Save the embeddings to a JSON file
    with open(file_name, "w") as f:
        json.dump(embeddings, f)
    print(f"Embeddings saved successfully to {file_name}.")


calculate_embeddings()
