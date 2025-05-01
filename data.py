import json
import os
import imageio
import pandas as pd
import torch
from PIL import Image
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset
from tqdm.auto import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"


class VideoDataset(Dataset):
    def __init__(self, df, processor):
        self.df = df
        self.processor = processor

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        idx = int(idx)  # Convert the index to an integer
        row = self.df.iloc[idx]
        features_path = row["features_path"]

        if not os.path.exists(features_path):
            return None, None, None

        features = torch.load(features_path).to(device)

        inputs = self.processor(
            text=row["captions"], return_tensors="pt", padding=True, truncation=True
        )
        captions = inputs["input_ids"].squeeze(0).to(device)
        labels = torch.tensor(row["labels"], dtype=torch.long).to(device)

        return features, captions, labels


def custom_collate_fn(batch):
    batch = [item for item in batch if item[0] is not None]
    if len(batch) == 0:
        return None
    features, captions, labels = zip(*batch)

    # Pad features to the same size
    max_len = max(f.shape[0] for f in features)
    padded_features = [
        torch.nn.functional.pad(f, (0, 0, 0, max_len - f.shape[0])) for f in features
    ]

    features = torch.stack(padded_features).to(device)
    captions = torch.nn.utils.rnn.pad_sequence(captions, batch_first=True).to(device)
    labels = torch.stack(labels).to(device)
    return features, captions, labels


class DataProcessor:
    """
    Class to process the segmented and captions data
    """

    def __init__(
        self, segmented_manifest_path, captions_manifest_path, segmented_folder
    ):
        self.segmented_manifest = self.load_json(file_path=segmented_manifest_path)
        self.captions_manifest = self.load_json(file_path=captions_manifest_path)
        self.segmented_folder = segmented_folder

    def load_json(self, file_path):
        with open(file_path, "r") as f:
            data = json.load(f)
        return data

    def extract_identifier(self, url):
        return url.split("watch?v=")[-1]

    def add_captions_to_segment(self, segmented_data, captions_data):
        # Iterate through each entry in the segmented JSON file
        for segment_id, segment in segmented_data.items():
            if "url" not in segment:
                continue

            # Extract the identifier from the URL
            identifier = self.extract_identifier(segment["url"])

            # Retrieve the corresponding captions using the identifier
            if identifier not in captions_data:
                continue

            captions = captions_data[identifier]

            # Filter the captions to include only those that fall within the start and end times of the segment
            start_time = segment.get("start", 0)
            end_time = segment.get("end", float("inf"))
            matching_captions = [
                caption
                for caption in captions
                if caption["start"] < end_time and caption["end"] > start_time
            ]

            # Keep increasing the range until a caption is found
            range_increment = 60  # Increased range increment
            while not matching_captions and captions:
                start_time -= range_increment
                end_time += range_increment
                matching_captions = [
                    caption
                    for caption in captions
                    if caption["start"] < end_time and caption["end"] > start_time
                ]

            # Add the filtered captions to the entry in the segmented JSON file
            segment["captions"] = matching_captions

        return segmented_data

    def filter_segmented_data(self, data, segmented_folder="data/segmented"):
        # Filter the segmented_manifest to only include the files that are present in the segmented folder
        # 5846 (original) -> 5752 (filtered) 94 diff
        # why 94 diff? negative start and end times in the manifest
        files = [
            os.path.splitext(f)[0]
            for f in os.listdir(segmented_folder)
            if f.endswith(".mp4")
        ]
        data = {k: v for k, v in data.items() if k in files}

        # from the data filter out the segments that have no captions
        filtered_data = {k: v for k, v in data.items() if len(v["captions"]) > 0}
        return filtered_data

    def process_data(self):
        # Add captions to the segmented manifest
        segmented_manifest_with_captions = self.add_captions_to_segment(
            segmented_data=self.segmented_manifest,
            captions_data=self.captions_manifest,
        )

        # Filter the segmented manifest
        filtered_segmented_manifest = self.filter_segmented_data(
            data=segmented_manifest_with_captions,
            segmented_folder=self.segmented_folder,
        )

        # convert the dictionary to a pandas dataframe
        df = pd.DataFrame(filtered_segmented_manifest).T
        df["captions"] = df["captions"].apply(
            lambda x: " ".join([caption["caption"] for caption in x])
        )

        # for the labels, each entry is a list of variables, so pick the last one and convert it to a string
        df["labels"] = df["labels"].apply(lambda x: x[-1])
        label_encoder = LabelEncoder()
        df["labels"] = label_encoder.fit_transform(df["labels"])

        # add a column for features path
        df["features_path"] = df.index.map(
            lambda x: os.path.join("precomputed_features", x + ".pt")
        )
        df["labelsname"] = label_encoder.inverse_transform(df["labels"])
        return df


def precompute_features(df, segmented_folder_path, model, processor, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    for idx, row in tqdm(
        df.iterrows(), total=len(df), desc="Precomputing Features", leave=False
    ):
        file_path = os.path.join(segmented_folder_path, row.name + ".mp4")
        feature_path = os.path.join(output_folder, row.name + ".pt")

        # if the feature file already exists, skip
        if os.path.exists(feature_path):
            continue

        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            continue

        try:
            reader = imageio.get_reader(file_path, "ffmpeg")
            frames = []
            for frame in reader:
                frames.append(frame)
            reader.close()
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            continue

        if len(frames) == 0:
            print(f"No frames found in {file_path}")
            continue

        frames = [Image.fromarray(frame) for frame in frames]
        inputs = processor(images=frames, return_tensors="pt", padding=True)
        pixel_values = inputs["pixel_values"].to(device)

        with torch.no_grad():
            features = model.get_image_features(pixel_values)  # Precompute features

        torch.save(features.cpu(), feature_path)