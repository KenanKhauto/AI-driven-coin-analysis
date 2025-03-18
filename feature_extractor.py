import torch
import numpy as np
from torchvision import models
from torch.utils.data import DataLoader


class FeatureExtractor:
    def __init__(self, dataset, label, batch_size=16, device=None):
        """
        Feature extractor for obverse, reverse, or paired datasets.
        :param dataset: A CoinDataset object.
        :param label: Dataset type - 'obverse', 'reverse', or 'paired'.
        :param batch_size: Batch size for feature extraction.
        :param device: Device to use for extraction ('cuda' or 'cpu').
        """
        self.dataset = dataset
        self.label = label
        self.batch_size = batch_size
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize a pretrained ResNet50 model for feature extraction
        model = models.resnet50(pretrained=True)
        self.feature_extractor = torch.nn.Sequential(*list(model.children())[:-1])  # Remove FC layer
        self.feature_extractor = self.feature_extractor.to(self.device)
        self.feature_extractor.eval()

    def extract_features(self, output_features_file, output_filenames_file):
        """
        Extract features for the given dataset and save to files.
        :param output_features_file: File path to save features (e.g., 'features.npy').
        :param output_filenames_file: File path to save filenames (e.g., 'filenames.txt').
        """
        dataloader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=False)
        features = []
        filenames = []
        counter = 0
        with torch.no_grad():
            for data in dataloader:
                counter += 1
                if self.label == "paired":
                    # Extract features for paired obverse and reverse
                    obverse_images, reverse_images, obverse_filenames, reverse_filenames = data
                    obverse_images = obverse_images.to(self.device)
                    reverse_images = reverse_images.to(self.device)

                    # Extract features
                    obverse_features = self.feature_extractor(obverse_images).squeeze(-1).squeeze(-1)
                    reverse_features = self.feature_extractor(reverse_images).squeeze(-1).squeeze(-1)

                    # Store results
                    features.extend(obverse_features.tolist() + reverse_features.tolist())
                    filenames.extend(obverse_filenames + reverse_filenames)

                else:
                    # Extract features for obverse or reverse
                    images, batch_filenames = data
                    images = images.to(self.device)

                    # Extract features
                    batch_features = self.feature_extractor(images).squeeze(-1).squeeze(-1)

                    # Store results
                    features.extend(batch_features.tolist())
                    filenames.extend(batch_filenames)
                print(f"Batch {counter} extracted!")
        # Save features and filenames
        np.save(output_features_file, features)
        with open(output_filenames_file, "w") as f:
            for filename in filenames:
                f.write(f"{filename}\n")

        print(f"Features saved to {output_features_file}")
        print(f"Filenames saved to {output_filenames_file}")

    def load_features(self, features_file, filenames_file):
        """
        Load features and filenames from saved files.
        :param features_file: Path to the features file (e.g., 'features.npy').
        :param filenames_file: Path to the filenames file (e.g., 'filenames.txt').
        :return: Tuple (features, filenames).
        """
        features = np.load(features_file)
        with open(filenames_file, "r") as f:
            filenames = [line.strip() for line in f]
        return features, filenames
