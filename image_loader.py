import os
import re
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from matplotlib import pyplot as plt


class CoinDataset(Dataset):
    def __init__(self, obverse_folder, reverse_folder, type_folder=None, transform=None, label="obverse", use_grayscale=False):
        """
        A PyTorch Dataset for obverse or reverse images with optional pairing.
        :param obverse_folder: Path to the obverse images folder.
        :param reverse_folder: Path to the reverse images folder.
        :param type_folder: Path to the folder where images are categorized by type.
        :param transform: Transformations to apply to the images.
        :param label: Specify 'obverse', 'reverse', or 'paired' mode.
        :param use_grayscale: If True, convert images to grayscale while keeping 3 channels.
        """
        self.obverse_folder = obverse_folder
        self.reverse_folder = reverse_folder
        self.type_folder = type_folder
        self.label = label
        self.use_grayscale = use_grayscale
        self.obverse_images = []
        self.reverse_images = []
        self.paired_images = []
        if transform is None:
            self.transform = self._default_transforms()
        else:
            self.transform = transform

        # Load images
        self._load_images()

    def _default_transforms(self):
        """
        Define default image transformations, including grayscale conversion if enabled.
        """
        transforms_list = []

        # Convert to grayscale if the option is enabled (but keep 3 channels for ResNet)
        if self.use_grayscale:
            transforms_list.append(transforms.Grayscale(num_output_channels=3))

        transforms_list.extend([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        return transforms.Compose(transforms_list)
    
    def _load_images(self):
        """
        Load obverse and reverse images from their respective folders.
        """
        self.obverse_images = self._load_folder(self.obverse_folder, "obverse")
        self.reverse_images = self._load_folder(self.reverse_folder, "reverse")
        if self.label == "paired":
            self._pair_images()

    def _load_folder(self, folder_path, label):
        """
        Helper function to load images from a folder.
        :param folder_path: Path to the folder.
        :param label: Label to identify the folder (e.g., "obverse" or "reverse").
        :return: List of dictionaries with image data.
        """
        images = []
        if not os.path.exists(folder_path):
            raise FileNotFoundError(f"The folder '{folder_path}' does not exist.")
        
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            if os.path.isfile(file_path) and file_path.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):
                image_type = self._get_image_type(filename) if self.type_folder else "Unknown"
                images.append({
                    "filename": filename,
                    "path": file_path,
                    "label": label,
                    "type": image_type,
                    "image": Image.open(file_path)
                })
        return images
    
    def _get_image_type(self, filename):
        """
        Determine the type of the image by searching for its filename in the type_folder.
        :param filename: The filename to look for in the type_folder.
        :return: The category/type of the image based on its folder name.
        """
        if not os.path.exists(self.type_folder):
            return "Unknown"

        for category in os.listdir(self.type_folder):
            category_path = os.path.join(self.type_folder, category)
            if os.path.isdir(category_path):
                if filename in os.listdir(category_path):
                    return category  # Return the folder name as the type

        return "Unknown"
    
    def _extract_base_name(self, filename):
        """
        Extract the base name from the filename to pair obverse and reverse images.
        """
        patterns = [
            r"(.*?)_a$", r"(.*?)_r$", r"(.*?)_av$", r"(.*?)_rv$",
            r"(.*?)_obv$", r"(.*?)_rev$", r"(.*?)a\d+$", r"(.*?)r\d+$"
        ]
        filename = filename.replace(".jpg", "")
        for pattern in patterns:
            match = re.match(pattern, filename, re.IGNORECASE)
            if match:
                cleaned_filename = re.sub(pattern, r"\1", filename, flags=re.IGNORECASE)
                return cleaned_filename
        return filename

    def _pair_images(self):
        """
        Pair obverse and reverse images based on their filenames.
        """
        obverse_dict = {self._extract_base_name(img['filename']): img for img in self.obverse_images}
        reverse_dict = {self._extract_base_name(img['filename']): img for img in self.reverse_images}

        self.paired_images = [
            {"obverse": obverse_dict[key], "reverse": reverse_dict[key]}
            for key in obverse_dict.keys() & reverse_dict.keys()
        ]

    def __len__(self):
        """
        Return the length of the dataset.
        """
        if self.label == "paired":
            return len(self.paired_images)
        elif self.label == "obverse":
            return len(self.obverse_images)
        else:
            return len(self.reverse_images)

    def __getitem__(self, idx):
        """
        Retrieve an item by index.
        :param idx: Index of the item to retrieve.
        :return: Preprocessed image tensor and filename, or paired tensors.
        """
        if self.label == "paired":
            pair = self.paired_images[idx]
            obverse_image = self.transform(pair['obverse']['image'])
            reverse_image = self.transform(pair['reverse']['image'])
            return obverse_image, reverse_image, pair['obverse']['filename'], pair['reverse']['filename']
        elif self.label == "obverse":
            img = self.obverse_images[idx]
        else:
            img = self.reverse_images[idx]

        image_tensor = self.transform(img['image'])
        return image_tensor, img['filename'], img["type"], img["label"]


    def show_image(self, idx, normalise=False, return_array=False):
        if self.label == "obverse":
            img = self.obverse_images[idx]["image"]
        else:
            img = self.reverse_images[idx]["image"]
        transforms_list = []

        if self.use_grayscale:
            transforms_list.append(transforms.Grayscale(num_output_channels=3))

        transforms_list.extend([
            transforms.Resize((224, 224)),
            transforms.ToTensor(), 
        ])
        
        if normalise:
            transforms_list.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))

        transforms_ = transforms.Compose(transforms_list)
        img = transforms_(img).permute(1, 2, 0).numpy()
        img = img.clip(0, 1)
        plt.imshow(img)
        plt.axis("off")
        plt.show()

        if return_array:
            return img
    

if __name__ == "__main__":
    obverse_folder = r"F:\data_ba\kleinsilber_linz\kleinsilber_linz\obverse"
    reverse_folder = r"F:\data_ba\kleinsilber_linz\kleinsilber_linz\reverse"
    type_folder = r"F:\data_ba\wetransfer_kleinsilber_nach_typen_2024-10-23_1100\Kleinsilber_nach_Typen"

    obverse_dataset = CoinDataset(
        obverse_folder=obverse_folder,  
        reverse_folder=reverse_folder,
        type_folder=type_folder,
        label="obverse",
        use_grayscale=True
    )
    print(obverse_dataset[0])
    # sample_img, sample_filename, sample_type = obverse_dataset[0]
    # print(f"Image: {sample_filename}, Type: {sample_type}")
    # obverse_dataset.show_image(0)
    unknown_count = sum(1 for _, _, img_type, in obverse_dataset if img_type == "Unknown")
    print(unknown_count)
