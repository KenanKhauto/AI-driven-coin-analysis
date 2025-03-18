import torch
import clip
from torch.utils.data import Dataset, DataLoader

class CLIPCoinEmbedder:
    def __init__(self, dataset, model_name="ViT-B/32", batch_size=16, device=None):
        """
        CLIP-based embedding extractor for coin datasets.
        :param dataset: A CoinDataset object
        :param model_name: CLIP model variant (default "ViT-B/32").
        :param batch_size: Batch size for efficient processing.
        :param device: Device to use ('cuda' or 'cpu').
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.model, self.preprocess = clip.load(model_name, device=self.device)

    def generate_text_description(self, side, coin_type):
        """
        Generates a textual description for CLIP embedding.
        :param side: "obverse" or "reverse".
        :param coin_type: The type of the coin (or "Unknown").
        :return: Formatted text description.
        """
        if coin_type.lower() == "unknown":
            return f"The {side} of this ancient Celtic coin has an unidentified design."
        else:
            return f"This is the {side} of an ancient Celtic coin depicting {coin_type}."

    def extract_embeddings(self):
        """
        Extracts both text and image embeddings from the dataset.
        :return: Lists of image embeddings, text embeddings, filenames.
        """
        dataloader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=False)
        image_embeddings, text_embeddings, filenames, base_text = [], [], [], []

        with torch.no_grad():
            for batch in dataloader:
                images, batch_filenames, batch_types, batch_labels = batch
                
                # Convert images to CLIP format
                images = images.to(self.device)

                # Generate text descriptions dynamically
                text_descriptions = [self.generate_text_description(label, ctype) for ctype, label in zip(batch_types, batch_labels)]
                
                # Compute text and image embeddings
                text_tokens = clip.tokenize(text_descriptions).to(self.device)
                text_emb = self.model.encode_text(text_tokens)
                image_emb = self.model.encode_image(images)

                # Store embeddings and filenames
                image_embeddings.extend(image_emb.cpu().numpy())
                text_embeddings.extend(text_emb.cpu().numpy())
                filenames.extend(batch_filenames)
                base_text.extend(text_descriptions)

        return image_embeddings, text_embeddings, filenames, base_text

# Example Usage
if __name__ == "__main__":
    from image_loader import CoinDataset 

    
    obverse_folder = r"F:\data_ba\kleinsilber_linz\kleinsilber_linz\obverse"
    reverse_folder = r"F:\data_ba\kleinsilber_linz\kleinsilber_linz\reverse"
    type_folder = r"F:\data_ba\kleinsilber_linz\types"

    dataset = CoinDataset(
        obverse_folder=obverse_folder,
        reverse_folder=reverse_folder,
        type_folder=type_folder,
        label="obverse",
        use_grayscale=True
    )

    
    embedder = CLIPCoinEmbedder(dataset)

    
    img_embs, txt_embs, filenames = embedder.extract_embeddings()

    print(f"Extracted {len(img_embs)} image embeddings and {len(txt_embs)} text embeddings.")
