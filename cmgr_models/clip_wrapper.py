"""CLIP ViT-B/32 wrapper for CMGR.

Wraps OpenAI's CLIP model as a frozen backbone for:
- Encoding text prompts ("a photo of a [class]")
- Encoding enhanced depth-rendered images
- Computing image-text similarity scores

Key properties:
- All parameters are FROZEN during the entire CMGR training process
- Provides encode_image(), encode_text(), and combined scoring
- Uses ViT-B/32 architecture (512-dim features)

Usage:
    clip_model = CLIPWrapper(device='cuda')
    text_features = clip_model.encode_text(["a photo of a chair", ...])
    image_features = clip_model.encode_image(images)
    similarity = clip_model.compute_similarity(image_features, text_features)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CLIPWrapper(nn.Module):
    """Frozen CLIP ViT-B/32 wrapper.

    Wraps OpenAI's CLIP model for text and image encoding.
    All parameters are frozen.

    Args:
        model_name: CLIP model variant (default: 'ViT-B/32').
        device: Device to load the model on.
    """

    def __init__(self, model_name='ViT-B/32', device='cuda'):
        super().__init__()
        self.model_name = model_name
        self.device = device
        self.feat_dim = 512  # ViT-B/32 output dimension

        # Load CLIP model
        try:
            import clip
            self.model, self.preprocess = clip.load(model_name, device=device)
            print(f"[CLIPWrapper] Loaded CLIP {model_name}")
        except ImportError:
            print("[CLIPWrapper] Warning: clip package not installed.")
            print("Install with: pip install git+https://github.com/openai/CLIP")
            self.model = None
            self.preprocess = None

        # Freeze all parameters
        self._freeze()

        # Text template
        self.text_template = "a photo of a {}"

    def _freeze(self):
        """Freeze all CLIP parameters."""
        for param in self.parameters():
            param.requires_grad = False
        if self.model is not None:
            for param in self.model.parameters():
                param.requires_grad = False
        self.eval()

    def encode_text(self, texts):
        """Encode text descriptions into feature vectors.

        Args:
            texts: List of text strings, e.g., ["a photo of a chair", ...].
                   Or list of class names to be wrapped with the template.

        Returns:
            text_features: [N, 512] normalized text features.
        """
        if self.model is None:
            raise RuntimeError("CLIP model not loaded. Install the clip package.")

        import clip

        # Wrap class names with template if they don't already contain it
        processed_texts = []
        for text in texts:
            if not text.startswith("a photo of"):
                text = self.text_template.format(text)
            processed_texts.append(text)

        tokens = clip.tokenize(processed_texts).to(self.device)

        with torch.no_grad():
            text_features = self.model.encode_text(tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        return text_features.float()

    def encode_image(self, images):
        """Encode images into feature vectors.

        Args:
            images: [B, C, H, W] image tensor (already preprocessed).
                    Expected to be in the correct CLIP preprocessing format.

        Returns:
            image_features: [B, 512] normalized image features.
        """
        if self.model is None:
            raise RuntimeError("CLIP model not loaded. Install the clip package.")

        with torch.no_grad():
            image_features = self.model.encode_image(images)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        return image_features.float()

    def encode_image_with_grad(self, images):
        """Encode images preserving gradient flow through the input.

        Unlike encode_image(), this does NOT use torch.no_grad(), allowing
        gradients to flow through the input tensor (e.g., for TAM's color
        alignment loss). CLIP's own parameters remain frozen via requires_grad=False.

        Args:
            images: [B, C, H, W] image tensor.

        Returns:
            image_features: [B, 512] normalized image features.
        """
        if self.model is None:
            raise RuntimeError("CLIP model not loaded.")

        image_features = self.model.encode_image(images)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        return image_features.float()

    def compute_similarity(self, image_features, text_features):
        """Compute image-text similarity scores.

        Args:
            image_features: [B, 512] image feature vectors.
            text_features: [N, 512] text feature vectors (one per class).

        Returns:
            similarity: [B, N] similarity matrix (logit-scaled).
        """
        # CLIP uses a learned temperature parameter
        if self.model is not None:
            temperature = self.model.logit_scale.exp()
        else:
            temperature = 100.0  # Default CLIP temperature

        # Compute cosine similarity
        similarity = image_features @ text_features.t() * temperature
        return similarity

    def get_text_features_for_classes(self, class_names):
        """Convenience method to get normalized text features for a list of classes.

        Args:
            class_names: List of class name strings.

        Returns:
            text_features: [N, 512] normalized text features.
        """
        texts = [self.text_template.format(name) for name in class_names]
        return self.encode_text(texts)

    def forward(self, images=None, texts=None):
        """Forward pass (encode images and/or texts).

        Args:
            images: Optional [B, C, H, W] image tensor.
            texts: Optional list of text strings.

        Returns:
            dict with 'image_features' and/or 'text_features'.
        """
        result = {}
        if images is not None:
            result['image_features'] = self.encode_image(images)
        if texts is not None:
            result['text_features'] = self.encode_text(texts)
        return result
