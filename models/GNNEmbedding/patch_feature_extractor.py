import torchvision
import torch.nn as nn
import torch
import importlib

def dynamic_import_from(source_file: str, class_name: str):
    """Do a from source_file import class_name dynamically

    Args:
        source_file (str): Where to import from
        class_name (str): What to import

    Returns:
        Any: The class to be imported
    """
    module = importlib.import_module(source_file)
    return getattr(module, class_name)

class PatchFeatureExtractor(nn.Module):
    """Helper class to use a CNN to extract features from an image"""

    def __init__(self, architecture: str, device: torch.device) -> None:
        """
        Create a patch feature extracter of a given architecture and put it on GPU if available.

        Args:
            architecture (str): String of architecture. According to torchvision.models syntax.
            device (torch.device): Torch Device.
        """
        super(PatchFeatureExtractor, self).__init__()
        self.device = device

        if architecture.startswith("s3://mlflow"):
            model = self._get_mlflow_model(url=architecture)
        elif architecture.endswith(".pth"):
            model = self._get_local_model(path=architecture)
        else:
            model = self._get_torchvision_model(architecture).to(self.device)

        self.num_features = self._get_num_features(model)
        self.model = self._remove_classifier(model)

    @staticmethod
    def _get_num_features(model: nn.Module) -> int:
        """
        Get the number of features of a given model.

        Args:
            model (nn.Module): A PyTorch model.

        Returns:
            int: Number of output features.
        """
        if hasattr(model, "model"):
            model = model.model
        if isinstance(model, torchvision.models.resnet.ResNet):
            return model.fc.in_features
        else:
            classifier = model.classifier[-1]
            if isinstance(classifier, nn.Sequential):
                classifier = classifier[-1]
            return classifier.in_features

    def _get_local_model(self, path: str) -> nn.Module:
        """
        Load a model from a local path.

        Args:
            path (str): Path to the model.

        Returns:
            nn.Module: A PyTorch model.
        """
        model = torch.load(path, map_location=self.device)
        return model

    def _get_mlflow_model(self, url: str) -> nn.Module:
        """
        Load a MLflow model from a given URL.

        Args:
            url (str): Model url.

        Returns:
            nn.Module: A PyTorch model.
        """
        import mlflow

        model = mlflow.pytorch.load_model(url, map_location=self.device)
        return model

    def _get_torchvision_model(self, architecture: str) -> nn.Module:
        """
        Returns a torchvision model from a given architecture string.

        Args:
            architecture (str): Torchvision model description.

        Returns:
            nn.Module: A pretrained pytorch model.
        """
        model_class = dynamic_import_from("torchvision.models", architecture)
        model = model_class(pretrained=True)
        model = model.to(self.device)
        return model

    @staticmethod
    def _remove_classifier(model: nn.Module) -> nn.Module:
        """
        Returns the model without the classifier to get embeddings.

        Args:
            model (nn.Module): Classifiation model.

        Returns:
            nn.Module: Embedding model.
        """
        if hasattr(model, "model"):
            model = model.model
        if isinstance(model, torchvision.models.resnet.ResNet):
            model.fc = nn.Sequential()
        else:
            model.classifier[-1] = nn.Sequential()
        return model

    def forward(self, patch: torch.Tensor) -> torch.Tensor:
        """
        Computes the embedding of a normalized image input.

        Args:
            image (torch.Tensor): Normalized image input.

        Returns:
            torch.Tensor: Embedding of image.
        """
        patch = patch.to(self.device)
        with torch.no_grad():
            embeddings = self.model(patch).squeeze()
        return embeddings
