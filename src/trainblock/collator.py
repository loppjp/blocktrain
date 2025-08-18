import torch

class Collator():
    """
    Creates a training or eval batch from dataset outputs
    """

    def __call__(self, batches):
        images = []
        targets = []
        for minibatch in batches:
            X, y = minibatch
            X = X.float()
            images.append(X)
            labels = torch.Tensor([[1]]).int()
            targets.append({"boxes":y, "labels":labels})
        return images, targets