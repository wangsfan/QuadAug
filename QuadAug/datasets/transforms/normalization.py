import torchvision.transforms as transforms
class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean  # 均值
        self.std = std  #方差

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        tr = transforms.Normalize(self.mean, self.std)
        return tr(tensor)