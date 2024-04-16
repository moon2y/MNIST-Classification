
# import some packages you need here
from torch.utils.data import Dataset
import glob as glob
from PIL import Image
import numpy as np
from torchvision import transforms

class MNIST(Dataset):
    """ MNIST dataset

        To write custom datasets, refer to
        https://pytorch.org/tutorials/beginner/data_loading_tutorial.html

    Args:
        data_dir: directory path containing images

    Note:
        1) Each image should be preprocessed as follows:
            - First, all values should be in a range of [0,1]
            - Substract mean of 0.1307, and divide by std 0.30812881
            - These preprocessing can be implemented using torchvision.transforms
        2) Labels can be obtained from filenames: {number}_{label}.png
    """

    def __init__(self, data_dir, model = None):

        # write your codes here
        self.data = data_dir # path your dir
        self.img = glob.glob(self.data+"/*.png")
        self.label = [int(name.split("_")[-1][0]) for name in self.img]
        self.transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.30812881,))
        ])
        self.model_name = model

    def __len__(self):

        # write your codes here
        return len(self.img)

    def __getitem__(self, idx):

        # write your codes here
        img = Image.open(self.img[idx])
        img = self.transform(img)
        if self.model_name == "LeNet5":
            img = np.array(img)
        elif self.model_name == "CustomMLP":
            img = np.array(img).reshape(1, -1)
        label = self.label[idx]

        return img, label

if __name__ == '__main__':

    # write test codes to verify your implementations
    print("image loaded")


