from torch.utils.data import Dataset
import pandas as pd
from PIL import Image
from typing import Any, Callable, Tuple

class VQADataset(Dataset):
    """
      This class loads a shrinked version of the VQA dataset (https://visualqa.org/)
      Our shrinked version focus on yes/no questions.
      To load the dataset, we pass a descriptor csv file.

      Each entry of the csv file has this form:

      question_id ; question_type ; image_name ; question ; answer ; image_id

    """

    def __init__(self, path: str, dataset_descriptor: str, image_folder: str, transform: Callable) -> None:
        """
          :param: path : a string that indicates the path to the image and question dataset.
          :param: dataset_descriptor : a string to the csv file name that stores the question ; answer and image name
          :param: image_folder : a string that indicates the name of the folder that contains the images
          :param: transform : a torchvision.transforms wrapper to transform the images into tensors
        """
        super(VQADataset, self).__init__()
        self.descriptor = pd.read_csv(path + '/' + dataset_descriptor, delimiter=';')
        self.path = path
        self.image_folder = image_folder
        self.transform = transform
        self.size = len(self.descriptor)

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int) -> Tuple[Any, Any, Any]:
        """
          returns a tuple : (image, question, answer)
          image is a Tensor representation of the image
          question and answer are strings
        """
        image_name = self.path + '/' + self.image_folder + '/' + self.descriptor["image_name"][idx]

        image = Image.open(image_name).convert("RGB")

        image = self.transform(image)

        question = self.descriptor["question"][idx]

        answer = self.descriptor["answer"][idx]

        return (image, question, answer)
