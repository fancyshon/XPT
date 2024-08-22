from torchvision.datasets import ImageFolder
import pickle
from torch.utils.data import Dataset
import torch
from torchvision import transforms


class CustomImageFolder(ImageFolder):
    def get_labels(self):
        # Assuming each image's label is the same as its folder name
        return [self.classes[i] for i in self.targets]
    
class CifarFS(Dataset):
    def __init__(self, pickle_file, transform=None, train=True):
        with open(pickle_file, 'rb') as f:
            self.raw_data = pickle.load(f, encoding='bytes')
        self.data = self.raw_data[b'data']
        self.labels = self.raw_data[b'labels']
        self.train = train       
        self.transform = transform
            
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        label = torch.tensor(self.labels[index])
        data = torch.tensor(self.data[index])        
        data = data.permute(2, 0, 1)

        # normalize = transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        # train_transform = transforms.Compose([
        #     transforms.ToPILImage(),
        #     transforms.Resize(224),
        #     transforms.RandomCrop(224, padding=4),
        #     transforms.RandomHorizontalFlip(),
        #     transforms.ToTensor(),
        #     normalize,
        # ])
        # test_transform = transforms.Compose([
        #     transforms.ToPILImage(),
        #     transforms.Resize(224),
        #     transforms.ToTensor(),
        #     normalize,
        # ])

        # add transforms.ToPILImage() to original transform
        train_transform = transforms.Compose([
            transforms.ToPILImage(),
            self.transform
        ])
        test_transform = transforms.Compose([
            transforms.ToPILImage(),
            self.transform
        ])

        if self.train == True:
            transform = train_transform
            is_train = True
        else:
            transform = test_transform
            is_train = False

        data = transform(data)

        # import cv2

        # image = data.permute(1, 2, 0).numpy()
        # cv2.imwrite('image.jpg', image)
        
        # padding = transforms.Compose([
        #     transforms.ToPILImage(),
        #     transforms.Pad((96, 96)),  # Add padding of 96 on each side to get from 32 to 224
        #     transforms.ToTensor()
        # ])
        # data = padding(data)
        
        # padding = transforms.
        
        return data, label
    
    def get_labels(self):
        return self.labels