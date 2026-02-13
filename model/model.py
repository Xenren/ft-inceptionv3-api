import torch
import torchvision.transforms as transforms

def preprocess():
    # InceptionNetV3 data transforms
    # imagenet normalization values
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    data_transforms = {
        "train": transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    299
                ),  # randomly scale and crop to target size
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(
                    brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
                ),  # augment color properties
                transforms.ToTensor(),
                transforms.Normalize(imagenet_mean, imagenet_std),
            ]
        ),
        "val": transforms.Compose(
            [
                transforms.Resize((320, 320)),  # resize slightly larger
                transforms.CenterCrop(299),  # crop to target size
                transforms.ToTensor(),
                transforms.Normalize(imagenet_mean, imagenet_std),
            ]
        ),
    }
