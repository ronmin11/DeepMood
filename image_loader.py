# image_loader.py
import os
import torch
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import Subset
import timm

def get_image_dataloaders(dataset_path, batch_size, num_workers, loadNumImages=-1, model_name='resnet50.a1_in1k'):
    if model_name == 'custom':
        # Use basic transforms for custom CNN
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3),
        ])
    elif model_name == 'resnet50.a1_in1k':
        # Load model-specific transform
        model = timm.create_model(model_name, pretrained=True, num_classes=0)
        data_config = timm.data.resolve_model_data_config(model)
        transform = timm.data.create_transform(**data_config, is_training=False)
        # train_transform = timm.data.create_transform(**data_config, is_training=True)
        # test_transform = timm.data.create_transform(**data_config, is_training=False)

        # train_transform = transforms.Compose([
        #     transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        #     transforms.RandomHorizontalFlip(),
        #     transforms.RandomApply([
        #         transforms.ColorJitter(0.3, 0.3, 0.2, 0.1),
        #         transforms.RandomAffine(15, translate=(0.1, 0.1), scale=(0.9, 1.1))
        #     ], p=0.7),
        #     transforms.RandomGrayscale(p=0.1),
        #     transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0)),
        #     transforms.ToTensor(),
        #     transforms.Normalize(data_config['mean'], data_config['std']),
        # ])

        # test_transform = transforms.Compose([
        #     transforms.Resize((224, 224)),
        #     transforms.ToTensor(),
        #     transforms.Normalize(data_config['mean'], data_config['std']),
        # ])

    else:
        return ValueError("Model not found")

    def limit_dataset(dataset):
        if loadNumImages <= 0:
            return dataset

        # Ensure balanced sampling
        class_counts = {i: 0 for i in range(len(dataset.classes))}
        indices = []

        for idx, (_, label_idx) in enumerate(dataset.samples):
            if class_counts[label_idx] < loadNumImages:
                indices.append(idx)
                class_counts[label_idx] += 1

        return Subset(dataset, indices)

    train_path = os.path.join(dataset_path, 'train')
    test_path = os.path.join(dataset_path, 'test')

    train_dataset = datasets.ImageFolder(train_path, transform=transform)
    test_dataset = datasets.ImageFolder(test_path, transform=transform)

    train_dataset = limit_dataset(train_dataset)
    test_dataset = limit_dataset(test_dataset)

    #pin_memory is new
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True if torch.cuda.is_available() else False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True if torch.cuda.is_available() else False)

    return train_loader, test_loader
