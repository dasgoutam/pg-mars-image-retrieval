#!/usr/bin/env python3
import torch
import torch.nn as nn
import torchvision.models as models
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchvision.utils import save_image
import sys
from PIL import Image
import os

PATH_DATA_TRAIN = '~/codebase-v1/data/data/train/'
# PATH_DATA_TEST = '~/codebase-v1/data/data/test/'
PATH_DATA_TEST = '~/Documents/TUD/project_group/data/test/'
PATH_SAVE = '~/codebase-v1/aec_model/results/'

# Define input image from sys arguments
input_image_class = sys.argv[1]
num_epochs = sys.argv[2]
distance_measure = sys.argv[3]

# Saved autoencoder
saved_model_path = "aec_resnet_"+str(num_epochs)+"ep_001lr.pth"

# Set device CPU or GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load the pre-trained Resnet-18 model
resnet18 = models.resnet18(pretrained=True)

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

def normalize_feature_vector(feature_vector):
    l2_norm = torch.norm(feature_vector, p=2)
    normalized_vector = feature_vector / l2_norm
    return normalized_vector

def transform_and_to_device(img):
    img = transform(img)
    img = img.to(device)
    return img

def global_pool_and_flatten(feature_vector):
    global_pooling = torch.nn.AdaptiveAvgPool2d((1, 1))
    pooled_features = global_pooling(feature_vector)
    flattened_features = torch.flatten(pooled_features, start_dim=1)
    flattened_features = torch.squeeze(flattened_features)
    return flattened_features

class Autoencoder(nn.Module):
    def __init__(self, encoder):
        super(Autoencoder, self).__init__()

        self.encoder = nn.Sequential(*list(encoder.children())[:-2])
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, 
                               padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


# Create an instance of the autoencoder
autoencoder = Autoencoder(resnet18)
autoencoder.to(device)

saved_model_dict = torch.load(saved_model_path, map_location=device)

autoencoder.load_state_dict(saved_model_dict)

feature_extractor = nn.Sequential(*list(autoencoder.encoder.children()))
feature_extractor.load_state_dict(autoencoder.encoder.state_dict())

RESULT_CLASS_COUNT = 0
TOTAL_CLASS_COUNT = 0

for image in os.listdir('data/data/test/' + input_image_class + '/'):
    INPUT_IMAGE_PATH = 'data/data/test/' + input_image_class + '/' + image
    print("Processing - " + INPUT_IMAGE_PATH)

    img_input = Image.open(INPUT_IMAGE_PATH)
    img_input = transform_and_to_device(img_input)
    img_input = img_input.unsqueeze(0)

    fe_img_input = feature_extractor(img_input)
    fe_img_input = global_pool_and_flatten(fe_img_input)
    fe_img_input = normalize_feature_vector(fe_img_input)

    test_dataset = datasets.ImageFolder(PATH_DATA_TRAIN, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

    score_vectors = []
    image_list = []
    class_list = []
    file_path_list = []

    for batch_idx, (image, labels) in enumerate(test_loader):
        image = image.to(device)
        fe_image = feature_extractor(image)
        fe_image = global_pool_and_flatten(fe_image)
        fe_image = normalize_feature_vector(fe_image)
        if distance_measure == 'l2':
            score = torch.nn.functional.pairwise_distance(fe_img_input, fe_image, p=2)
        elif distance_measure == 'cosine':
            score = torch.nn.functional.cosine_similarity(fe_img_input, fe_image, dim=0)
            score = 1 - score
        score_vectors.append(score.item())
        image_list.append(image)
        file_paths = test_loader.dataset.samples[batch_idx][0]
        class_names = os.path.basename(os.path.dirname(file_paths))
        file_path_list.append(file_paths)
        class_list.append(class_names) 

    # sorted_images = [x for _,x in sorted(zip(score_vectors, image_list))]
    sorted_indices = sorted(range(len(score_vectors)), key=lambda i: score_vectors[i])
    sorted_images = [image_list[i] for i in sorted_indices]
    sorted_class_list = [class_list[i] for i in sorted_indices]
    sorted_file_list = [file_path_list[i] for i in sorted_indices]

    print(sorted_class_list[:10])
    results = sorted_class_list[:10]
    count_result = results.count(input_image_class)
    print(input_image_class + " appears " + str(count_result) + " times")
    RESULT_CLASS_COUNT = RESULT_CLASS_COUNT + count_result
    TOTAL_CLASS_COUNT = TOTAL_CLASS_COUNT + 10


print("Final Results - ")
print(RESULT_CLASS_COUNT)
print(TOTAL_CLASS_COUNT)