import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence


class ImageModel(nn.Module):
    def __init__(self, hash_code_len):
        """Load the pretrained ResNet-152 and replace top fc layer."""
        super(ImageModel, self).__init__()
        resnet = models.resnet152(pretrained=True)
        modules = list(resnet.children())[:-1]      # delete the last fc layer.
        self.resnet = nn.Sequential(*modules)
        self.linear = nn.Linear(resnet.fc.in_features, hash_code_len)
        self.bn = nn.BatchNorm1d(hash_code_len, momentum=0.01)
        
    def forward(self, images):
        """Extract feature vectors from input images."""
        with torch.no_grad():
            features = self.resnet(images)
        features = features.reshape(features.size(0), -1)
        features = self.bn(self.linear(features))
        return features


class TextModel(nn.Module):
    def __init__(self, vocab_size, hash_code_len):
        """Set the hyper-parameters and build the layers."""
        super(TextModel, self).__init__()
        self.conv1 = nn.Conv1d(1, 8192, kernel_size=vocab_size, stride=1)
        self.conv2 = nn.Conv1d(8192, hash_code_len, kernel_size=1, stride=1)
        self.fc1 = nn.Linear(vocab_size, 512)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(512, hash_code_len)
        self.bn = nn.BatchNorm1d(hash_code_len, momentum=0.01)
        
    def forward(self, input):
        """Extract feature vectors from input text."""
        hidden = self.fc1(input)
        hidden = self.relu(hidden)
        outputs = self.bn(self.fc2(hidden))
        return outputs
