import torch
import torchvision.transforms as transforms
import torch.utils.data as data
import cv2
import os
import pickle
import numpy as np
import random
import nltk
from PIL import Image
from build_vocab import Vocabulary
from pycocotools.coco import COCO


class F8kDataset(data.Dataset):
    """COCO Custom Dataset compatible with torch.utils.data.DataLoader."""

    def __init__(self, root_txt, image_path, vocab, transform=None):

        """Set the path for images, captions and vocabulary wrapper.

        Args:
            root_txt: image directory.
            vocab: vocabulary wrapper.
            transform: image transformer.
         """
        self.image_list = []
        self.caption_list = []
        self.neg_image_list = []
        self.image_path = image_path
        self.root_txt = root_txt
        self.vocab = vocab
        self.transform = transform
        with open(self.root_txt, 'r') as f:
            lines = f.readlines()
            for line in lines:
                line_list = line.split(',')
                self.image_list.append(line_list[0])
                self.caption_list.append(line_list[1].replace('\n', ''))
        self.neg_image_list = self.image_list.copy()
        for i in range(10 % len(self.neg_image_list)):                            
            self.neg_image_list.insert(0,self.neg_image_list.pop())
        
    def __getitem__(self, index):
        """Returns one data pair (image and caption)."""
        # print(os.path.join(self.image_path,"//"+self.image_list[index]))
        # # image = cv2.imread(self.image_path+self.image_list[index])
        # print(image.shape)
        # cv2.imshow("index", image)
        # cv2.WaitKey(0)
        vocab = self.vocab
        l = len(self.image_list)
        neg_index = random.randint(0,l-1)
        while neg_index > index - 5 and neg_index < index + 5:
            neg_index = random.randint(0,l-1)
        image = Image.open(os.path.join(self.image_path, self.image_list[index])).convert('RGB')
        neg_image = Image.open(os.path.join(self.image_path, self.image_list[neg_index])).convert('RGB')
        caption = self.caption_list[index]
        if self.transform is not None:
            image = self.transform(image)
            neg_image = self.transform(neg_image)

        # Convert caption (string) to word ids.
        tokens = nltk.tokenize.word_tokenize(str(caption).lower())

        caption = []
        neg_caption = []
        caption.append(vocab('<start>'))
        caption.extend([vocab(token) for token in tokens])
        caption.append(vocab('<end>'))
        target = torch.Tensor(caption)
        return image, target, neg_image

    def __len__(self):
        return len(self.image_list)


def collate_fn(data):
    """Creates mini-batch tensors from the list of tuples (image, caption).

    We should build custom collate_fn rather than using default collate_fn,
    because merging caption (including padding) is not supported in default.

    Args:
        data: list of tuple (image, caption).
            - image: torch tensor of shape (3, 256, 256).
            - caption: torch tensor of shape (?); variable length.

    Returns:
        images: torch tensor of shape (batch_size, 3, 256, 256).
        targets: torch tensor of shape (batch_size, padded_length).
        lengths: list; valid length for each padded caption.
    """
    # Sort a data list by caption length (descending order).
    data.sort(key=lambda x: len(x[1]), reverse=True)
    images, captions, neg_images = zip(*data)

    # Merge images (from tuple of 3D tensor to 4D tensor).
    images = torch.stack(images, 0)
    neg_images = torch.stack(neg_images, 0)

    # Merge captions (from tuple of 1D tensor to 2D tensor).
    lengths = [len(cap) for cap in captions]
    targets = torch.zeros(len(captions), max(lengths)).long()
    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end]

    return images, neg_images, targets, lengths


def get_loader(root_txt, image_path, vocab, transform, batch_size, shuffle, num_workers):
    """Returns torch.utils.data.DataLoader for custom coco dataset."""
    # self, root_txt,image_path, vocab, transform = None

    F8K = F8kDataset(root_txt=root_txt,
                     image_path=image_path,
                     vocab=vocab,
                     transform=transform)

    # Data loader for COCO dataset
    # This will return (images, captions, lengths) for each iteration.
    # images: a tensor of shape (batch_size, 3, 224, 224).
    # captions: a tensor of shape (batch_size, padded_length).
    # lengths: a list indicating valid length for each caption. length is (batch_size).
    data_loader = torch.utils.data.DataLoader(dataset=F8K,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              num_workers=num_workers,
                                              collate_fn=collate_fn)
    return data_loader


# if __name__ == '__main__':
#     f8k = F8kDataset(
#         root_txt=r'/Users/a.w/Desktop/pythonProject/attack-on-multimodal-models/attack-on-multimodal-models/image_captioning/Flickr8k/captions.txt',
#         image_path=r'/Users/a.w/Desktop/pythonProject/attack-on-multimodal-models/attack-on-multimodal-models/image_captioning/Flickr8k/Flickr8k_Dataset/Flicker8k_Dataset/')
#     img, txt = f8k.__getitem__(1)
#     print(txt)
#     img.show()
