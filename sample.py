import torch
import math
import matplotlib.pyplot as plt
import numpy as np 
import argparse
import pickle 
import os
from torchvision import transforms 
from build_vocab import Vocabulary
from model_vgg import EncoderCNN as EncoderResnet
from model_vgg import DecoderRNN as DecoderResnet
from PIL import Image


# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_image(image_path, transform=None):
    image = Image.open(image_path).convert('RGB')
    image = image.resize([224, 224], Image.LANCZOS)
    
    if transform is not None:
        image = transform(image).unsqueeze(0)
    
    return image

def main(args):
    # Image preprocessing
    transform = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Normalize((0.485, 0.456, 0.406), 
                             (0.229, 0.224, 0.225))])
    
    # Load vocabulary wrapper
    with open(args.vocab_path, 'rb') as f:
        vocab = pickle.load(f)

    # Build models
    encoder = EncoderResnet(args.embed_size).eval()  # eval mode (batchnorm uses moving mean/variance)
    decoder = DecoderResnet(args.embed_size, args.hidden_size, len(vocab), args.num_layers)
    encoder = encoder.to(device)
    decoder = decoder.to(device)

    # Load the trained model parameters
    encoder.load_state_dict(torch.load(args.encoder_path,  map_location=torch.device('cpu')))
    decoder.load_state_dict(torch.load(args.decoder_path,  map_location=torch.device('cpu')))

    # Prepare an image
    image = load_image(args.image, transform)
    image_tensor = image.to(device)
    
    # Generate an caption from the image
    feature = encoder(image_tensor)
    print(feature.shape)
    probs,sampled_ids = decoder.sample(feature)
    feature = feature.unsqueeze(1)
    features = feature[0].detach().numpy()
    features = features.reshape(256,)
    print(features.shape)
    probs = probs[0].cpu().numpy()
    sampled_ids = sampled_ids[0].cpu().numpy()          # (1, max_seq_length) -> (max_seq_length)
    
    # Convert word_ids to words
    sampled_caption = []
    i = 0
    prob = 1.0
    print(len(vocab.word2idx))
    for word_id in sampled_ids:
        word = vocab.idx2word[word_id]
        sampled_caption.append(word)
        if word != '<start>':
            prob = prob * probs[i]
        if word == '<end>':
            break
        i = i + 1
    prob = math.pow(prob, 1.0/(i-1))
    sentence = ' '.join(sampled_caption)
    
    # Print out the image and the generated caption
    print (prob)
    print (sentence)
    image = Image.open(args.image)
    plt.imshow(np.asarray(image))
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, required=True, help='input image for generating caption')
    parser.add_argument('--encoder_path', type=str, default='models/f8k_pkl/vgg/encoder-vgg-f8k-final.ckpt', help='path for trained encoder')
    parser.add_argument('--decoder_path', type=str, default='models/f8k_pkl/vgg/decoder-vgg-f8k-final.ckpt', help='path for trained decoder')
    parser.add_argument('--vocab_path', type=str, default='data/vocab_f8k.pkl', help='path for vocabulary wrapper')
    
    # Model parameters (should be same as paramters in train.py)
    parser.add_argument('--embed_size', type=int , default=256, help='dimension of word embedding vectors')
    parser.add_argument('--hidden_size', type=int , default=512, help='dimension of lstm hidden states')
    parser.add_argument('--num_layers', type=int , default=1, help='number of layers in lstm')
    args = parser.parse_args()
    main(args)
