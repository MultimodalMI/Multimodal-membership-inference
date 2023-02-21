import argparse
import torch
import torch.nn as nn
import torch.nn.functional as func
import numpy as np
import os
import pickle
from data_loader_f8k import get_loader
from build_vocab import Vocabulary
from model_resnet import EncoderCNN, DecoderRNN
from torch.nn.utils.rnn import pack_padded_sequence
from torchvision import transforms
from opacus import PrivacyEngine


# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class ConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self):
        super(ConLoss, self).__init__()

    def forward(self, positive, negative):
        dist = torch.pow(func.pairwise_distance(negative, positive), 2)
        dist = torch.exp(1 / dist)
        loss = torch.mean(dist)
        return loss

def main(args):
    # Create model directory
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)
    
    # Image preprocessing, normalization for the pretrained resnet
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(args.crop_size),
        transforms.ToTensor(), 
        transforms.Normalize((0.485, 0.456, 0.406), 
                             (0.229, 0.224, 0.225))])
    
    # Load vocabulary wrapper
    with open(args.vocab_path, 'rb') as f:
        vocab = pickle.load(f)
    
    # Build data loader
    data_loader = get_loader(args.caption_path, args.image_dir, vocab,
                             transform, args.batch_size,
                             shuffle=True, num_workers=args.num_workers)

    # Build th_ models
    encoder = EncoderCNN(args.embed_size).to(device)
    decoder = DecoderRNN(args.embed_size, args.hidden_size, len(vocab), args.num_layers).to(device)
    
    # Loss and optimizer
    criterion1 = ConLoss()
    criterion2 = nn.CrossEntropyLoss()
    params = list(decoder.parameters()) + list(encoder.parameters())
    optimizer = torch.optim.SGD(params, lr=args.learning_rate)
    optimizer1 = torch.optim.SGD(list(encoder.parameters()), lr=args.learning_rate)
    # Train the models
    total_step = len(data_loader)
    print(total_step)
    for epoch in range(args.num_epochs):
        for i, (images, neg_images, captions, lengths) in enumerate(data_loader):
            
            # Set mini-batch dataset
            images = images.to(device)
            neg_images = neg_images.to(device)
            features = encoder(images)
            neg_features = encoder(neg_images)
            # Forward, backward and optimize
            encoder.zero_grad()
            loss1 = criterion1(features, neg_features)
            loss1.backward()
            optimizer1.step()

            if i % args.log_step == 0:
                print('For encoder: Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                      .format(epoch, args.num_epochs, i, total_step, loss1.item()))

    for epoch in range(args.num_epochs):
        for i, (images, neg_images, captions, lengths) in enumerate(data_loader):
            images = images.to(device) 
            captions = captions.to(device)
            targets = pack_padded_sequence(captions, lengths, batch_first=True)[0]
            features = encoder(images)
            outputs = decoder(features, captions, lengths)
            loss = criterion2(outputs, targets)
            decoder.zero_grad()
            encoder.zero_grad()
            loss.backward()
            optimizer.step()

            # Print log info
            if i % args.log_step == 0:
                print('For decoder: Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Perplexity: {:5.4f}'
                      .format(epoch, args.num_epochs, i, total_step, loss.item(), np.exp(loss.item()))) 
                
            # Save the model checkpoints
            if (i+1) % args.save_step == 0:
                torch.save(decoder.state_dict(), os.path.join(
                    args.model_path, 'decoder-{}-{}.ckpt'.format(epoch+1, i+1)))
                torch.save(encoder.state_dict(), os.path.join(
                    args.model_path, 'encoder-{}-{}.ckpt'.format(epoch+1, i+1)))

    print("finish training:")
    torch.save(encoder.state_dict(), os.path.join(
                    args.model_path, 'encoder-f8k-final.ckpt'))
    torch.save(decoder.state_dict(), os.path.join(
                    args.model_path, 'decoder-f8k-final.ckpt'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='models/', help='path for saving trained models')
    parser.add_argument('--crop_size', type=int, default=224 , help='size for randomly cropping images')
    parser.add_argument('--vocab_path', type=str, default='data/vocab_f8k.pkl', help='path for vocabulary wrapper')
    parser.add_argument('--image_dir', type=str, default='data/Flickr8k/Images', help='directory for images')
    parser.add_argument('--caption_path', type=str, default='data/Flickr8k/captions.txt', help='path for train annotation txt file')
    parser.add_argument('--log_step', type=int , default=10, help='step size for printing log info')
    parser.add_argument('--save_step', type=int , default=30, help='step size for saving trained models')
    
    # Model parameters
    parser.add_argument('--embed_size', type=int , default=256, help='dimension of word embedding vectors')
    parser.add_argument('--hidden_size', type=int , default=512, help='dimension of lstm hidden states')
    parser.add_argument('--num_layers', type=int , default=1, help='number of layers in lstm')
    
    parser.add_argument('--num_epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--learning_rate', type=float, default=0.01)
    args = parser.parse_args()
    print(args)
    main(args)