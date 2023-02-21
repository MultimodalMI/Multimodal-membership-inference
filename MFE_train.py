import argparse
import torch
import torch.nn as nn
import numpy as np
import os
import pickle
#from data_loader_f8k import get_loader
from dcmh_data_loader import get_loader
from build_vocab_f8k import Vocabulary
# from model import EncoderCNN, DecoderRNN
from dcmh_model import ImageModel, TextModel
from torch.nn.utils.rnn import pack_padded_sequence
from torchvision import transforms


# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main(args):
    # Create model directory
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)
    
    # Image preprocessing, normalization for the pretrained resnet
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(args.crop_size),
        transforms.RandomHorizontalFlip(), 
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

    # Build the models
    img_model = ImageModel(args.embed_size).to(device)
    text_model = TextModel(len(vocab), args.embed_size).to(device)
    
    # Loss and optimizer
    loss_f = nn.MSELoss()
    text_optimizer = torch.optim.Adam(list(text_model.parameters()), lr=args.learning_rate)
    img_optimizer = torch.optim.Adam(list(img_model.parameters()), lr=args.learning_rate)
    
    # Train the models
    total_step = len(data_loader)
    print(total_step)
    for epoch in range(args.num_epochs):
        # train image_net
        for i, (images, captions, lengths) in enumerate(data_loader):
            # print(captions.shape)
            # Set mini-batch dataset
            images = images.to(device)
            captions = captions.to(device)
            
            # Forward, backward and optimize
            img_feature = img_model(images)
            text_feature = text_model(captions)
            loss = loss_f(img_feature, text_feature)
            img_model.zero_grad()
            # encoder.zero_grad()
            loss.backward()
            img_optimizer.step()

            # Print log info
            if i % args.log_step == 0:
                print('Image_net: Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                      .format(epoch, args.num_epochs, i, total_step, loss.item())) 

        for i, (images, captions, lengths) in enumerate(data_loader):
            
            # Set mini-batch dataset
            images = images.to(device)
            captions = captions.to(device)
            
            # Forward, backward and optimize
            img_feature = img_model(images)
            text_feature = text_model(captions)
            loss = loss_f(img_feature, text_feature)
            text_model.zero_grad()
            # encoder.zero_grad()
            loss.backward()
            text_optimizer.step()

            # Print log info
            if i % args.log_step == 0:
                print('Text_net: Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Perplexity: {:5.4f}'
                      .format(epoch, args.num_epochs, i, total_step, loss.item(), np.exp(loss.item())))     
            
            # Save the model checkpoints
            if (i+1) % args.save_step == 0:
                torch.save(img_model.state_dict(), os.path.join(
                    args.model_path, 'com_imgmodel-{}-{}.ckpt'.format(epoch+1, i+1)))
                torch.save(text_model.state_dict(), os.path.join(
                    args.model_path, 'com_textmodel-{}-{}.ckpt'.format(epoch+1, i+1)))
    
    torch.save(img_model.state_dict(), os.path.join(
                    args.model_path, 'com_imgmodel-final.ckpt'.format(epoch+1, i+1)))
    torch.save(text_model.state_dict(), os.path.join(
                    args.model_path, 'com_textmodel-final.ckpt'.format(epoch+1, i+1)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='models/', help='path for saving trained models')
    parser.add_argument('--crop_size', type=int, default=224 , help='size for randomly cropping images')
    parser.add_argument('--vocab_path', type=str, default='data/com_vocab.pkl', help='path for vocabulary wrapper')
    parser.add_argument('--image_dir', type=str, default='data/CombinedSet/image', help='directory for images')
    parser.add_argument('--caption_path', type=str, default='data/CombinedSet/com.txt', help='path for train annotation txt file')
    parser.add_argument('--log_step', type=int , default=10, help='step size for printing log info')
    parser.add_argument('--save_step', type=int , default=470, help='step size for saving trained models')
    
    # Model parameters
    parser.add_argument('--embed_size', type=int , default=256, help='dimension of word embedding vectors')
    parser.add_argument('--hidden_size', type=int , default=512, help='dimension of lstm hidden states')
    parser.add_argument('--num_layers', type=int , default=1, help='number of layers in lstm')
    
    parser.add_argument('--num_epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    args = parser.parse_args()
    print(args)
    main(args)
