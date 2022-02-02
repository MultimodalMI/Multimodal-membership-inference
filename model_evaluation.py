import torch
from torch import nn
from torch.autograd import Variable
from torch.optim import SGD
import torch.utils.data as Data
import nltk
# nltk.download('punkt')
# from nltk.corpus import stopwords
# nltk.download('stopwords')
import difflib
import math
import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd
import argparse
import pickle 
import os
from pycocotools.coco import COCO
from torchvision import transforms 
from build_vocab import Vocabulary
from model_resnet import EncoderCNN as EncoderResnet
from model_resnet import DecoderRNN as DecoderResnet
from model_vgg import EncoderCNN as EncoderVGG
from model_vgg import DecoderRNN as DecoderVGG
from dcmh_model import ImageModel, TextModel
from PIL import Image
import random
from sklearn import svm
from nltk.translate.bleu_score import sentence_bleu
from sklearn.metrics import roc_auc_score
from rouge import Rouge
from mpl_toolkits.mplot3d import Axes3D

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Image preprocessing
transform = transforms.Compose([
    transforms.ToTensor(), 
    transforms.Normalize((0.485, 0.456, 0.406), 
                             (0.229, 0.224, 0.225))])

# image loader
def load_image(image_path, transform=None):
    image = Image.open(image_path).convert('RGB')
    image = image.resize([224, 224], Image.LANCZOS)
    
    if transform is not None:
        image = transform(image).unsqueeze(0)
    
    return image

# Load vocabulary wrapper
with open("data/vocab_f8k.pkl", 'rb') as f:
    vocab_f8k = pickle.load(f)
with open("data/co17_vocab.pkl", 'rb') as f:
    vocab_coco = pickle.load(f)
with open("data/ipar_vocab.pkl", 'rb') as f:
    vocab_iapr = pickle.load(f)
with open("dcmh_models/vocab.pkl", 'rb') as f:
    vocab_dcmh = pickle.load(f)

# model initialization
# Build models
# f8k-resnet
encoder_fr = EncoderResnet(256).eval()  # eval mode (batchnorm uses moving mean/variance)
decoder_fr = DecoderResnet(256, 512, len(vocab_f8k), 1)
encoder_fr = encoder_fr.to(device)
decoder_fr = decoder_fr.to(device)
# f8k-vgg
encoder_fv = EncoderVGG(256).eval()  # eval mode (batchnorm uses moving mean/variance)
decoder_fv = DecoderVGG(256, 512, len(vocab_f8k), 1)
encoder_fv = encoder_fv.to(device)
decoder_fv = decoder_fv.to(device)
# coco-resnet
encoder_cr = EncoderResnet(256).eval()  # eval mode (batchnorm uses moving mean/variance)
decoder_cr = DecoderResnet(256, 512, len(vocab_coco), 1)
encoder_cr = encoder_cr.to(device)
decoder_cr = decoder_cr.to(device)
# coco-vgg
encoder_cv = EncoderVGG(256).eval()  # eval mode (batchnorm uses moving mean/variance)
decoder_cv = DecoderVGG(256, 512, len(vocab_coco), 1)
encoder_cv = encoder_cv.to(device)
decoder_cv = decoder_cv.to(device)
# ipar-resnet
encoder_ir = EncoderResnet(256).eval()  # eval mode (batchnorm uses moving mean/variance)
decoder_ir = DecoderResnet(256, 512, len(vocab_iapr), 1)
encoder_ir = encoder_ir.to(device)
decoder_ir = decoder_ir.to(device)
# iapr-vgg
encoder_iv = EncoderVGG(256).eval()  # eval mode (batchnorm uses moving mean/variance)
decoder_iv = DecoderVGG(256, 512, len(vocab_iapr), 1)
encoder_iv = encoder_iv.to(device)
decoder_iv = decoder_iv.to(device)
# Build dcmh models
# coco-f8k
image_encoder_cf = ImageModel(256).eval()
text_encoder_cf = TextModel(len(vocab_dcmh),256).eval()
image_encoder_cf = image_encoder_cf.to(device)
text_encoder_cf = text_encoder_cf.to(device)
# coco-iapr
image_encoder_ci = ImageModel(256).eval()
text_encoder_ci = TextModel(len(vocab_dcmh),256).eval()
image_encoder_ci = image_encoder_ci.to(device)
text_encoder_ci = text_encoder_ci.to(device)
# iapr-f8k
image_encoder_if = ImageModel(256).eval()
text_encoder_if = TextModel(len(vocab_dcmh),256).eval()
image_encoder_if = image_encoder_if.to(device)
text_encoder_if = text_encoder_if.to(device)


# Load the trained model parameters
# f8k-resnet
encoder_fr.load_state_dict(torch.load('models/f8k_pkl/resnet/encoder-resnet-f8k-final.pkl',  map_location=torch.device('cpu')))
decoder_fr.load_state_dict(torch.load('models/f8k_pkl/resnet/decoder-resnet-f8k-final.pkl',  map_location=torch.device('cpu')))
# f8k-vgg
encoder_fv.load_state_dict(torch.load('models/f8k_pkl/vgg/encoder-vgg-f8k-final.pkl',  map_location=torch.device('cpu')))
decoder_fv.load_state_dict(torch.load('models/f8k_pkl/vgg/decoder-vgg-f8k-final.pkl',  map_location=torch.device('cpu')))
# coco-resnet
encoder_cr.load_state_dict(torch.load('models/coco_pkl/resnet/encoder-coco17-final.pkl',  map_location=torch.device('cpu')))
decoder_cr.load_state_dict(torch.load('models/coco_pkl/resnet/decoder-coco17-final.pkl',  map_location=torch.device('cpu')))
# coco-vgg
encoder_cv.load_state_dict(torch.load('models/coco_pkl/vgg/vgg_encoder-coco17-final.pkl',  map_location=torch.device('cpu')))
decoder_cv.load_state_dict(torch.load('models/coco_pkl/vgg/vgg_decoder-coco17-final.pkl',  map_location=torch.device('cpu')))
# iapr-resnet
encoder_ir.load_state_dict(torch.load('models/iapr_pkl/resnet/resnet_encoder-ipar-final.pkl',  map_location=torch.device('cpu')))
decoder_ir.load_state_dict(torch.load('models/iapr_pkl/resnet/resnet_decoder-ipar-final.pkl',  map_location=torch.device('cpu')))
# iapr-vgg
encoder_iv.load_state_dict(torch.load('models/iapr_pkl/vgg/vgg_encoder-ipar-final.pkl',  map_location=torch.device('cpu')))
decoder_iv.load_state_dict(torch.load('models/iapr_pkl/vgg/vgg_decoder-ipar-final.pkl',  map_location=torch.device('cpu')))
# dcmh models
# coco-f8k
image_encoder_cf.load_state_dict(torch.load('dcmh_models/CocoF8k_imgmodel-final.ckpt',  map_location=torch.device('cpu')))
text_encoder_cf.load_state_dict(torch.load('dcmh_models/CocoF8k_textmodel-final.ckpt',  map_location=torch.device('cpu')))
# coco-iapr
image_encoder_ci.load_state_dict(torch.load('dcmh_models/CocoIapr_imgmodel-final.ckpt',  map_location=torch.device('cpu')))
text_encoder_ci.load_state_dict(torch.load('dcmh_models/CocoIapr_textmodel-final.ckpt',  map_location=torch.device('cpu')))
# iapr-f8k
image_encoder_if.load_state_dict(torch.load('dcmh_models/IaprF8k_imgmodel-final.ckpt',  map_location=torch.device('cpu')))
text_encoder_if.load_state_dict(torch.load('dcmh_models/IaprF8k_textmodel-final.ckpt',  map_location=torch.device('cpu')))

# load all the captions
image_list = []
caption_list = []
with open('f8k_captioning.txt', 'r') as f:
    lines = f.readlines()
    for line in lines:
        line_list = line.split(',', 1)
        if(len(line_list) == 2):
            image_list.append(line_list[0])
            caption_list.append(line_list[1].replace('\n', ''))

with open('coco_captioning.txt', 'r') as f:
    lines = f.readlines()
    for line in lines:
        line_list = line.split(',', 1)
        if(len(line_list) == 2):
            image_list.append(line_list[0])
            caption_list.append(line_list[1].replace('\n', ''))

with open('iapr_captioning.txt', 'r') as f:
    lines = f.readlines()
    for line in lines:
        line_list = line.split(',', 1)
        if(len(line_list) == 2):
            image_list.append(line_list[0])
            caption_list.append(line_list[1].replace('\n', ''))


# load image path
f8k_path = 'Flickr8k/images/'
f8k_image_path = os.listdir(f8k_path)
coco_path = 'coco_data/resized2017/'
coco_image_path = os.listdir(coco_path)
iapr_path = 'ipar_data/resizedIapr/'
iapr_image_path = os.listdir(iapr_path)
output_path = 'output.txt'

# randomly choose 1000 image from three dataset
# random.seed(10)
f8k_images = random.sample(f8k_image_path, 1000)
f8k_images_shadow = random.sample(f8k_image_path, 1000)
coco_images = random.sample(coco_image_path, 1000)
coco_images_shadow = random.sample(coco_image_path, 1000)
iapr_images = random.sample(iapr_image_path, 1000)
iapr_images_shadow = random.sample(iapr_image_path, 1000)

# image-captions
f8k_captions = {}
f8k_captions_shadow = {}
coco_captions = {}
coco_captions_shadow = {}
iapr_captions = {}
iapr_captions_shadow = {}

for image in f8k_images:
    f8k_captions[image] = []
    for i in range(len(image_list)):
        if image_list[i] == image:
            tokens = nltk.tokenize.word_tokenize(str(caption_list[i]).lower())
            caption = []
            caption.extend([token for token in tokens])
            f8k_captions[image].append(caption)

for image in f8k_images_shadow:
    f8k_captions_shadow[image] = []
    for i in range(len(image_list)):
        if image_list[i] == image:
            tokens = nltk.tokenize.word_tokenize(str(caption_list[i]).lower())
            caption = []
            caption.extend([token for token in tokens])
            f8k_captions_shadow[image].append(caption)

for image in coco_images:
    coco_captions[image] = []
    for i in range(len(image_list)):
        if image_list[i] == image:
            tokens = nltk.tokenize.word_tokenize(str(caption_list[i]).lower())
            caption = []
            caption.extend([token for token in tokens])
            coco_captions[image].append(caption)

for image in coco_images_shadow:
    coco_captions_shadow[image] = []
    for i in range(len(image_list)):
        if image_list[i] == image:
            tokens = nltk.tokenize.word_tokenize(str(caption_list[i]).lower())
            caption = []
            caption.extend([token for token in tokens])
            coco_captions_shadow[image].append(caption)

for image in iapr_images:
    iapr_captions[image] = []
    for i in range(len(image_list)):
        if image_list[i] == image:
            tokens = nltk.tokenize.word_tokenize(str(caption_list[i]).lower())
            caption = []
            caption.extend([token for token in tokens])
            iapr_captions[image].append(caption)

for image in iapr_images_shadow:
    iapr_captions_shadow[image] = []
    for i in range(len(image_list)):
        if image_list[i] == image:
            tokens = nltk.tokenize.word_tokenize(str(caption_list[i]).lower())
            caption = []
            caption.extend([token for token in tokens])
            iapr_captions_shadow[image].append(caption)

def result(image_name, encoder, decoder, vocab):
    # Prepare an image
    image = load_image(image_name, transform)
    image_tensor = image.to(device)
    # Generate an caption from the image
    feature = encoder(image_tensor)
    probs,sampled_ids = decoder.sample(feature)
    probs = probs[0].cpu().numpy()
    sampled_ids = sampled_ids[0].cpu().numpy()
    sampled_caption = []
    for word_id in sampled_ids:
        word = vocab.idx2word[word_id]
        sampled_caption.append(word)
        if word == '<end>':
            break    
    if '<start>' in sampled_caption:
        sampled_caption.remove('<start>')
    if '<end>' in sampled_caption:
        sampled_caption.remove('<end>')
    return probs, sampled_caption

def onehot(sampled_caption):
    caption_vec = torch.zeros(len(vocab_dcmh))
    for cap in sampled_caption:
        caption_vec[vocab_dcmh(cap)] += 1
    return caption_vec

def feature_similarity(image_path, caption, image_model,text_model):
    # Prepare an image
    image = load_image(image_path, transform)
    image_tensor = image.to(device)
    image_feature = image_model(image_tensor)
    image_feature = image_feature.unsqueeze(1)
    image_feature = image_feature[0].cpu().detach().numpy()
    image_feature = image_feature.reshape(256,)
    caption = torch.reshape(caption, (1,5580))
    text_tensor = caption.to(device)
    text_feature = text_model(text_tensor)
    text_feature = text_feature.unsqueeze(1)
    text_feature = text_feature[0].cpu().detach().numpy()
    text_feature = text_feature.reshape(256,)
    return np.absolute(image_feature - text_feature)

def get_rouge_score(references, candidate):
    rouge1 = 0.0
    rouge2 = 0.0
    rougel = 0.0
    cand = ' '.join(candidate)
    rouge = Rouge()
    for reference in references:
        ref = ' '.join(reference)
        rouge_score = rouge.get_scores(hyps=cand, refs=ref)
        rouge1 += rouge_score[0]["rouge-1"]['f']
        rouge2 += rouge_score[0]["rouge-2"]['f']
        rougel += rouge_score[0]["rouge-l"]['f']
    rouge1 /= len(references)
    rouge2 /= len(references)
    rougel /= len(references)
    return [rouge1, rouge2, rougel]

# rouge score
def models_evaluation(target_encoder, target_decoder, target_dataset, target_dataset_path, target_captioning, target_vocab, n):
    # train attack model
    #     target_prob_data = []
    target_bleu_data = []
    rouge_score_data = []
    target_label = []
    for image in target_dataset:
        imagepath = target_dataset_path + image
        bleu_score = []
        _, captions = result(imagepath, target_encoder, target_decoder, target_vocab)
        bleu_score.append(sentence_bleu(target_captioning[image], captions, weights=(1, 0, 0, 0)))
        bleu_score.append(sentence_bleu(target_captioning[image], captions, weights=(0.5, 0.5, 0, 0)))
        bleu_score.append(sentence_bleu(target_captioning[image], captions, weights=(0.33, 0.33, 0.33, 0)))
        target_bleu_data.append(bleu_score)
        rouge_score = get_rouge_score(target_captioning[image], captions)
        rouge_score_data.append(rouge_score)
    rouge_score_data = np.array(rouge_score_data)
    target_bleu_data = np.array(target_bleu_data)
    print('rouge: ', np.mean(rouge_score_data, axis=0))
    print('bleu:', np.mean(target_bleu_data, axis=0))

models_evaluation(encoder_fr, decoder_fr, f8k_images, f8k_path, f8k_captions, vocab_f8k, 1)