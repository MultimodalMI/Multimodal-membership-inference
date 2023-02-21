import torch
from torch import nn
from torch.autograd import Variable
from torch.optim import SGD
import torch.utils.data as Data
import nltk
import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd
import pickle 
import os
import difflib
import math
from pycocotools.coco import COCO
from torchvision import transforms 
from build_vocab import Vocabulary
from model_resnet import EncoderCNN as EncoderResnet
from model_resnet import DecoderRNN as DecoderResnet
from MFE_model import ImageModel, TextModel
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
with open("data/MFE_vocab.pkl", 'rb') as f:
    vocab_MFE = pickle.load(f)

# model initialization
# Build models
# f8k-resnet
encoder_target = EncoderResnet(256).eval()  # eval mode (batchnorm uses moving mean/variance)
decoder_target = DecoderResnet(256, 512, len(vocab_f8k), 1)
encoder_target = encoder_target.to(device)
decoder_target = decoder_target.to(device)

encoder_shadow = EncoderResnet(256).eval()  # eval mode (batchnorm uses moving mean/variance)
decoder_shadow = DecoderResnet(256, 512, len(vocab_f8k), 1)
encoder_shadow = encoder_shadow.to(device)
decoder_shadow = decoder_shadow.to(device)

# Build MFE models
image_encoder_com = ImageModel(256).eval()
text_encoder_com = TextModel(len(vocab_MFE),256).eval()
image_encoder_com = image_encoder_com.to(device)
text_encoder_com = text_encoder_com.to(device)


# Load the trained model parameters
# f8k-resnet
encoder_target.load_state_dict(torch.load('models/encoder-resnet-f8k-target.ckpt',  map_location=torch.device('cpu')))
decoder_target.load_state_dict(torch.load('models/decoder-resnet-f8k-target.ckpt',  map_location=torch.device('cpu')))
encoder_shadow.load_state_dict(torch.load('models/encoder-resnet-f8k-shadow.ckpt',  map_location=torch.device('cpu')))
decoder_shadow.load_state_dict(torch.load('models/decoder-resnet-f8k-shadow.ckpt',  map_location=torch.device('cpu')))

# MFE models
image_encoder_com.load_state_dict(torch.load('models/com_imgmodel-final.ckpt',  map_location=torch.device('cpu')))
text_encoder_com.load_state_dict(torch.load('models/com_textmodel-final.ckpt',  map_location=torch.device('cpu')))

# load all the captions
image_list = []
caption_list = []
with open('data/Flickr8k/f8k_captioning.txt', 'r') as f:
    lines = f.readlines()
    for line in lines:
        line_list = line.split(',', 1)
        if(len(line_list) == 2):
            image_list.append(line_list[0])
            caption_list.append(line_list[1].replace('\n', ''))

with open('data/coco_data/coco_captioning.txt', 'r') as f:
    lines = f.readlines()
    for line in lines:
        line_list = line.split(',', 1)
        if(len(line_list) == 2):
            image_list.append(line_list[0])
            caption_list.append(line_list[1].replace('\n', ''))

# load image path
f8k_path = 'data/Flickr8k/images/train'
f8k_image_path = os.listdir(f8k_path)
f8k_image_test = 'data/coco_data/resized2017/test'
f8k_image_test_path = os.listdir(f8k_image_test)
output_path = 'output.txt'

# randomly choose 1000 image from three dataset
# random.seed(10)
f8k_images = random.sample(f8k_image_path, 1000)
f8k_images_out = random.sample(f8k_image_test_path, 1000)
f8k_images_shadow = random.sample(f8k_image_path, 1000)

# image-captions
f8k_captions = {}
f8k_captions_out = {}
f8k_captions_shadow = {}

for image in f8k_images:
    f8k_captions[image] = []
    for i in range(len(image_list)):
        if image_list[i] == image:
            tokens = nltk.tokenize.word_tokenize(str(caption_list[i]).lower())
            caption = []
            caption.extend([token for token in tokens])
            f8k_captions[image].append(caption)

for image in f8k_images_out:
    f8k_captions_out[image] = []
    for i in range(len(image_list)):
        if image_list[i] == image:
            tokens = nltk.tokenize.word_tokenize(str(caption_list[i]).lower())
            caption = []
            caption.extend([token for token in tokens])
            f8k_captions_out[image].append(caption)

for image in f8k_images_shadow:
    f8k_captions_shadow[image] = []
    for i in range(len(image_list)):
        if image_list[i] == image:
            tokens = nltk.tokenize.word_tokenize(str(caption_list[i]).lower())
            caption = []
            caption.extend([token for token in tokens])
            f8k_captions_shadow[image].append(caption)

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
    caption_vec = torch.zeros(len(vocab_MFE))
    for cap in sampled_caption:
        caption_vec[vocab_MFE(cap)] += 1
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
        try:
            rouge_score = rouge.get_scores(hyps=cand, refs=ref)
            rouge1 += rouge_score[0]["rouge-1"]['f']
            rouge2 += rouge_score[0]["rouge-2"]['f']
            rougel += rouge_score[0]["rouge-l"]['f']
        except ValueError:
            return [rouge1, rouge2, rougel]
    rouge1 /= len(references)
    rouge2 /= len(references)
    rougel /= len(references)
    return [rouge1, rouge2, rougel]

def fbmi_attack_evaluation(target_encoder, target_decoder, shadow_encoder, shadow_decoder, target_dataset, target_dataset_path, target_captioning, target_vocab, shadow_dataset, shadow_dataset_path, shadow_captioning, shadow_vocab, nonmember_dataset, nonmember_dataset_path, nonmember_captioning, shadow_nonmember_dataset, shadow_nonmember_dataset_path, shadow_nonmember_captioning, imagemodel, textmodel, n):
    # train attack model
    shadow_similarity_data = []
    shadow_label = []
    for image in shadow_dataset:
        imagepath = shadow_dataset_path + image
        prob, captions = result(imagepath, shadow_encoder, shadow_decoder, shadow_vocab)
        captions_vec = onehot(captions)
        similarity = feature_similarity(imagepath, captions_vec, imagemodel, textmodel)
        shadow_similarity_data.append(similarity)
        shadow_label.append(1)
    for image in shadow_nonmember_dataset:
        imagepath = shadow_nonmember_dataset_path + image
        bleu_score = []
        prob, captions = result(imagepath, shadow_encoder, shadow_decoder, shadow_vocab)
        captions_vec = onehot(captions)
        similarity = feature_similarity(imagepath, captions_vec, imagemodel, textmodel)
        shadow_similarity_data.append(similarity)
        shadow_label.append(0)
    shadow_similarity_data = np.array(shadow_similarity_data)
    shadow_label = np.array(shadow_label)
    np.savetxt("results/fb_shadow_data{:d}.txt".format(n), shadow_similarity_data)
    np.savetxt("results/fb_shadow_label{:d}.txt".format(n), shadow_label)
    target_similarity_data = []
    target_label = []
    for image in target_dataset:
        imagepath = target_dataset_path + image
        bleu_score = []
        prob, captions = result(imagepath, target_encoder, target_decoder, target_vocab)
        captions_vec = onehot(captions)
        similarity = feature_similarity(imagepath, captions_vec, imagemodel, textmodel)
        target_similarity_data.append(similarity)
        target_label.append(1)
    for image in nonmember_dataset:
        imagepath = nonmember_dataset_path + image
        bleu_score = []
        prob, captions = result(imagepath, target_encoder, target_decoder, target_vocab)
        captions_vec = onehot(captions)
        similarity = feature_similarity(imagepath, captions_vec, imagemodel, textmodel)
        target_similarity_data.append(similarity)
        target_label.append(0)
    target_similarity_data = np.array(target_similarity_data)
    target_label = np.array(target_label)
    np.savetxt("results/fb_target_data{:d}.txt".format(n), target_similarity_data)
    np.savetxt("results/fb_target_label{:d}.txt".format(n), np.array(target_label))

# bleu score
# def mbmi_attack_evaluation(target_encoder, target_decoder, shadow_encoder, shadow_decoder, target_dataset, target_dataset_path, target_captioning, target_vocab, shadow_dataset, shadow_dataset_path, shadow_captioning, shadow_vocab, nonmember_dataset, nonmember_dataset_path, nonmember_captioning, n):
#     # train attack model
#     shadow_prob_data = []
#     shadow_bleu_data = []
#     shadow_label = []
#     for image in shadow_dataset:
#         imagepath = shadow_dataset_path + image
#         bleu_score = []
#         prob, captions = result(imagepath, shadow_encoder, shadow_decoder, shadow_vocab)
#         shadow_prob_data.append(prob)
#         bleu_score.append(sentence_bleu(shadow_captioning[image], captions, weights=(1, 0, 0, 0)))
#         bleu_score.append(sentence_bleu(shadow_captioning[image], captions, weights=(0.5, 0.5, 0, 0)))
#         bleu_score.append(sentence_bleu(shadow_captioning[image], captions, weights=(0.33, 0.33, 0.33, 0)))
#         shadow_bleu_data.append(bleu_score)
#         shadow_label.append(1)
#     for image in nonmember_dataset:
#         imagepath = nonmember_dataset_path + image
#         bleu_score = []
#         prob, captions = result(imagepath, shadow_encoder, shadow_decoder, shadow_vocab)
#         shadow_prob_data.append(prob)
#         bleu_score.append(sentence_bleu(nonmember_captioning[image], captions, weights=(1, 0, 0, 0)))
#         bleu_score.append(sentence_bleu(nonmember_captioning[image], captions, weights=(0.5, 0.5, 0, 0)))
#         bleu_score.append(sentence_bleu(nonmember_captioning[image], captions, weights=(0.33, 0.33, 0.33, 0)))
#         shadow_bleu_data.append(bleu_score)
#         shadow_label.append(-1)
#     # attack model svm model
#     attack_model = svm.SVC(kernel='linear', C=1e5)
#     attack_model.fit(shadow_bleu_data, shadow_label)
#     # target
#     target_prob_data = []
#     target_bleu_data = []
#     target_label = []
#     for image in target_dataset:
#         imagepath = target_dataset_path + image
#         bleu_score = []
#         prob, captions = result(imagepath, target_encoder, target_decoder, target_vocab)
#         target_prob_data.append(prob)
#         bleu_score.append(sentence_bleu(target_captioning[image], captions, weights=(1, 0, 0, 0)))
#         bleu_score.append(sentence_bleu(target_captioning[image], captions, weights=(0.5, 0.5, 0, 0)))
#         bleu_score.append(sentence_bleu(target_captioning[image], captions, weights=(0.33, 0.33, 0.33, 0)))
#         target_bleu_data.append(bleu_score)
#         target_label.append(1)
#     for image in nonmember_dataset:
#         imagepath = nonmember_dataset_path + image
#         bleu_score = []
#         prob, captions = result(imagepath, target_encoder, target_decoder, target_vocab)
#         target_prob_data.append(prob)
#         bleu_score.append(sentence_bleu(nonmember_captioning[image], captions, weights=(1, 0, 0, 0)))
#         bleu_score.append(sentence_bleu(nonmember_captioning[image], captions, weights=(0.5, 0.5, 0, 0)))
#         bleu_score.append(sentence_bleu(nonmember_captioning[image], captions, weights=(0.33, 0.33, 0.33, 0)))
#         target_bleu_data.append(bleu_score)
#         target_label.append(-1)
#     attack_label = attack_model.predict(target_bleu_data)
#     acs = accuracy_score(target_label, attack_label)
#     pcs = precision_score(target_label, attack_label, average = None)
#     rcs = recall_score(target_label, attack_label, average = None)
#     with open('output.txt','a') as f:
#         str = "accuracy_score: {:f}\n".format(acs)
#         str += "precision_score: {:f}\n".format(pcs[1])
#         str += "recall_score: {:f}\n".format(rcs[1])
#         f.write(str)
#     target_bleu_data = np.array(target_bleu_data)

# rouge score
def mbmi_attack_evaluation(target_encoder, target_decoder, shadow_encoder, shadow_decoder, target_dataset, target_dataset_path, target_captioning, target_vocab, shadow_dataset, shadow_dataset_path, shadow_captioning, shadow_vocab, nonmember_dataset, nonmember_dataset_path, nonmember_captioning, shadow_nonmember_dataset, shadow_nonmember_dataset_path, shadow_nonmember_captioning, n):
    # train attack model
    shadow_prob_data = []
    shadow_bleu_data = []
    shadow_label = []
    for image in shadow_dataset:
        imagepath = shadow_dataset_path + image
        bleu_score = []
        prob, captions = result(imagepath, shadow_encoder, shadow_decoder, shadow_vocab)
        shadow_prob_data.append(prob)
        bleu_score = get_rouge_score(shadow_captioning[image], captions)
        shadow_bleu_data.append(bleu_score)
        shadow_label.append(1)
    for image in shadow_nonmember_dataset:
        imagepath = shadow_nonmember_dataset_path + image
        bleu_score = []
        prob, captions = result(imagepath, shadow_encoder, shadow_decoder, shadow_vocab)
        shadow_prob_data.append(prob)
        bleu_score = get_rouge_score(shadow_nonmember_captioning[image], captions)
        shadow_bleu_data.append(bleu_score)
        shadow_label.append(-1)
    shadow_bleu_data = np.array(shadow_bleu_data)
    shadow_label = np.array(shadow_label)
    np.savetxt("results/rouge_shadow_data{:d}.txt".format(n), shadow_bleu_data)
    np.savetxt("results/rouge_shadow_label{:d}.txt".format(n), shadow_label)
    # attack model svm model
    attack_model = svm.SVC(kernel='linear', C=1e5)
    attack_model.fit(shadow_bleu_data, shadow_label)
    # target
    target_prob_data = []
    target_bleu_data = []
    target_label = []
    for image in target_dataset:
        imagepath = target_dataset_path + image
        bleu_score = []
        prob, captions = result(imagepath, target_encoder, target_decoder, target_vocab)
        target_prob_data.append(prob)
        bleu_score = get_rouge_score(target_captioning[image], captions)
        target_bleu_data.append(bleu_score)
        target_label.append(1)
    for image in nonmember_dataset:
        imagepath = nonmember_dataset_path + image
        bleu_score = []
        prob, captions = result(imagepath, target_encoder, target_decoder, target_vocab)
        target_prob_data.append(prob)
        bleu_score = get_rouge_score(nonmember_captioning[image], captions)
        target_bleu_data.append(bleu_score)
        target_label.append(-1)
    target_bleu_data = np.array(target_bleu_data)
    target_label = np.array(target_label)
    np.savetxt("results/rouge_target_data{:d}.txt".format(n), target_bleu_data)
    np.savetxt("results/rouge_target_label{:d}.txt".format(n), np.array(target_label))

mbmi_attack_evaluation(encoder_target, decoder_target, encoder_shadow, decoder_shadow, f8k_images, f8k_path, f8k_captions, vocab_f8k, f8k_images_shadow, f8k_path, f8k_captions_shadow, vocab_f8k, f8k_images_out, f8k_image_test, f8k_captions_out, f8k_images_out, f8k_image_test, f8k_captions_out, 1)
fbmi_attack_evaluation(encoder_target, decoder_target, encoder_shadow, decoder_shadow, f8k_images, f8k_path, f8k_captions, vocab_f8k, f8k_images_shadow, f8k_path, f8k_captions_shadow, vocab_f8k, f8k_images_out, f8k_image_test, f8k_captions_out, f8k_images_out, f8k_image_test, f8k_captions_out, image_encoder_if, text_encoder_if, 1)
