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
f8k_path = 'Flickr8k/images/train'
f8k_image_path = os.listdir(f8k_path)
coco_path = 'coco_data/resized2017/train'
coco_image_path = os.listdir(coco_path)
iapr_path = 'ipar_data/resizedIapr/train'
iapr_image_path = os.listdir(iapr_path)
f8k_image_test = 'coco_data/resized2017/test'
f8k_image_test_path = os.listdir(f8k_image_test)
coco_image_test = 'ipar_data/resizedIapr/test'
coco_image_test_path = os.listdir(coco_image_test)
iapr_image_test = 'Flickr8k/images/test'
iapr_image_test_path = os.listdir(iapr_image_test)
output_path = 'output.txt'

# randomly choose 1000 image from three dataset
# random.seed(10)
f8k_images = random.sample(f8k_image_path, 1000)
f8k_images_out = random.sample(f8k_image_test_path, 1000)
f8k_images_shadow = random.sample(f8k_image_path, 1000)
coco_images = random.sample(coco_image_path, 1000)
coco_images_out = random.sample(coco_image_test_path, 1000)
coco_images_shadow = random.sample(coco_image_path, 1000)
iapr_images = random.sample(iapr_image_path, 1000)
iapr_images_out = random.sample(iapr_image_test_path, 1000)
iapr_images_shadow = random.sample(iapr_image_path, 1000)

# image-captions
f8k_captions = {}
f8k_captions_out = {}
f8k_captions_shadow = {}
coco_captions = {}
coco_captions_out = {}
coco_captions_shadow = {}
iapr_captions = {}
iapr_captions_out = {}
iapr_captions_shadow = {}

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

for image in coco_images:
    coco_captions[image] = []
    for i in range(len(image_list)):
        if image_list[i] == image:
            tokens = nltk.tokenize.word_tokenize(str(caption_list[i]).lower())
            caption = []
            caption.extend([token for token in tokens])
            coco_captions[image].append(caption)

for image in coco_images_out:
    coco_captions_out[image] = []
    for i in range(len(image_list)):
        if image_list[i] == image:
            tokens = nltk.tokenize.word_tokenize(str(caption_list[i]).lower())
            caption = []
            caption.extend([token for token in tokens])
            coco_captions_out[image].append(caption)

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

for image in iapr_images_out:
    iapr_captions_out[image] = []
    for i in range(len(image_list)):
        if image_list[i] == image:
            tokens = nltk.tokenize.word_tokenize(str(caption_list[i]).lower())
            caption = []
            caption.extend([token for token in tokens])
            iapr_captions_out[image].append(caption)

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

# def get_rouge_score(references, candidate):
#     rouge1 = 0.0
#     rouge2 = 0.0
#     rougel = 0.0
#     cand = ' '.join(candidate)
#     rouge = Rouge()
#     for reference in references:
#         ref = ' '.join(reference)
#         rouge_score = rouge.get_scores(hyps=cand, refs=ref)
#         rouge1 += rouge_score[0]["rouge-1"]['f']
#         rouge2 += rouge_score[0]["rouge-2"]['f']
#         rougel += rouge_score[0]["rouge-l"]['f']
#     rouge1 /= len(references)
#     rouge2 /= len(references)
#     rougel /= len(references)
#     return [rouge1, rouge2, rougel]

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

def fbmi_attacl_evaluation(target_encoder, target_decoder, shadow_encoder, shadow_decoder, target_dataset, target_dataset_path, target_captioning, target_vocab, shadow_dataset, shadow_dataset_path, shadow_captioning, shadow_vocab, nonmember_dataset, nonmember_dataset_path, nonmember_captioning, shadow_nonmember_dataset, shadow_nonmember_dataset_path, shadow_nonmember_captioning, imagemodel, textmodel, n):
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
# def mi_attack_evaluation(target_encoder, target_decoder, shadow_encoder, shadow_decoder, target_dataset, target_dataset_path, target_captioning, target_vocab, shadow_dataset, shadow_dataset_path, shadow_captioning, shadow_vocab, nonmember_dataset, nonmember_dataset_path, nonmember_captioning, n):
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
#     # draw a picture
#     in_data = target_bleu_data[:1000, :]
#     x1 = in_data[:, 0]
#     y1 = in_data[:, 2] 
#     z1 = in_data[:, 1]
#     out_data = target_bleu_data[1000:, :]
#     x2 = in_data[:, 0]
#     y2= in_data[:, 2] 
#     z2 = in_data[:, 1]
#     fig = plt.figure()
#     ax = Axes3D(fig)
#     ax.scatter(x1, y1, z1, c='b', label='in')
#     ax.scatter(x2, y2, z2, c='r', label='out')
#     ax.legend(loc='best')
#     ax.set_zlabel('2-gram', fontdict={'size': 15, 'color': 'red'})
#     ax.set_ylabel('3-gram', fontdict={'size': 15, 'color': 'red'})
#     ax.set_xlabel('1-gram', fontdict={'size': 15, 'color': 'red'})
#     plt.savefig('./result{:d}.jpg'.format(n))

auc_list = []

# rouge score
def mi_attack_evaluation(target_encoder, target_decoder, shadow_encoder, shadow_decoder, target_dataset, target_dataset_path, target_captioning, target_vocab, shadow_dataset, shadow_dataset_path, shadow_captioning, shadow_vocab, nonmember_dataset, nonmember_dataset_path, nonmember_captioning, shadow_nonmember_dataset, shadow_nonmember_dataset_path, shadow_nonmember_captioning, n):
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
    # in_data = target_bleu_data[:1000, :]
    # x1 = in_data[:, 0]
    # y1 = in_data[:, 1] 
    # z1 = in_data[:, 2]
    # out_data = target_bleu_data[1000:, :]
    # x2 = out_data[:, 0]
    # y2 = out_data[:, 1] 
    # z2 = out_data[:, 2]
    # fig = plt.figure()
    # ax = Axes3D(fig)
    # ax.scatter(x1, y1, z1, c='b', label='in')
    # ax.scatter(x2, y2, z2, c='r', label='out')
    # ax.legend(loc='best')
    # ax.set_zlabel('rouge-l', fontdict={'size': 15, 'color': 'red'})
    # ax.set_ylabel('rouge-2', fontdict={'size': 15, 'color': 'red'})
    # ax.set_xlabel('rouge-1', fontdict={'size': 15, 'color': 'red'})
    # plt.savefig('./results/{:d}.jpg'.format(n))


# target_encoder, target_decoder, shadow_encoder, shadow_decoder, target_dataset, target_dataset_path, target_captioning, target_vocab, shadow_dataset, shadow_dataset_path, shadow_captioning, shadow_vocab, nonmember_dataset, nonmember_dataset_path, nonmember_captioning, n
#1
with open(output_path,'a') as f:
    str = "#FRFR\n"
    f.write(str)
mi_attack_evaluation(encoder_fr, decoder_fr, encoder_fr, decoder_fr, f8k_images, f8k_path, f8k_captions, vocab_f8k, f8k_images_shadow, f8k_path, f8k_captions_shadow, vocab_f8k, f8k_images_out, f8k_image_test, f8k_captions_out, f8k_images_out, f8k_image_test, f8k_captions_out, 1)
fbmi_attacl_evaluation(encoder_fr, decoder_fr, encoder_fr, decoder_fr, f8k_images, f8k_path, f8k_captions, vocab_f8k, f8k_images_shadow, f8k_path, f8k_captions_shadow, vocab_f8k, f8k_images_out, f8k_image_test, f8k_captions_out, f8k_images_out, f8k_image_test, f8k_captions_out, image_encoder_if, text_encoder_if, 1)
print(auc_list)
with open(output_path,'a') as f:
    str = "#FRFV\n"
    f.write(str)
mi_attack_evaluation(encoder_fr, decoder_fr, encoder_fv, decoder_fv, f8k_images, f8k_path, f8k_captions, vocab_f8k, f8k_images_shadow, f8k_path, f8k_captions_shadow, vocab_f8k, f8k_images_out, f8k_image_test, f8k_captions_out, f8k_images_out, f8k_image_test, f8k_captions_out, 2)
fbmi_attacl_evaluation(encoder_fr, decoder_fr, encoder_fv, decoder_fv, f8k_images, f8k_path, f8k_captions, vocab_f8k, f8k_images_shadow, f8k_path, f8k_captions_shadow, vocab_f8k, f8k_images_out, f8k_image_test, f8k_captions_out, f8k_images_out, f8k_image_test, f8k_captions_out, image_encoder_if, text_encoder_if, 2)
with open(output_path,'a') as f:
    str = "#FRCR\n"
    f.write(str)
mi_attack_evaluation(encoder_fr, decoder_fr, encoder_cr, decoder_cr, f8k_images, f8k_path, f8k_captions, vocab_f8k, coco_images, coco_path, coco_captions, vocab_coco, f8k_images_out, f8k_image_test, f8k_captions_out, coco_images_out, coco_image_test, coco_captions_out, 3)
fbmi_attacl_evaluation(encoder_fr, decoder_fr, encoder_cr, decoder_cr, f8k_images, f8k_path, f8k_captions, vocab_f8k, coco_images, coco_path, coco_captions, vocab_coco, f8k_images_out, f8k_image_test, f8k_captions_out, coco_images_out, coco_image_test, coco_captions_out, image_encoder_ci, text_encoder_ci, 3)
with open(output_path,'a') as f:
    str = "#FRCV\n"
    f.write(str)
mi_attack_evaluation(encoder_fr, decoder_fr, encoder_cv, decoder_cv, f8k_images, f8k_path, f8k_captions, vocab_f8k, coco_images, coco_path, coco_captions, vocab_coco, f8k_images_out, f8k_image_test, f8k_captions_out, coco_images_out, coco_image_test, coco_captions_out, 4)
fbmi_attacl_evaluation(encoder_fr, decoder_fr, encoder_cv, decoder_cv, f8k_images, f8k_path, f8k_captions, vocab_f8k, coco_images, coco_path, coco_captions, vocab_coco, f8k_images_out, f8k_image_test, f8k_captions_out, coco_images_out, coco_image_test, coco_captions_out, image_encoder_ci, text_encoder_ci, 4)
with open(output_path,'a') as f:
    str = "#FRIR\n"
    f.write(str)
mi_attack_evaluation(encoder_fr, decoder_fr, encoder_ir, decoder_ir, f8k_images, f8k_path, f8k_captions, vocab_f8k, iapr_images, iapr_path, iapr_captions, vocab_iapr, f8k_images_out, f8k_image_test, f8k_captions_out, iapr_images_out, iapr_image_test, iapr_captions_out, 5)
fbmi_attacl_evaluation(encoder_fr, decoder_fr, encoder_ir, decoder_ir, f8k_images, f8k_path, f8k_captions, vocab_f8k, iapr_images, iapr_path, iapr_captions, vocab_iapr, f8k_images_out, f8k_image_test, f8k_captions_out, iapr_images_out, iapr_image_test, iapr_captions_out, image_encoder_ci, text_encoder_ci, 5)
with open(output_path,'a') as f:
    str = "#FRIV\n"
    f.write(str)
mi_attack_evaluation(encoder_fr, decoder_fr, encoder_iv, decoder_iv, f8k_images, f8k_path, f8k_captions, vocab_f8k, iapr_images, iapr_path, iapr_captions, vocab_iapr, f8k_images_out, f8k_image_test, f8k_captions_out, iapr_images_out, iapr_image_test, iapr_captions_out, 6)
fbmi_attacl_evaluation(encoder_fr, decoder_fr, encoder_iv, decoder_iv, f8k_images, f8k_path, f8k_captions, vocab_f8k, iapr_images, iapr_path, iapr_captions, vocab_iapr, f8k_images_out, f8k_image_test, f8k_captions_out, iapr_images_out, iapr_image_test, iapr_captions_out, image_encoder_ci, text_encoder_ci, 6)
#2
with open(output_path,'a') as f:
    str = "#FVFR\n"
    f.write(str)
mi_attack_evaluation(encoder_fv, decoder_fv, encoder_fr, decoder_fr, f8k_images, f8k_path, f8k_captions, vocab_f8k, f8k_images_shadow, f8k_path, f8k_captions_shadow, vocab_f8k, f8k_images_out, f8k_image_test, f8k_captions_out, f8k_images_out, f8k_image_test, f8k_captions_out, 7)
fbmi_attacl_evaluation(encoder_fv, decoder_fv, encoder_fr, decoder_fr, f8k_images, f8k_path, f8k_captions, vocab_f8k, f8k_images_shadow, f8k_path, f8k_captions_shadow, vocab_f8k, f8k_images_out, f8k_image_test, f8k_captions_out, f8k_images_out, f8k_image_test, f8k_captions_out, image_encoder_if, text_encoder_if, 7)
with open(output_path,'a') as f:
    str = "#FVFV\n"
    f.write(str)
mi_attack_evaluation(encoder_fv, decoder_fv, encoder_fv, decoder_fv, f8k_images, f8k_path, f8k_captions, vocab_f8k, f8k_images_shadow, f8k_path, f8k_captions_shadow, vocab_f8k, f8k_images_out, f8k_image_test, f8k_captions_out, f8k_images_out, f8k_image_test, f8k_captions_out, 8)
fbmi_attacl_evaluation(encoder_fv, decoder_fv, encoder_fv, decoder_fv, f8k_images, f8k_path, f8k_captions, vocab_f8k, f8k_images_shadow, f8k_path, f8k_captions_shadow, vocab_f8k, f8k_images_out, f8k_image_test, f8k_captions_out, f8k_images_out, f8k_image_test, f8k_captions_out, image_encoder_if, text_encoder_if, 8)
with open(output_path,'a') as f:
    str = "#FVCR\n"
    f.write(str)
mi_attack_evaluation(encoder_fv, decoder_fv, encoder_cr, decoder_cr, f8k_images, f8k_path, f8k_captions, vocab_f8k, coco_images, coco_path, coco_captions, vocab_coco, f8k_images_out, f8k_image_test, f8k_captions_out, coco_images_out, coco_image_test, coco_captions_out, 9)
fbmi_attacl_evaluation(encoder_fv, decoder_fv, encoder_cr, decoder_cr, f8k_images, f8k_path, f8k_captions, vocab_f8k, coco_images, coco_path, coco_captions, vocab_coco, f8k_images_out, f8k_image_test, f8k_captions_out, coco_images_out, coco_image_test, coco_captions_out, image_encoder_ci, text_encoder_ci, 9)
with open(output_path,'a') as f:
    str = "#FVCV\n"
    f.write(str)
mi_attack_evaluation(encoder_fv, decoder_fv, encoder_cv, decoder_cv, f8k_images, f8k_path, f8k_captions, vocab_f8k, coco_images, coco_path, coco_captions, vocab_coco, f8k_images_out, f8k_image_test, f8k_captions_out, coco_images_out, coco_image_test, coco_captions_out, 10)
fbmi_attacl_evaluation(encoder_fv, decoder_fv, encoder_cv, decoder_cv, f8k_images, f8k_path, f8k_captions, vocab_f8k, coco_images, coco_path, coco_captions, vocab_coco, f8k_images_out, f8k_image_test, f8k_captions_out, coco_images_out, coco_image_test, coco_captions_out, image_encoder_ci, text_encoder_ci, 10)
with open(output_path,'a') as f:
    str = "#FVIR\n"
    f.write(str)
mi_attack_evaluation(encoder_fv, decoder_fv, encoder_ir, decoder_ir, f8k_images, f8k_path, f8k_captions, vocab_f8k, iapr_images, iapr_path, iapr_captions, vocab_iapr, f8k_images_out, f8k_image_test, f8k_captions_out, iapr_images_out, iapr_image_test, iapr_captions_out, 11)
fbmi_attacl_evaluation(encoder_fv, decoder_fv, encoder_ir, decoder_ir, f8k_images, f8k_path, f8k_captions, vocab_f8k, iapr_images, iapr_path, iapr_captions, vocab_iapr, f8k_images_out, f8k_image_test, f8k_captions_out, iapr_images_out, iapr_image_test, iapr_captions_out, image_encoder_ci, text_encoder_ci, 11)
with open(output_path,'a') as f:
    str = "#FVIV\n"
    f.write(str)
mi_attack_evaluation(encoder_fv, decoder_fv, encoder_iv, decoder_iv, f8k_images, f8k_path, f8k_captions, vocab_f8k, iapr_images, iapr_path, iapr_captions, vocab_iapr, f8k_images_out, f8k_image_test, f8k_captions_out, iapr_images_out, iapr_image_test, iapr_captions_out, 12)
fbmi_attacl_evaluation(encoder_fv, decoder_fv, encoder_iv, decoder_iv, f8k_images, f8k_path, f8k_captions, vocab_f8k, iapr_images, iapr_path, iapr_captions, vocab_iapr, f8k_images_out, f8k_image_test, f8k_captions_out, iapr_images_out, iapr_image_test, iapr_captions_out, image_encoder_ci, text_encoder_ci, 12)
#3
with open(output_path,'a') as f:
    str = "#CRFR\n"
    f.write(str)
mi_attack_evaluation(encoder_cr, decoder_cr, encoder_fr, decoder_fr, coco_images, coco_path, coco_captions, vocab_coco, f8k_images, f8k_path, f8k_captions, vocab_f8k, coco_images_out, coco_image_test, coco_captions_out, f8k_images_out, f8k_image_test, f8k_captions_out, 13)
fbmi_attacl_evaluation(encoder_cr, decoder_cr, encoder_fr, decoder_fr, coco_images, coco_path, coco_captions, vocab_coco, f8k_images, f8k_path, f8k_captions, vocab_f8k, coco_images_out, coco_image_test, coco_captions_out, f8k_images_out, f8k_image_test, f8k_captions_out, image_encoder_if, text_encoder_if, 13)
with open(output_path,'a') as f:
    str = "#CRFV\n"
    f.write(str)
mi_attack_evaluation(encoder_cr, decoder_cr, encoder_fv, decoder_fv, coco_images, coco_path, coco_captions, vocab_coco, f8k_images, f8k_path, f8k_captions, vocab_f8k, coco_images_out, coco_image_test, coco_captions_out, f8k_images_out, f8k_image_test, f8k_captions_out, 14)
fbmi_attacl_evaluation(encoder_cr, decoder_cr, encoder_fv, decoder_fv, coco_images, coco_path, coco_captions, vocab_coco, f8k_images, f8k_path, f8k_captions, vocab_f8k, coco_images_out, coco_image_test, coco_captions_out, f8k_images_out, f8k_image_test, f8k_captions_out, image_encoder_if, text_encoder_if, 14)
with open(output_path,'a') as f:
    str = "#CRCR\n"
    f.write(str)
mi_attack_evaluation(encoder_cr, decoder_cr, encoder_cr, decoder_cr, coco_images, coco_path, coco_captions, vocab_coco, coco_images_shadow, coco_path, coco_captions_shadow, vocab_coco, coco_images_out, coco_image_test, coco_captions_out, coco_images_out, coco_image_test, coco_captions_out, 15)
fbmi_attacl_evaluation(encoder_cr, decoder_cr, encoder_cr, decoder_cr, coco_images, coco_path, coco_captions, vocab_coco, coco_images_shadow, coco_path, coco_captions_shadow, vocab_coco, coco_images_out, coco_image_test, coco_captions_out, coco_images_out, coco_image_test, coco_captions_out, image_encoder_ci, text_encoder_ci, 15)
with open(output_path,'a') as f:
    str = "#CRCV\n"
    f.write(str)
mi_attack_evaluation(encoder_cr, decoder_cr, encoder_cv, decoder_cv, coco_images, coco_path, coco_captions, vocab_coco, coco_images_shadow, coco_path, coco_captions_shadow, vocab_coco, coco_images_out, coco_image_test, coco_captions_out, coco_images_out, coco_image_test, coco_captions_out, 16)
fbmi_attacl_evaluation(encoder_cr, decoder_cr, encoder_cv, decoder_cv, coco_images, coco_path, coco_captions, vocab_coco, coco_images_shadow, coco_path, coco_captions_shadow, vocab_coco, coco_images_out, coco_image_test, coco_captions_out, coco_images_out, coco_image_test, coco_captions_out, image_encoder_ci, text_encoder_ci, 16)
with open(output_path,'a') as f:
    str = "#CRIR\n"
    f.write(str)
mi_attack_evaluation(encoder_cr, decoder_cr, encoder_ir, decoder_ir, coco_images, coco_path, coco_captions, vocab_coco, iapr_images, iapr_path, iapr_captions, vocab_iapr, coco_images_out, coco_image_test, coco_captions_out, iapr_images_out, iapr_image_test, iapr_captions_out, 17)
fbmi_attacl_evaluation(encoder_cr, decoder_cr, encoder_ir, decoder_ir, coco_images, coco_path, coco_captions, vocab_coco, iapr_images, iapr_path, iapr_captions, vocab_iapr, coco_images_out, coco_image_test, coco_captions_out, iapr_images_out, iapr_image_test, iapr_captions_out, image_encoder_if, text_encoder_if, 17)
with open(output_path,'a') as f:
    str = "#CRIV\n"
    f.write(str)
mi_attack_evaluation(encoder_cr, decoder_cr, encoder_iv, decoder_iv, coco_images, coco_path, coco_captions, vocab_coco, iapr_images, iapr_path, iapr_captions, vocab_iapr, coco_images_out, coco_image_test, coco_captions_out, iapr_images_out, iapr_image_test, iapr_captions_out, 18)
fbmi_attacl_evaluation(encoder_cr, decoder_cr, encoder_iv, decoder_iv, coco_images, coco_path, coco_captions, vocab_coco, iapr_images, iapr_path, iapr_captions, vocab_iapr, coco_images_out, coco_image_test, coco_captions_out, iapr_images_out, iapr_image_test, iapr_captions_out, image_encoder_if, text_encoder_if, 18)
#4
with open(output_path,'a') as f:
    str = "#CVFR\n"
    f.write(str)
mi_attack_evaluation(encoder_cv, decoder_cv, encoder_fr, decoder_fr, coco_images, coco_path, coco_captions, vocab_coco, f8k_images, f8k_path, f8k_captions, vocab_f8k, coco_images_out, coco_image_test, coco_captions_out, f8k_images_out, f8k_image_test, f8k_captions_out, 19)
fbmi_attacl_evaluation(encoder_cv, decoder_cv, encoder_fr, decoder_fr, coco_images, coco_path, coco_captions, vocab_coco, f8k_images, f8k_path, f8k_captions, vocab_f8k, coco_images_out, coco_image_test, coco_captions_out, f8k_images_out, f8k_image_test, f8k_captions_out, image_encoder_if, text_encoder_if, 19)
with open(output_path,'a') as f:
    str = "#CVFV\n"
    f.write(str)
mi_attack_evaluation(encoder_cv, decoder_cv, encoder_fv, decoder_fv, coco_images, coco_path, coco_captions, vocab_coco, f8k_images, f8k_path, f8k_captions, vocab_f8k, coco_images_out, coco_image_test, coco_captions_out, f8k_images_out, f8k_image_test, f8k_captions_out, 20)
fbmi_attacl_evaluation(encoder_cv, decoder_cv, encoder_fv, decoder_fv, coco_images, coco_path, coco_captions, vocab_coco, f8k_images, f8k_path, f8k_captions, vocab_f8k, coco_images_out, coco_image_test, coco_captions_out, f8k_images_out, f8k_image_test, f8k_captions_out, image_encoder_if, text_encoder_if, 20)
with open(output_path,'a') as f:
    str = "#CVCR\n"
    f.write(str)
mi_attack_evaluation(encoder_cv, decoder_cv, encoder_cr, decoder_cr, coco_images, coco_path, coco_captions, vocab_coco, coco_images_shadow, coco_path, coco_captions_shadow, vocab_coco, coco_images_out, coco_image_test, coco_captions_out, coco_images_out, coco_image_test, coco_captions_out, 21)
fbmi_attacl_evaluation(encoder_cv, decoder_cv, encoder_cr, decoder_cr, coco_images, coco_path, coco_captions, vocab_coco, coco_images_shadow, coco_path, coco_captions_shadow, vocab_coco, coco_images_out, coco_image_test, coco_captions_out, coco_images_out, coco_image_test, coco_captions_out, image_encoder_ci, text_encoder_ci, 21)
with open(output_path,'a') as f:
    str = "#CVCV\n"
    f.write(str)
mi_attack_evaluation(encoder_cv, decoder_cv, encoder_cv, decoder_cv, coco_images, coco_path, coco_captions, vocab_coco, coco_images_shadow, coco_path, coco_captions_shadow, vocab_coco, coco_images_out, coco_image_test, coco_captions_out, coco_images_out, coco_image_test, coco_captions_out, 22)
fbmi_attacl_evaluation(encoder_cv, decoder_cv, encoder_cv, decoder_cv, coco_images, coco_path, coco_captions, vocab_coco, coco_images_shadow, coco_path, coco_captions_shadow, vocab_coco, coco_images_out, coco_image_test, coco_captions_out, coco_images_out, coco_image_test, coco_captions_out, image_encoder_ci, text_encoder_ci, 22)
with open(output_path,'a') as f:
    str = "#CVIR\n"
    f.write(str)
mi_attack_evaluation(encoder_cv, decoder_cv, encoder_ir, decoder_ir, coco_images, coco_path, coco_captions, vocab_coco, iapr_images, iapr_path, iapr_captions, vocab_iapr, coco_images_out, coco_image_test, coco_captions_out, iapr_images_out, iapr_image_test, iapr_captions_out, 23)
fbmi_attacl_evaluation(encoder_cv, decoder_cv, encoder_ir, decoder_ir, coco_images, coco_path, coco_captions, vocab_coco, iapr_images, iapr_path, iapr_captions, vocab_iapr, coco_images_out, coco_image_test, coco_captions_out, iapr_images_out, iapr_image_test, iapr_captions_out, image_encoder_if, text_encoder_if, 23)
with open(output_path,'a') as f:
    str = "#CVIV\n"
    f.write(str)
mi_attack_evaluation(encoder_cv, decoder_cv, encoder_iv, decoder_iv, coco_images, coco_path, coco_captions, vocab_coco, iapr_images, iapr_path, iapr_captions, vocab_iapr, coco_images_out, coco_image_test, coco_captions_out, iapr_images_out, iapr_image_test, iapr_captions_out, 24)
fbmi_attacl_evaluation(encoder_cv, decoder_cv, encoder_iv, decoder_iv, coco_images, coco_path, coco_captions, vocab_coco, iapr_images, iapr_path, iapr_captions, vocab_iapr, coco_images_out, coco_image_test, coco_captions_out, iapr_images_out, iapr_image_test, iapr_captions_out, image_encoder_if, text_encoder_if, 24)
#5
with open(output_path,'a') as f:
    str = "#IRFR\n"
    f.write(str)
mi_attack_evaluation(encoder_ir, decoder_ir, encoder_fr, decoder_fr, iapr_images, iapr_path, iapr_captions, vocab_iapr, f8k_images, f8k_path, f8k_captions, vocab_f8k, iapr_images_out, iapr_image_test, iapr_captions_out, f8k_images_out, f8k_image_test, f8k_captions_out, 25)
fbmi_attacl_evaluation(encoder_ir, decoder_ir, encoder_fr, decoder_fr, iapr_images, iapr_path, iapr_captions, vocab_iapr, f8k_images, f8k_path, f8k_captions, vocab_f8k, iapr_images_out, iapr_image_test, iapr_captions_out, f8k_images_out, f8k_image_test, f8k_captions_out, image_encoder_cf, text_encoder_cf, 25)
with open(output_path,'a') as f:
    str = "#IRFV\n"
    f.write(str)
mi_attack_evaluation(encoder_ir, decoder_ir, encoder_fv, decoder_fv, iapr_images, iapr_path, iapr_captions, vocab_iapr, f8k_images, f8k_path, f8k_captions, vocab_f8k, iapr_images_out, iapr_image_test, iapr_captions_out, f8k_images_out, f8k_image_test, f8k_captions_out, 26)
fbmi_attacl_evaluation(encoder_ir, decoder_ir, encoder_fv, decoder_fv, iapr_images, iapr_path, iapr_captions, vocab_iapr, f8k_images, f8k_path, f8k_captions, vocab_f8k, iapr_images_out, iapr_image_test, iapr_captions_out, f8k_images_out, f8k_image_test, f8k_captions_out, image_encoder_cf, text_encoder_cf, 26)
with open(output_path,'a') as f:
    str = "#IRCR\n"
    f.write(str)
mi_attack_evaluation(encoder_ir, decoder_ir, encoder_cr, decoder_cr, iapr_images, iapr_path, iapr_captions, vocab_iapr, coco_images, coco_path, coco_captions, vocab_coco, iapr_images_out, iapr_image_test, iapr_captions_out, coco_images_out, coco_image_test, coco_captions_out, 27)
fbmi_attacl_evaluation(encoder_ir, decoder_ir, encoder_cr, decoder_cr, iapr_images, iapr_path, iapr_captions, vocab_iapr, coco_images, coco_path, coco_captions, vocab_coco, iapr_images_out, iapr_image_test, iapr_captions_out, coco_images_out, coco_image_test, coco_captions_out, image_encoder_cf, text_encoder_cf, 27)
with open(output_path,'a') as f:
    str = "#IRCV\n"
    f.write(str)
mi_attack_evaluation(encoder_ir, decoder_ir, encoder_cv, decoder_cv, iapr_images, iapr_path, iapr_captions, vocab_iapr, coco_images, coco_path, coco_captions, vocab_coco, iapr_images_out, iapr_image_test, iapr_captions_out, coco_images_out, coco_image_test, coco_captions_out, 28)
fbmi_attacl_evaluation(encoder_ir, decoder_ir, encoder_cv, decoder_cv, iapr_images, iapr_path, iapr_captions, vocab_iapr, coco_images, coco_path, coco_captions, vocab_coco, iapr_images_out, iapr_image_test, iapr_captions_out, coco_images_out, coco_image_test, coco_captions_out, image_encoder_cf, text_encoder_cf, 28)
with open(output_path,'a') as f:
    str = "#IRIR\n"
    f.write(str)
mi_attack_evaluation(encoder_ir, decoder_ir, encoder_ir, decoder_ir, iapr_images, iapr_path, iapr_captions, vocab_iapr, iapr_images_shadow, iapr_path, iapr_captions_shadow, vocab_iapr, iapr_images_out, iapr_image_test, iapr_captions_out, iapr_images_out, iapr_image_test, iapr_captions_out, 29)
fbmi_attacl_evaluation(encoder_ir, decoder_ir, encoder_ir, decoder_ir, iapr_images, iapr_path, iapr_captions, vocab_iapr, iapr_images_shadow, iapr_path, iapr_captions_shadow, vocab_iapr, iapr_images_out, iapr_image_test, iapr_captions_out, iapr_images_out, iapr_image_test, iapr_captions_out, image_encoder_if, text_encoder_if, 29)
with open(output_path,'a') as f:
    str = "#IRIV\n"
    f.write(str)
mi_attack_evaluation(encoder_ir, decoder_ir, encoder_iv, decoder_iv, iapr_images, iapr_path, iapr_captions, vocab_iapr, iapr_images_shadow, iapr_path, iapr_captions_shadow, vocab_iapr, iapr_images_out, iapr_image_test, iapr_captions_out, iapr_images_out, iapr_image_test, iapr_captions_out, 30)
fbmi_attacl_evaluation(encoder_ir, decoder_ir, encoder_iv, decoder_iv, iapr_images, iapr_path, iapr_captions, vocab_iapr, iapr_images_shadow, iapr_path, iapr_captions_shadow, vocab_iapr, iapr_images_out, iapr_image_test, iapr_captions_out, iapr_images_out, iapr_image_test, iapr_captions_out, image_encoder_if, text_encoder_if, 30)
#6
with open(output_path,'a') as f:
    str = "#IVFR\n"
    f.write(str)
mi_attack_evaluation(encoder_iv, decoder_iv, encoder_fr, decoder_fr, iapr_images, iapr_path, iapr_captions, vocab_iapr, f8k_images, f8k_path, f8k_captions, vocab_f8k, iapr_images_out, iapr_image_test, iapr_captions_out, f8k_images_out, f8k_image_test, f8k_captions_out, 31)
fbmi_attacl_evaluation(encoder_iv, decoder_iv, encoder_fr, decoder_fr, iapr_images, iapr_path, iapr_captions, vocab_iapr, f8k_images, f8k_path, f8k_captions, vocab_f8k, iapr_images_out, iapr_image_test, iapr_captions_out, f8k_images_out, f8k_image_test, f8k_captions_out, image_encoder_cf, text_encoder_cf, 31)
with open(output_path,'a') as f:
    str = "#IVFV\n"
    f.write(str)
mi_attack_evaluation(encoder_iv, decoder_iv, encoder_fv, decoder_fv, iapr_images, iapr_path, iapr_captions, vocab_iapr, f8k_images, f8k_path, f8k_captions, vocab_f8k, iapr_images_out, iapr_image_test, iapr_captions_out, f8k_images_out, f8k_image_test, f8k_captions_out, 32)
fbmi_attacl_evaluation(encoder_iv, decoder_iv, encoder_fv, decoder_fv, iapr_images, iapr_path, iapr_captions, vocab_iapr, f8k_images, f8k_path, f8k_captions, vocab_f8k, iapr_images_out, iapr_image_test, iapr_captions_out, f8k_images_out, f8k_image_test, f8k_captions_out, image_encoder_cf, text_encoder_cf, 32)
with open(output_path,'a') as f:
    str = "#IVCR\n"
    f.write(str)
mi_attack_evaluation(encoder_iv, decoder_iv, encoder_cr, decoder_cr, iapr_images, iapr_path, iapr_captions, vocab_iapr, coco_images, coco_path, coco_captions, vocab_coco, iapr_images_out, iapr_image_test, iapr_captions_out, coco_images_out, coco_image_test, coco_captions_out, 33)
fbmi_attacl_evaluation(encoder_iv, decoder_iv, encoder_cr, decoder_cr, iapr_images, iapr_path, iapr_captions, vocab_iapr, coco_images, coco_path, coco_captions, vocab_coco, iapr_images_out, iapr_image_test, iapr_captions_out, coco_images_out, coco_image_test, coco_captions_out, image_encoder_cf, text_encoder_cf, 33)
with open(output_path,'a') as f:
    str = "#IVCV\n"
    f.write(str)
mi_attack_evaluation(encoder_iv, decoder_iv, encoder_cv, decoder_cv, iapr_images, iapr_path, iapr_captions, vocab_iapr, coco_images, coco_path, coco_captions, vocab_coco, iapr_images_out, iapr_image_test, iapr_captions_out, coco_images_out, coco_image_test, coco_captions_out, 34)
fbmi_attacl_evaluation(encoder_iv, decoder_iv, encoder_cv, decoder_cv, iapr_images, iapr_path, iapr_captions, vocab_iapr, coco_images, coco_path, coco_captions, vocab_coco, iapr_images_out, iapr_image_test, iapr_captions_out, coco_images_out, coco_image_test, coco_captions_out, image_encoder_cf, text_encoder_cf, 34)
with open(output_path,'a') as f:
    str = "#IVIR\n"
    f.write(str)
mi_attack_evaluation(encoder_iv, decoder_iv, encoder_ir, decoder_ir, iapr_images, iapr_path, iapr_captions, vocab_iapr, iapr_images_shadow, iapr_path, iapr_captions_shadow, vocab_iapr, iapr_images_out, iapr_image_test, iapr_captions_out, iapr_images_out, iapr_image_test, iapr_captions_out, 35)
fbmi_attacl_evaluation(encoder_iv, decoder_iv, encoder_ir, decoder_ir, iapr_images, iapr_path, iapr_captions, vocab_iapr, iapr_images_shadow, iapr_path, iapr_captions_shadow, vocab_iapr, iapr_images_out, iapr_image_test, iapr_captions_out, iapr_images_out, iapr_image_test, iapr_captions_out, image_encoder_if, text_encoder_if, 35)
with open(output_path,'a') as f:
    str = "#IVIV\n"
    f.write(str)
mi_attack_evaluation(encoder_iv, decoder_iv, encoder_iv, decoder_iv, iapr_images, iapr_path, iapr_captions, vocab_iapr, iapr_images_shadow, iapr_path, iapr_captions_shadow, vocab_iapr, iapr_images_out, iapr_image_test, iapr_captions_out, iapr_images_out, iapr_image_test, iapr_captions_out, 36)
fbmi_attacl_evaluation(encoder_iv, decoder_iv, encoder_iv, decoder_iv, iapr_images, iapr_path, iapr_captions, vocab_iapr, iapr_images_shadow, iapr_path, iapr_captions_shadow, vocab_iapr, iapr_images_out, iapr_image_test, iapr_captions_out, iapr_images_out, iapr_image_test, iapr_captions_out, image_encoder_if, text_encoder_if, 36)
