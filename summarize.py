# encoding=utf-8
import torch
import time
import argparse
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)-8s %(message)s')
logFormatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
rootLogger = logging.getLogger()
import random
import shutil
import os
from model.noisyChannel import ChannelModel
from model.sentence import SentenceEmbedding
from dataset.data import Dataset
import numpy as np
from utils import recursive_to_device, visualize_tensor, genSubset
from rouge import Rouge
from pyrouge.rouge import Rouge155
from train import rouge_atten_matrix
import copy
from tqdm import tqdm
from IPython import embed
import my_utils


def rouge_atten_matrix(doc, summ):
    doc_len = len(doc)
    summ_len = len(summ)
    temp_mat = np.zeros([doc_len, summ_len])
    for i in range(doc_len):
        for j in range(summ_len):
            temp_mat[i, j] = Rouge().get_scores(doc[i], summ[j])[0]['rouge-1']['f']
    return temp_mat


def evalLead3(args):
    data = Dataset(path=args.data_path)
    Rouge_list, Rouge155_list = [], []
    Rouge155_obj = Rouge155(stem=True, tmp='./tmp2')
    for batch_iter, valid_batch in tqdm(enumerate(data.gen_train_minibatch()), total=data.test_size):
        if not (batch_iter % 100 == 0):
            continue
        doc, sums, doc_len, sums_len = valid_batch
        selected_indexs = range(min(doc.size(0), 1))
        doc_matrix = doc.data.numpy()
        doc_len_arr = doc_len.data.numpy()
        golden_summ_matrix = sums[0].data.numpy()
        golden_summ_len_arr = sums_len[0].data.numpy()
        doc_arr = []
        for i in range(np.shape(doc_matrix)[0]):
            temp_sent = " ".join([data.itow[x] for x in doc_matrix[i]][:doc_len_arr[i]])
            doc_arr.append(temp_sent)

        golden_summ_arr = []
        for i in range(np.shape(golden_summ_matrix)[0]):
            temp_sent = " ".join([data.itow[x] for x in golden_summ_matrix[i]][:golden_summ_len_arr[i]])
            golden_summ_arr.append(temp_sent)

        summ_matrix = torch.stack([doc[x] for x in selected_indexs]).data.numpy()
        summ_len_arr = torch.stack([doc_len[x] for x in selected_indexs]).data.numpy()

        summ_arr = []
        for i in range(np.shape(summ_matrix)[0]):
            temp_sent = " ".join([data.itow[x] for x in summ_matrix[i]][:summ_len_arr[i]])
            summ_arr.append(temp_sent)
        score_Rouge = Rouge().get_scores(" ".join(summ_arr), " ".join(golden_summ_arr))
        Rouge_list.append(score_Rouge[0]['rouge-l']['f'])
        print(Rouge_list[-1])
    print('=' * 60)
    print(np.mean(Rouge_list))


def genSentences(args):
    np.set_printoptions(threshold=1e10)

    print('Loading data......')
    data = Dataset(path=args.data_path)

    print('Building model......')
    args.num_words = len(data.weight)  # number of words, use to init word embedding

    sentenceEncoder = SentenceEmbedding(**vars(args))
    args.se_dim = sentenceEncoder.getDim()  # sentence embedding dim
    channelModel = ChannelModel(**vars(args))

    print('Initializing word embeddings......')
    # sentenceEncoder.word_embedding.weight.data.set_(data.weight)
    # sentenceEncoder.word_embedding.weight.requires_grad = False
    # print('Fix word embeddings')

    device = torch.device('cuda' if args.cuda else 'cpu')
    if args.cuda:
        print('Transfer models to cuda......')
    sentenceEncoder, channelModel = sentenceEncoder.to(device), channelModel.to(device)
    identityMatrix = torch.eye(100).to(device)

    print('Initializing optimizer and summary writer......')
    params = [p for p in sentenceEncoder.parameters() if p.requires_grad] + \
             [p for p in channelModel.parameters() if p.requires_grad]

    # sentenceEncoder.load_state_dict(torch.load(os.path.join(args.save_dir, 'se.pkl')))
    # channelModel.load_state_dict(torch.load(os.path.join(args.save_dir, 'channel.pkl')))
    checkpoints = torch.load(args.ckpt_path)
    sentenceEncoder.load_state_dict(checkpoints["se_state_dict"])
    channelModel.load_state_dict(checkpoints["channel_state_dict"])

    valid_count = 0
    Rouge_list, Rouge155_list = [], []
    Rouge_list_2, Rouge_list_l = [], []
    Rouge155_list_2, Rouge155_list_l = [], []
    total_score = None
    # Rouge155_obj = Rouge155(n_bytes=75, stem=True, tmp='.tmp')
    Rouge155_obj = Rouge155(stem=True, tmp=".tmp")
    best_rouge1_arr = []
    redundancy_arr = []

    doc_dir = os.path.join(args.save_dir, "doc")
    my_utils.make_dirs(doc_dir)
    ref_dir = os.path.join(args.save_dir, "ref")
    my_utils.make_dirs(ref_dir)
    sum_dir = os.path.join(args.save_dir, "sum")
    my_utils.make_dirs(sum_dir)

    for batch_iter, valid_batch in tqdm(enumerate(data.gen_test_minibatch()), total=data.test_size):

        if 0 < args.max_test_docs < batch_iter:
            break

        # print(valid_count)
        with torch.no_grad():
            sentenceEncoder.eval()
            channelModel.eval()
            doc, sums, doc_len, sums_len = recursive_to_device(device, *valid_batch)
            num_sent_of_sum = sums[0].size(0)
            D = sentenceEncoder(doc, doc_len)
            S = sentenceEncoder(sums[0], sums_len[0])
            l = D.size(0)
            doc_matrix = doc.cpu().data.numpy()
            doc_len_arr = doc_len.cpu().data.numpy()
            golden_summ_matrix = sums[0].cpu().data.numpy()
            golden_summ_len_arr = sums_len[0].cpu().data.numpy()

            candidate_indexes = [i for i in range(len(doc_len_arr))
                                 if (0 <= doc_len_arr[i] <= 10000)]

            if len(candidate_indexes) < 3:
                continue

            doc_ = ""
            doc_arr = []
            for i in range(np.shape(doc_matrix)[0]):
                temp_sent = " ".join([data.itow[x] for x in doc_matrix[i]][:doc_len_arr[i]])
                doc_ += str(i) + ": " + temp_sent + "\n\n"
                doc_arr.append(temp_sent)

            golden_summ_ = ""
            golden_summ_arr = []
            for i in range(np.shape(golden_summ_matrix)[0]):
                temp_sent = " ".join([data.itow[x] for x in golden_summ_matrix[i]][:golden_summ_len_arr[i]])
                golden_summ_ += str(i) + ": " + temp_sent + "\n\n"
                golden_summ_arr.append(temp_sent)
            selected_indexs = []

            if args.method == 'iterative':
                for _ in range(3):
                    probs = np.zeros([l]) - 100000
                    for i in candidate_indexes:
                        temp = [D[x] for x in selected_indexs]
                        temp.append(D[i])
                        temp_prob, addition = channelModel(D, torch.stack(temp))
                        probs[i] = temp_prob.item()
                    best_index = np.argmax(probs)
                    while best_index in selected_indexs:
                        probs[best_index] = - 100000
                        best_index = np.argmax(probs)
                    selected_indexs.append(best_index)
                _, addition = channelModel(D, S)
                selected_indexs.sort()

            if args.method == 'iterative-delete':
                current_sent_set = range(l)
                best_index = -1
                doc_rouge_matrix = rouge_atten_matrix(doc_arr, doc_arr)
                for i_ in range(num_sent_of_sum):
                    D_ = torch.stack([D[x] for x in current_sent_set])
                    probs = []
                    print(i_, current_sent_set)
                    for i in current_sent_set:
                        temp_prob, addition = channelModel(D_, torch.stack([D[i]]))
                        probs.append(temp_prob.item())
                    best_index = np.argmax(probs)
                    print(current_sent_set[best_index])
                    selected_indexs.append(current_sent_set[best_index])
                    temp = []
                    for i in current_sent_set:
                        if doc_rouge_matrix[current_sent_set[best_index], i] < 0.9:
                            temp.append(i)
                    if len(temp) == 0:
                        break
                    current_sent_set = temp

            probs_arr = []
            if args.method == 'top-k-simple':
                for i in range(l):
                    temp_prob, addition = channelModel(D, torch.stack([D[i]]))
                    probs_arr.append(temp_prob.item())
                for _ in range(3):
                    best_index = np.argmax(probs_arr)
                    probs_arr[best_index] = - 1000000
                    selected_indexs.append(best_index)

            if args.method == 'top-k':
                k_subset = genSubset(range(l), 3)
                probs = []
                for subset in k_subset:
                    temp_prob, addition = channelModel(D, torch.stack([D[i] for i in subset]))
                    probs.append(temp_prob.item())
                index = np.argmax(probs)
                selected_indexs = k_subset[index]

            if args.method == 'random':
                selected_indexs = random.sample(range(l), min(3, l))

            summ_matrix = torch.stack([doc[x] for x in selected_indexs]).cpu().data.numpy()
            summ_len_arr = torch.stack([doc_len[x] for x in selected_indexs]).cpu().data.numpy()

            summ_ = ""
            summ_arr = []
            for i in range(np.shape(summ_matrix)[0]):
                temp_sent = " ".join([data.itow[x] for x in summ_matrix[i]][:summ_len_arr[i]])
                summ_ += str(i) + ": " + temp_sent + "\n\n"
                summ_arr.append(temp_sent)
            # f_ref = open("ref/" + str(batch_iter) + "_reference.txt", "w")
            if args.save_org_doc:
                doc_path = os.path.join(doc_dir, "{}_original_doc.txt".format(batch_iter))
                with open(doc_path, "w") as f_doc:
                    f_doc.write("\n".join(doc_arr))

            ref_path = os.path.join(ref_dir, "{}_reference.txt".format(batch_iter))
            f_ref = open(ref_path, "w")

            sum_path = os.path.join(sum_dir, "{}_decoded.txt".format(batch_iter))
            f_sum = open(sum_path, "w")

            f_ref.write("\n".join(golden_summ_arr))
            f_sum.write("\n".join(summ_arr))

            f_ref.close()
            f_sum.close()

    print('=' * 60)

    total_score = Rouge155_obj.evaluate_folder(sum_dir, ref_dir)
    print(total_score)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_path', required=True, type=str, help='checkpoint path')
    parser.add_argument('--data_path', required=True,
                        help='pickle file obtained by dataset dump or datadir for torchtext')
    parser.add_argument('--SE_type', default='BiGRU', choices=['GRU', 'BiGRU', 'LSTM', 'BiLSTM', 'AVG'])
    parser.add_argument('--method', default='iterative',
                        choices=['random', 'top-k-simple', 'top-k', 'iterative', 'iterative-delete', 'lead-3'])
    parser.add_argument('--word_dim', type=int, default=300, help='dimension of word embeddings')
    parser.add_argument('--hidden_dim', type=int, default=1024, help='dimension of hidden units per layer')
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--num_layers', type=int, default=1, help='number of layers in LSTM/BiLSTM')
    parser.add_argument('--max_test_docs', type=int, default=-1, help='max test docs')
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--save_org_doc', action='store_true')
    parser.add_argument('--save_dir', default="./eval", type=str, help='dir save evaluate')
    args = parser.parse_args()
    return args


def prepare():
    args = parse_args()
    args.cuda = not args.cpu
    args.save_dir = os.path.join(args.save_dir, my_utils.get_time_str())
    my_utils.make_dirs(args.save_dir)

    fileHandler = logging.FileHandler(os.path.join(args.save_dir, 'examples.log'))
    fileHandler.setFormatter(logFormatter)
    rootLogger.addHandler(fileHandler)
    for k, v in vars(args).items():
        print(k + ':' + str(v))
    return args


def main():
    args = prepare()
    if args.method == 'lead-3':
        evalLead3(args)
    else:
        genSentences(args)


if __name__ == "__main__":
    main()
