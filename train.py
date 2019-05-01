import torch
import time
import argparse
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)-8s %(message)s')
# logFormatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
logFormatter = logging.Formatter('%(message)s')
rootLogger = logging.getLogger()
import random
import shutil
import os
import heapq
import json
from model.noisyChannel import ChannelModel
from model.sentence import SentenceEmbedding
from dataset.data import Dataset
from torch import nn, optim
import numpy as np
from tensorboardX import SummaryWriter
from utils import recursive_to_device, visualize_tensor, genPowerSet
from rouge import Rouge
import my_utils


# from IPython import embed


def rouge_atten_matrix(doc, summ):
    doc_len = len(doc)
    summ_len = len(summ)
    temp_mat = np.zeros([doc_len, summ_len])
    for i in range(doc_len):
        for j in range(summ_len):
            temp_mat[i, j] = Rouge().get_scores(doc[i], summ[j])[0]['rouge-1']['f']
    return temp_mat


def trainChannelModel(args):
    np.set_printoptions(threshold=1e10)

    print('Loading data......')
    data = Dataset(path=args.data_path, fraction=args.fraction)

    print('Loading offline pyrouge max index.....')
    # the index of document sentence which has maximum pyrouge score with current summary sentence
    pyrouge_max_index = json.load(open(args.pyrouge_index))

    print('Building model......')
    args.num_words = len(data.weight)  # number of words
    sentenceEncoder = SentenceEmbedding(**vars(args))
    args.se_dim = sentenceEncoder.getDim()  # sentence embedding dim
    channelModel = ChannelModel(**vars(args))
    logging.info(sentenceEncoder)
    logging.info(channelModel)

    print('Initializing word embeddings......')
    sentenceEncoder.word_embedding.weight.data.set_(data.weight)
    if args.fix_word_embedding:
        sentenceEncoder.word_embedding.weight.requires_grad = False
        print('Fix word embeddings')
    else:
        print('Tune word embeddings')

    device = torch.device('cuda' if args.cuda else 'cpu')
    if args.cuda:
        print('Transfer models to cuda......')
    sentenceEncoder, channelModel = sentenceEncoder.to(device), channelModel.to(device)
    # identityMatrix = torch.eye(100).to(device)

    print('Initializing optimizer and summary writer......')
    params = [p for p in sentenceEncoder.parameters() if p.requires_grad] + \
             [p for p in channelModel.parameters() if p.requires_grad]

    opt_name = args.optimizer
    scheduler_name = "MLR"
    model_name = args.model_name
    curr_time = my_utils.get_time_str()

    if args.resume_ckpt:
        checkpoints = torch.load(args.resume_ckpt)
        opt_name = checkpoints["opt_name"]
        scheduler_name = checkpoints["scheduler_name"]
        print("Load checkpoint from {} done".format(args.resume_ckpt))

        log = checkpoints["log"]
        log.update(dict(Current_Time=curr_time, Args=vars(args)))
    else:
        log = {
            "Train": dict(loss={}, good_prob={}, bad_prob={}, reg={},
                          loss_avg=0, good_prob_avg=0, bad_prob_avg=0, reg_avg=0),
            "Valid": dict(loss={}, good_prob={}, bad_prob={}, reg={}),
            "Args": vars(args),
            "Current_Time": curr_time,
            "Best_Valid_Epochs": [],
            "Epoch_Tize": data.train_size,
        }

    optimizer_class = {
        'adam': optim.Adam,
        'sgd': optim.SGD,
        'adadelta': optim.Adadelta,
    }[opt_name]

    optimizer = optimizer_class(params=params, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[5, 10, 20, 30], gamma=0.5)

    train_writer = SummaryWriter(os.path.join(args.save_dir, 'log', 'train'))
    tic = time.time()
    loss_arr = []
    valid_loss = 0
    valid_all_loss = 0
    valid_acc = 0
    valid_all_acc = 0

    start_epoch = 1
    print('Start training......')
    if args.resume_ckpt:
        # checkpoints = torch.load(args.resume_ckpt)
        sentenceEncoder.load_state_dict(checkpoints["se_state_dict"])
        channelModel.load_state_dict(checkpoints["channel_state_dict"])
        start_epoch = checkpoints["epoch"] + 1

        print("Load resume checkpoints from {} done".format(args.resume_ckpt))

        # sentenceEncoder.load_state_dict(torch.load(os.path.join(args.save_dir, 'se.pkl')))
        # channelModel.load_state_dict(torch.load(os.path.join(args.save_dir, 'channel.pkl')))

    # if args.validation:
    #     validate(data, sentenceEncoder, channelModel, device, args)
    #     return 0
    try:
        os.mkdir(os.path.join(args.save_dir, "checkpoints"))
    except:
        pass

    loss_arr = list(log["Train"]["loss"].values())
    good_prob_arr = list(log["Train"]["good_prob"].values())
    bad_prob_arr = list(log["Train"]["bad_prob"].values())
    reg_arr = list(log["Train"]["reg"].values())

    loss_avg = float(log["Train"]["loss_avg"])
    good_prob_avg = float(log["Train"]["good_prob_avg"])
    bad_prob_avg = float(log["Train"]["bad_prob_avg"])
    reg_avg = float(log["Train"]["reg_avg"])

    global_batch_idx = (start_epoch - 1) * data.train_size
    end_epoch = args.max_epoch
    epoch_time = 0

    for epoch_num in range(start_epoch, end_epoch + 1):
        scheduler.step()
        if args.anneal:
            # from 1 to 0.01 as the epoch_num increases
            channelModel.temperature = 1 - epoch_num * 0.99 / (args.max_epoch - 1)

        if epoch_num % 1 == 0 and args.validation:
            valid_loss, valid_all_loss, valid_acc, valid_all_acc, rouge_score = validate(data, sentenceEncoder,
                                                                                         channelModel, device, args)
            train_writer.add_scalar('validation/loss', valid_loss, epoch_num)
            train_writer.add_scalar('validation/all_loss', valid_all_loss, epoch_num)
            train_writer.add_scalar('validation/acc', valid_acc, epoch_num)
            train_writer.add_scalar('validation/all_acc', valid_all_acc, epoch_num)
            train_writer.add_scalar('validation/rouge', rouge_score, epoch_num)

        eq = 0
        rouge_arr = []
        for batch_iter, train_batch in enumerate(data.gen_train_minibatch()):
            start_batch_time = time.time()

            sentenceEncoder.train()
            channelModel.train()

            # progress = epoch_num + batch_iter / data.train_size
            global_batch_idx += 1

            doc, sums, doc_len, sums_len = recursive_to_device(device, *train_batch)
            num_sent_of_sum = sums[0].size(0)

            if num_sent_of_sum == 1:  # if delete, summary should have more than one sentence
                continue

            D = sentenceEncoder(doc, doc_len)
            S_good = sentenceEncoder(sums[0], sums_len[0])
            # neg_sent_embed = sentenceEncoder(sums[1], sums_len[1])

            l = S_good.size(0)

            # doc_matrix = doc.cpu().data.numpy()
            # doc_len_arr = doc_len.cpu().data.numpy()
            # summ_matrix = sums[0].cpu().data.numpy()
            # summ_len_arr = sums_len[0].cpu().data.numpy()

            # doc_ = []
            # summ_ = []
            # for i in range(np.shape(doc_matrix)[0]):
            #     doc_.append(" ".join([data.itow[x] for x in doc_matrix[i]][:doc_len_arr[i]]))

            index = random.randint(0, l - 1)
            # summ_.append(" ".join([data.itow[x] for x in summ_matrix[index]][:summ_len_arr[index]]))

            # ----------- fetch best_index from pyrouge_max_index --------
            ori_index = data.train_ori_index[batch_iter]
            assert len(pyrouge_max_index[ori_index]) == l, \
                "number of pyrouge_max_index[i] must be equal to the number of summary sentences"
            best_index = pyrouge_max_index[ori_index][index]

            # Change original code
            candidate_rand_indexes = list(range(D.size(0)))
            candidate_rand_indexes.remove(best_index)
            worse_indexes = random.sample(candidate_rand_indexes, min(D.size(0), 1))

            temp_good = []
            for i in range(l):
                if i != index:
                    temp_good.append(S_good[i])
                else:
                    temp_good.append(D[best_index])

            S_good = torch.stack(temp_good)

            S_bads = []
            for worse_index in worse_indexes:
                temp_bad = []
                for i in range(l):
                    if i != index:
                        temp_bad.append(S_good[i])
                    else:
                        temp_bad.append(D[worse_index])
                S_bads.append(torch.stack(temp_bad))

            # prob calculation
            good_prob, addition = channelModel(D, S_good)
            good_prob_vector, good_attention_weight = addition['prob_vector'], addition['att_weight']

            bad_probs, bad_probs_vector = [], []
            bad_prob = 0.

            for S_bad in S_bads:
                bad_prob, addition = channelModel(D, S_bad)
                bad_probs.append(bad_prob)
                bad_probs_vector.append(addition['prob_vector'])
            bad_index = np.argmax([p.item() for p in bad_probs])
            bad_prob = bad_probs[bad_index]

            ########### loss ############
            loss_prob_term = bad_prob - good_prob
            n, m = good_attention_weight.size()
            regularization_term = torch.norm(
                torch.mm(good_attention_weight.t(), good_attention_weight) - n / m * torch.eye(m).to(device), 2)
            loss = loss_prob_term + args.alpha * regularization_term

            if loss_prob_term.item() > -args.margin:
                # Chi optimize khi model kem
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(parameters=params, max_norm=args.clip)
                optimizer.step()

            loss_val = loss.item()
            good_prob_val = good_prob.item()
            bad_prob_val = bad_prob.item()
            reg_val = regularization_term.item()

            loss_avg = (loss_avg * len(loss_arr) + loss_val) / (len(loss_arr) + 1)
            good_prob_avg = (good_prob_avg * len(good_prob_arr) + good_prob_val) / (len(good_prob_arr) + 1)
            bad_prob_avg = (bad_prob_avg * len(bad_prob_arr) + bad_prob_val) / (len(bad_prob_arr) + 1)
            reg_avg = (reg_avg * len(reg_arr) + reg_val) / (len(reg_arr) + 1)

            loss_arr.append(loss_val)
            good_prob_arr.append(good_prob_val)
            bad_prob_arr.append(bad_prob_val)
            reg_arr.append(reg_val)

            batch_time = time.time() - start_batch_time
            epoch_time += batch_time

            if global_batch_idx % 50 == 0:
                logging.info('Train ||Epoch: {}/{} ||Batch_idx: {}/{} ||(Loss/Bad/Good/Reg) ||'
                             'Curr: ({:.2f},{:.2f},{:.2f}) ||Avg: ({:.4f},{:.2f},{:.2f}) ||'
                             'Batch_time: {:.4f}s'.format(
                                epoch_num, end_epoch, batch_iter, data.train_size,
                                loss_val, bad_prob_val, good_prob_val, reg_val,
                                loss_avg, bad_prob_avg, good_prob_avg, reg_avg,
                                batch_time))

            # Update log
            log["Train"]["loss"][str(global_batch_idx)] = loss.item()
            log["Train"]["good_prob"][str(global_batch_idx)] = good_prob.item()
            log["Train"]["bad_prob"][str(global_batch_idx)] = bad_prob.item()
            log["Train"]["reg"][str(global_batch_idx)] = regularization_term.item()
            log["Train"]["time"][str(global_batch_idx)] = batch_time

        logging.info("\nTrain || Epoch time: {:.2f}s\n".format(epoch_time))

        if epoch_num % 1 == 0:
            # try:
            #     os.mkdir(os.path.join(args.save_dir, 'checkpoints/' + str(epoch_num)))
            # except:
            #     pass

            # Save checkpoints
            states = dict(epoch=epoch_num, se_state_dict=sentenceEncoder.state_dict(),
                          channel_state_dict=channelModel.state_dict(),
                          optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
                          opt_name=opt_name, scheduler_name=scheduler_name, model_name=model_name,
                          log=log)
            save_ckpt_path = os.path.join(args.save_dir, "checkpoints", '{}_ckpt_{}.pth'.format(model_name, epoch_num))
            torch.save(states, save_ckpt_path)

            # torch.save(sentenceEncoder.state_dict(),
            #            os.path.join(args.save_dir, 'checkpoints/' + str(epoch_num) + '/se.pkl'))
            # torch.save(channelModel.state_dict(),
            #            os.path.join(args.save_dir, 'checkpoints/' + str(epoch_num) + '/channel.pkl'))

    [rootLogger.removeHandler(h) for h in rootLogger.handlers if isinstance(h, logging.FileHandler)]


def validate(data_, sentenceEncoder_, channelModel_, device_, args):
    neg_count = 0
    valid_iter_count = 0
    all_neg_count = 0
    sent_count = 0
    loss_arr = []
    all_loss_arr = []
    Rouge_list = []

    for batch_iter, valid_batch in enumerate(data_.gen_valid_minibatch()):
        if not (batch_iter % 100 == 0):
            continue
        sentenceEncoder_.eval()
        channelModel_.eval()
        doc, sums, doc_len, sums_len = recursive_to_device(device_, *valid_batch)
        num_sent_of_sum = sums[0].size(0)
        D = sentenceEncoder_(doc, doc_len)
        l = D.size(0)
        if l < 2:
            continue

        doc_matrix = doc.cpu().data.numpy()
        doc_len_arr = doc_len.cpu().data.numpy()
        golden_summ_matrix = sums[0].cpu().data.numpy()
        golden_summ_len_arr = sums_len[0].cpu().data.numpy()

        doc_ = ""
        doc_arr = []
        for i in range(np.shape(doc_matrix)[0]):
            temp_sent = " ".join([data_.itow[x] for x in doc_matrix[i]][:doc_len_arr[i]])
            doc_ += str(i) + ": " + temp_sent + "\n\n"
            doc_arr.append(temp_sent)

        golden_summ_ = ""
        golden_summ_arr = []
        for i in range(np.shape(golden_summ_matrix)[0]):
            temp_sent = " ".join([data_.itow[x] for x in golden_summ_matrix[i]][:golden_summ_len_arr[i]])
            golden_summ_ += str(i) + ": " + temp_sent + "\n\n"
            golden_summ_arr.append(temp_sent)

        selected_indexs = []
        probs_arr = []

        for _ in range(3):
            probs = []
            for i in range(l):
                temp = [D[x] for x in selected_indexs]
                temp.append(D[i])
                temp_prob, addition = channelModel_(D, torch.stack(temp))
                probs.append(temp_prob.item())
            probs_arr.append(probs)
            best_index = np.argmax(probs)
            while best_index in selected_indexs:
                probs[best_index] = -100000
                best_index = np.argmax(probs)
            selected_indexs.append(best_index)
        summ_matrix = torch.stack([doc[x] for x in selected_indexs]).cpu().data.numpy()
        summ_len_arr = torch.stack([doc_len[x] for x in selected_indexs]).cpu().data.numpy()

        summ_ = ""
        summ_arr = []
        for i in range(np.shape(summ_matrix)[0]):
            temp_sent = " ".join([data_.itow[x] for x in summ_matrix[i]][:summ_len_arr[i]])
            summ_ += str(i) + ": " + temp_sent + "\n\n"
            summ_arr.append(temp_sent)

        best_rouge_summ_arr = []
        for s in golden_summ_arr:
            temp = []
            for d in doc_arr:
                temp.append(Rouge().get_scores(s, d)[0]['rouge-1']['f'])
            index = np.argmax(temp)
            best_rouge_summ_arr.append(doc_arr[index])
        score_Rouge = Rouge().get_scores(" ".join(summ_arr), " ".join(golden_summ_arr))
        Rouge_list.append(score_Rouge[0]['rouge-1']['f'])

    rouge_score = np.mean(Rouge_list)
    print("ROUGE 1/100 sample : ", rouge_score)

    for batch_iter, valid_batch in enumerate(data_.gen_valid_minibatch()):
        if not (batch_iter % 100 == 0):
            continue
        sentenceEncoder_.eval();
        channelModel_.eval()
        valid_iter_count += 1
        doc, sums, doc_len, sums_len = recursive_to_device(device_, *valid_batch)
        num_sent_of_sum = sums[0].size(0)
        if num_sent_of_sum == 1:  # if delete, summary should have more than one sentence
            continue
        D = sentenceEncoder_(doc, doc_len)
        S_good = sentenceEncoder_(sums[0], sums_len[0])
        # neg_sent_embed = sentenceEncoder_(sums[1], sums_len[1])

        l = S_good.size(0)
        S_bads = []

        doc_matrix = doc.cpu().data.numpy()
        doc_len_arr = doc_len.cpu().data.numpy()
        summ_matrix = sums[0].cpu().data.numpy()
        summ_len_arr = sums_len[0].cpu().data.numpy()
        doc_ = []
        summ_ = []
        for i in range(np.shape(doc_matrix)[0]):
            doc_.append(" ".join([data_.itow[x] for x in doc_matrix[i]][:doc_len_arr[i]]))

        index = random.randint(0, l - 1)
        summ_.append(" ".join([data_.itow[x] for x in summ_matrix[index]][:summ_len_arr[index]]))

        atten_mat = rouge_atten_matrix(summ_, doc_)
        best_index = np.argmax(atten_mat[0])
        worst_index = np.argmin(atten_mat[0])
        temp_good = []
        temp_bad = []
        for i in range(l):
            if (not i == index):
                temp_good.append(S_good[i])
                temp_bad.append(S_good[i])
            else:
                temp_good.append(D[best_index])
                temp_bad.append(D[worst_index])
        S_good = torch.stack(temp_good)
        S_bads.append(torch.stack(temp_bad))
        # prob calculation
        good_prob, addition = channelModel_(D, S_good)
        good_prob_vector, good_attention_weight = addition['prob_vector'], addition['att_weight']
        bad_probs, bad_probs_vector = [], []
        for S_bad in S_bads:
            bad_prob, addition = channelModel_(D, S_bad)
            bad_probs.append(bad_prob)
            bad_probs_vector.append(addition['prob_vector'])
        bad_index = np.argmax([p.item() for p in bad_probs])
        bad_prob = bad_probs[bad_index]

        ########### loss ############
        loss_prob_term = bad_prob - good_prob
        loss = loss_prob_term.item()
        loss_arr.append(loss)
        for bad in bad_probs:
            all_loss_arr.append((bad - good_prob).item())
        if args.visualize and valid_iter_count % 100 == 0:
            doc_matrix = doc.cpu().data.numpy()
            doc_len_arr = doc_len.cpu().data.numpy()
            summ_matrix = sums[0].cpu().data.numpy()
            summ_len_arr = sums_len[0].cpu().data.numpy()
            doc_ = ""
            for i in range(np.shape(doc_matrix)[0]):
                doc_ += str(i) + ": " + " ".join([data_.itow[x] for x in doc_matrix[i]][:doc_len_arr[i]]) + "\n\n"

            summ_ = ""
            for i in range(np.shape(summ_matrix)[0]):
                summ_ += str(i) + ": " + " ".join([data_.itow[x] for x in summ_matrix[i]][:summ_len_arr[i]]) + "\n\n"
            logging.info("\nsample case %d:\n\ndocument:\n\n%s\n\nsummary:\n\n%s\n\nattention matrix:\n\n%s\n\n" % (
                valid_iter_count, str(doc_), str(summ_), str(good_attention_weight.cpu().data.numpy())))

    valid_loss = float(np.mean(loss_arr))
    valid_all_loss = float(np.mean(all_loss_arr))
    valid_acc = (np.sum(np.int32(np.array(loss_arr) < 0)) + 0.) / len(loss_arr)
    valid_all_acc = (np.sum(np.int32(np.array(all_loss_arr) < 0)) + 0.) / len(all_loss_arr)
    logging.info("avg loss: %4f, avg all_loss: %4f, acc: %4f, all_acc: %4f" % (
        valid_loss, valid_all_loss, valid_acc, valid_all_acc))

    return valid_loss, valid_all_loss, valid_acc, valid_all_acc, rouge_score


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pyrouge_index', required=True, help='json file of offline max pyrouge index')
    parser.add_argument('--data_path', required=True, help='pickle file obtained by dataset dump or datadir for torchtext')
    parser.add_argument('--max_epoch', type=int, default=10)

    parser.add_argument('--resume_ckpt', help='path contain pretrained model')
    parser.add_argument('--save_dir', type=str, default="./experiments", help='path to save checkpoints and logs')
    parser.add_argument('--fix_word_embedding', action='store_true', help='specified to fix embedding vectors')
    parser.add_argument('--SE_type', default='BiGRU', choices=['GRU', 'BiGRU', 'AVG'])
    parser.add_argument('--word_dim', type=int, default=300, help='dimension of word embeddings')
    parser.add_argument('--hidden_dim', type=int, default=1024, help='dimension of hidden units per layer')
    parser.add_argument('--model_name', default='deep_channel')
    parser.add_argument('--validation', action='store_true')
    parser.add_argument('--optimizer', default='adam', choices=['adam', 'sgd', 'adadelta'])
    parser.add_argument('--lr', type=float, default=1e-5, help='initial learning rate')

    parser.add_argument('--num_layers', type=int, default=1, help='number of layers in LSTM/BiLSTM')
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--margin', type=float, default=1e10, help='margin of hinge loss, must >= 0')
    parser.add_argument('--clip', type=float, default=5, help='clip to prevent the too large grad')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='weight decay rate per batch')
    parser.add_argument('--cpu', action='store_true', default=False)
    parser.add_argument('--batch_size', type=int, default=1, help='batch size for training, not used now')
    parser.add_argument('--anneal', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--seed', type=int, default=666, help='random seed')
    parser.add_argument('--alpha', type=float, default=0.001, help='weight of regularization term')
    parser.add_argument('--fraction', type=float, default=1, help='fraction of training set reduction')

    args = parser.parse_args()
    return args


def prepare():
    # dir preparation
    args = parse_args()
    args.save_dir = os.path.join(args.save_dir, my_utils.get_time_str())
    my_utils.make_dirs(args.save_dir)
    args.cuda = not args.cpu

    # if not args.load_previous_model:
    #     if os.path.isdir(args.save_dir):
    #         shutil.rmtree(args.save_dir)
    #     os.mkdir(args.save_dir)
    # seed setting
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        if not args.cuda:
            print("WARNING: You have a CUDA device, so you should probably run with --cuda")
        else:
            torch.cuda.manual_seed(args.seed)
    # make logging.info display into both shell and file
    fileHandler = logging.FileHandler(os.path.join(args.save_dir, 'stdout.log'))
    fileHandler.setFormatter(logFormatter)
    rootLogger.addHandler(fileHandler)
    # args display
    for k, v in vars(args).items():
        logging.info(k + ':' + str(v))
    return args


def main():
    args = prepare()
    trainChannelModel(args)


if __name__ == '__main__':
    main()
