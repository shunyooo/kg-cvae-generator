import codecs
import time
import os
import glob
import sys
import re

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.optim import Adam

from .model_utils import gaussian_kld, get_bleu_stats


def get_checkpoint_state(ckp_dir):
    files = os.path.join(ckp_dir, "*.pth")
    files = glob.glob(files)
    files.sort(key=os.path.getmtime)
    return len(files) > 0 and files[-1] or None


class KorCVAETrainer:
    def __init__(self, model, config):
        self.model = model
        self.vocab_class = model.vocab_class
        self.config = config

        self.sentiment_vocab = ['Strong Negative', 'Weak Negative', 'Neutral', 'Weak Positive', 'Strong Positive']
        self.relation_vocab = ['Friends', 'Lover']
        self.meta_vocab = ['F_COLLEGIAN', 'F_CIVILIAN', 'F_STUDENT', 'M_COLLEGIAN', 'M_CIVILIAN', 'M_STUDENT']

        self.device = torch.device(self.config['device'])
        self.optimizer = Adam(model.parameters(), lr=config['init_lr'])
        self.vocab = self.vocab_class.vocab
        self.rev_vocab = self.vocab_class.rev_vocab

        self.global_ep = 0
        self.global_t = 0

        if config['resume']:
            log_dir = os.path.join(config['work_dir'], config['test_path'])
        else:
            log_dir = os.path.join(config['work_dir'], 'run'+str(int(time.time())))
        ckp_dir = os.path.join(log_dir, 'checkpoints')
        model_dir = os.path.join(log_dir, 'models')
        if not os.path.exists(log_dir):
            os.mkdir(log_dir)
            print('making directory', log_dir)
        if not os.path.exists(ckp_dir):
            os.mkdir(ckp_dir)
            print('making directory', ckp_dir)
        if not os.path.exists(model_dir):
            os.mkdir(model_dir)
            print('making directory', model_dir)
        ckpt = get_checkpoint_state(ckp_dir)
        print("Created models with fresh parameters.")
        self.model.apply(lambda m: [nn.init.uniform_(p.data, -1.0 * config['init_w'], config['init_w'])
                                    for p in m.parameters()])

        # load W2V
        if self.vocab_class.word2vec is not None and not config['forward_only']:
            print('Loaded Word2Vec')
            self.model.word_embedding.weight.data.copy_(torch.from_numpy(np.array(self.vocab_class.word2vec)))
        self.model.word_embedding.weight.data[self.rev_vocab['Ω']].fill_(0)
        self.model.word_embedding.weight.data[self.rev_vocab['<PAD>']].fill_(0)
        self.model.word_embedding.weight.data[self.rev_vocab['<SOS>']].fill_(0)
        self.model.word_embedding.weight.data[self.rev_vocab['<EOS>']].fill_(0)
        # self.model.word_embedding.weight.data[self.rev_vocab['<EMO>']].fill_(0)
        self.model.word_embedding.weight.data[self.rev_vocab['<UNK>']].fill_(0)

        if ckpt:
            print("Reading dm models parameters from %s" % ckpt)
            self.model.load_state_dict(torch.load(ckpt))

        self.model = self.model.to(self.device)

        self.dm_checkpoint_path = os.path.join(ckp_dir, model.__class__.__name__+ "-%d.pth")
        self.dm_model_path = os.path.join(model_dir, model.__class__.__name__+ "-%d.pth")
        self.patience = config['patience']
        self.dev_loss_threshold = np.inf
        self.best_dev_loss = np.inf

    @staticmethod
    def print_loss(prefix, loss_names, losses, postfix):
        template = "%s "
        for name in loss_names:
            template += "%s " % name
            template += " %f "
        template += "%s"
        template = re.sub(' +', ' ', template)
        avg_losses = []
        values = [prefix]

        for loss in losses:
            values.append(np.mean(loss))
            avg_losses.append(np.mean(loss))
        values.append(postfix)

        print(template % tuple(values))
        return avg_losses

    def run_epoch(self, epoch, train_loader, valid_loader, save_model=False):
        patience = self.patience
        print(">> Epoch %d with lr %f" % (epoch, self.optimizer.state_dict()['param_groups'][0]['lr']))
        train_loss = self.train_model(train_loader)
        self.model.eval()
        valid_loss = self.valid_model(valid_loader)
        self.model.train()

        if valid_loss < self.best_dev_loss:
            if valid_loss <= self.dev_loss_threshold * self.config['improve_threshold']:
                patience = max(self.patience, epoch * self.config['patient_increase'])
                self.dev_loss_threshold = valid_loss
            if save_model:
                self.save_model(self.dm_checkpoint_path % epoch)
                self.save_checkpoint(self.dm_model_path % epoch)
            self.best_dev_loss = valid_loss

        if self.config['early_stop'] and patience <= epoch:
            print("!!Early stop due to run out of patience!!")
        print("Best validation loss %f // %f" % (float(self.best_dev_loss), float(valid_loss)))

    def get_test_output(self, test_loader, output_path, epoch):
        self.load_checkpoint(self.dm_model_path % epoch)
        self.model.eval()
        dest_f = codecs.open(output_path, 'w', encoding='utf-8')
        self.test_model(test_loader, repeat=10, dest=dest_f)
        dest_f.close()
        self.model.train()

    def get_losses(self, dec_out, final_ctx, bow_logit, sent_logit, out_token, out_sent):

        # no sos tokens in target
        labels = out_token.clone()
        labels = labels[:, 1:]
        label_mask = torch.sign(labels).detach().float()

        rc_loss1 = F.cross_entropy(dec_out.view(-1, dec_out.size(-1)), labels.reshape(-1), reduction='none').view(dec_out.size()[:-1])

        # rc_loss = F.cross_entropy(dec_out.view(-1, dec_out.size(-1)), labels.reshape(-1), reduction='none')
        # rc_loss = rc_loss.view(dec_out.size()[:-1])
        rc_loss = torch.sum(rc_loss1 * label_mask, 1)

        avg_rc_loss = torch.mean(rc_loss)

        # used only for perpliexty calculation. Not used for optimzation
        rc_ppl = torch.exp(torch.sum(rc_loss) / torch.sum(label_mask))

        """ as n-trial multimodal distribution. """
        bow_loss1 = -F.log_softmax(bow_logit, dim=1).gather(1, labels) * label_mask
        bow_loss = torch.sum(bow_loss1, 1)
        avg_bow_loss = torch.mean(bow_loss)

        # reconstruct the meta info about X
        if self.config['use_hcf']:
            avg_sentiment_loss = F.cross_entropy(sent_logit, out_sent)
        else:
            avg_sentiment_loss = avg_bow_loss.new_tensor(0)

        return avg_rc_loss, rc_ppl, avg_bow_loss, avg_sentiment_loss

    def get_kl_loss(self, recog_mu, recog_logvar, prior_mu, prior_logvar, mode):
        kld = gaussian_kld(recog_mu, recog_logvar, prior_mu, prior_logvar)
        avg_kld = torch.mean(kld)
        if mode == 'train':
            kl_weights = min(self.global_t / self.config['full_kl_step'], 1.0)
        else:
            kl_weights = 1.0
        return avg_kld, kl_weights

    def train_model(self, loader):
        elbo_losses = []
        rc_losses = []
        rc_ppls = []
        kl_losses = []
        bow_losses = []
        start_time = time.time()
        loss_names = ["elbo_loss", "bow_loss", "rc_loss", "rc_peplexity", "kl_loss"]

        for ptr, feed_dict in enumerate(loader):
            dec_outs, final_ctx_state, bow_logit, sent_logit, out_tok, out_sentiment, recog_mulogvar, prior_mulogvar = self.model.forward(feed_dict, mode='train')

            recog_mu, recog_logvar = torch.chunk(recog_mulogvar, 2, 1)
            prior_mu, prior_logvar = torch.chunk(prior_mulogvar, 2, 1)
            avg_kld, kl_weights = self.get_kl_loss(recog_mu, recog_logvar, prior_mu, prior_logvar, mode='train')
            avg_rc_loss, rc_ppl, avg_bow_loss, avg_sentiment_loss = self.get_losses(dec_outs, final_ctx_state, bow_logit, sent_logit, out_tok, out_sentiment)

            elbo = avg_rc_loss + kl_weights * avg_kld
            aug_elbo = avg_bow_loss + avg_sentiment_loss + elbo

            elbo_loss, bow_loss, rc_loss, rc_ppl, kl_loss = elbo.item(), \
                                                            avg_bow_loss.item(), \
                                                            avg_rc_loss.item(), \
                                                            rc_ppl.item(), \
                                                            avg_kld.item()

            self.optimize(aug_elbo)
            elbo_losses.append(elbo_loss)
            bow_losses.append(bow_loss)
            rc_ppls.append(rc_ppl)
            rc_losses.append(rc_loss)
            kl_losses.append(kl_loss)

            self.global_t += 1

            if ptr % (self.config['log_freq']) == 0:
                self.print_loss("%.2f" % (ptr / float(len(loader))),
                                loss_names, [elbo_losses, bow_losses, rc_losses, rc_ppls, kl_losses], "kl_w %f" % kl_weights)

        # finish epoch!
        torch.cuda.synchronize()
        epoch_time = time.time() - start_time
        avg_losses = self.print_loss("Epoch Done", loss_names,
                                     [elbo_losses, bow_losses, rc_losses, rc_ppls, kl_losses],
                                     "step time %.4f" % (epoch_time / (len(loader))))
        return avg_losses[0]

    def valid_model(self, valid_loader):
        elbo_losses = []
        rc_losses = []
        rc_ppls = []
        bow_losses = []
        kl_losses = []

        for feed_dict in valid_loader:
            if feed_dict is None:
                break
            with torch.no_grad():
                dec_outs, final_ctx_state, bow_logit, sent_logit, out_tok, out_sentiment, recog_mulogvar, prior_mulogvar = self.model.forward(feed_dict, mode='valid')
            recog_mu, recog_logvar = torch.chunk(recog_mulogvar, 2, 1)
            prior_mu, prior_logvar = torch.chunk(prior_mulogvar, 2, 1)
            avg_kld, kl_weights = self.get_kl_loss(recog_mu, recog_logvar, prior_mu, prior_logvar, mode='valid')
            avg_rc_loss, rc_ppl, avg_bow_loss, avg_sentiment_loss = self.get_losses(dec_outs, final_ctx_state, bow_logit, sent_logit, out_tok, out_sentiment)

            elbo = avg_rc_loss + kl_weights * avg_kld
            aug_elbo = avg_bow_loss + avg_sentiment_loss + elbo

            elbo_loss, bow_loss, rc_loss, rc_ppl, kl_loss = elbo.item(), \
                                                            avg_bow_loss.item(), \
                                                            avg_rc_loss.item(), \
                                                            rc_ppl.item(), \
                                                            avg_kld.item()
            elbo_losses.append(elbo_loss)
            rc_losses.append(rc_loss)
            rc_ppls.append(rc_ppl)
            bow_losses.append(bow_loss)
            kl_losses.append(kl_loss)

        avg_losses = self.print_loss('VALID', ["elbo_loss", "bow_loss", "rc_loss", "rc_peplexity", "kl_loss"],
                                     [elbo_losses, bow_losses, rc_losses, rc_ppls, kl_losses], "")
        return avg_losses[0]

    def test_model(self, test_loader, repeat=5, dest=sys.stdout):
        recall_bleus = []
        prec_bleus = []

        for ptr, feed_dict in enumerate(test_loader):
            sample_words = list()
            sample_sentiment = list()
            with torch.no_grad():
                for rep in range(repeat):
                    dec_outs, final_ctx_state, bow_logit, sent_logit, out_tok, out_sentiment, recog_mulogvar, prior_mulogvar = self.model.forward(feed_dict, mode='test')
                    if final_ctx_state is not None:
                        dec_out_words = final_ctx_state
                    else:
                        _, dec_out_words = torch.max(dec_outs, 2)

                    # word_outs: (batch_size, seq_len) ex) (50, 40)
                    word_outs, sent_logit = dec_out_words.cpu().numpy(), sent_logit.cpu().numpy()

                    sample_words.append(word_outs)
                    sample_sentiment.append(sent_logit)

            true_floor = feed_dict["floors"].cpu().numpy()
            true_srcs = np.array(feed_dict["input_contexts"])
            true_src_lens = feed_dict["context_lens"].cpu().numpy()

            true_outs = np.array(feed_dict["output_tokens"])
            true_relation = feed_dict["relations"].cpu().numpy()
            true_my = feed_dict['my_profile'].cpu().numpy()
            true_ot = feed_dict['ot_profile'].cpu().numpy()
            true_sentiment = feed_dict["output_sentiments"].cpu().numpy()

            if dest != sys.stdout:
                if ptr % (len(test_loader)) == 0:
                    print("%.2f >> " % (ptr / float(len(test_loader)))),

            for e, b_id in enumerate(range(test_loader.batch_size)):
                # print the dialog context
                dest.write("Batch %d index %d of relation %s\n" % (ptr, b_id, self.relation_vocab[true_relation[b_id]]))
                dest.write("my_profile(%d): %s // ot_profile(%d): %s\n" % (int(np.argmax(true_floor[b_id, 0]) == 0),
                                                                           self.meta_vocab[true_my[b_id]],
                                                                           int(np.argmax(true_floor[b_id, 0])),
                                                                           self.meta_vocab[true_ot[b_id]]))
                start = np.maximum(0, len(true_src_lens[b_id]) - 5)
                for t_id in range(start, true_srcs.shape[1], 1):
                    src_str = " ".join([self.vocab[e] for e in true_srcs[b_id, t_id] if e != 0])
                    dest.write("Src %d-%d: %s\n" % (t_id, int(np.argmax(true_floor[b_id, t_id])), src_str))
                # print the true outputs
                true_tokens = [self.vocab[e] for e in true_outs[b_id] if e not in [0, self.model.eos_id, self.model.sos_id]]
                true_str = " ".join(true_tokens).replace(" ' ", "'")
                sent_str = self.sentiment_vocab[true_sentiment[b_id]]
                # print the predicted outputs
                dest.write("Target (%s) >> %s\n" % (sent_str, true_str))
                local_tokens = []
                for r_id in range(repeat):
                    pred_outs = sample_words[r_id]
                    pred_sent = np.argmax(sample_sentiment[r_id], axis=1)[0]
                    pred_tokens = [self.vocab[e] for e in pred_outs[b_id].tolist() if e != self.model.eos_id and e != 0]
                    if len(pred_tokens) == 0:
                        pred_str = " "
                    else:
                        pred_str = " ".join(pred_tokens).replace(" ' ", "'")
                    dest.write("Sample %d (%s) >> %s\n" % (r_id, self.sentiment_vocab[pred_sent], pred_str))
                    local_tokens.append(pred_tokens)

                max_bleu, avg_bleu = get_bleu_stats(true_tokens, local_tokens)
                recall_bleus.append(max_bleu)
                prec_bleus.append(avg_bleu)
                # make a new line for better readability
                dest.write("\n")

                # 한 batch 에서 10개 뽑으면 나옴
                if e == 10:
                    break

            # test 결과 100 개 뽑은 후에는 나옴
            if ptr == 100:
                break

        avg_recall_bleu = float(np.mean(recall_bleus))
        avg_prec_bleu = float(np.mean(prec_bleus))
        avg_f1 = 2 * (avg_prec_bleu * avg_recall_bleu) / (avg_prec_bleu + avg_recall_bleu + 10e-12)
        report = "Avg recall BLEU %f, avg precision BLEU %f and F1 %f (only 1 reference response. Not final result)" \
                 % (avg_recall_bleu, avg_prec_bleu, avg_f1)
        print(report)
        dest.write(report + "\n")
        print("Done testing")

    def optimize(self, loss):
        with torch.autograd.set_detect_anomaly(True):
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def save_model(self, path):
        self.optimizer.zero_grad()
        params = [
            {"param": param, "require_grad": param.requires_grad}
            for param in self.model.parameters()
        ]

        for param in params:
            param["param"].require_grad = False

        with torch.no_grad():
            torch.save(self.model.state_dict(), path)

        print(f"Model Saved on {path}")

        for param in params:
            if param["require_grad"]:
                param["param"].require_grad = True

    def save_checkpoint(self, path):
        torch.save(
            {
                "epoch": self.global_ep,
                "steps": self.global_t,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
            },
            path,
        )

    def load_checkpoint(self, path):
        checkpoint = torch.load(path)
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.global_ep = checkpoint["epoch"]
        self.global_t = checkpoint["steps"]


