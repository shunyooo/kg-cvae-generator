"""
Copyright 2019 Pingpong AI Research, ScatterLab

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""


import torch
import torch.nn as nn
import torch.nn.functional as fnn


class CVAELoss(nn.Module):
    def __init__(self, config):
        super(CVAELoss, self).__init__()
        self.config = config
        self.use_hcf = config["use_hcf"]
        self.full_kl_step = config["full_kl_step"]

    def forward(self, model_output, model_input, current_step, is_train, is_valid):
        out_token = model_output["out_token"]
        out_das = model_output["out_das"]
        da_logit = model_output["da_logit"]
        bow_logit = model_output["bow_logit"]
        dec_out = model_output["dec_out"]

        return self.calculate_loss(dec_out, bow_logit, da_logit, out_token,
                                   out_das, model_output, is_train, is_valid, current_step)

    def calculate_seq_loss(self, dec_out, bow_logit, da_logit, out_token, out_das):
        labels = out_token.clone()
        labels = labels[:, 1:]
        label_mask = torch.sign(labels).detach().float()

        dec_out = dec_out.contiguous()

        rc_loss1 = fnn.cross_entropy(dec_out.view(-1, dec_out.size(-1)), labels.reshape(-1),
                                     reduction='none').view(dec_out.size()[:-1])
        rc_loss = torch.sum(rc_loss1 * label_mask, 1)

        avg_rc_loss = torch.mean(rc_loss)

        rc_ppl = torch.exp(torch.sum(rc_loss) / torch.sum(label_mask))

        bow_loss1 = -fnn.log_softmax(bow_logit, dim=1).gather(1, labels) * label_mask
        bow_loss = torch.sum(bow_loss1, 1)
        avg_bow_loss = torch.mean(bow_loss)

        if self.use_hcf:
            avg_sentiment_loss = fnn.cross_entropy(da_logit, out_das)
            _, da_logits = torch.max(da_logit, 1)
            avg_sentiment_acc = (da_logits == out_das).float().sum()/out_das.shape[0]

        else:
            avg_sentiment_loss = avg_bow_loss.new_tensor(0)
            avg_sentiment_acc = 0.0

        return {"avg_rc_loss": avg_rc_loss,
                "rc_ppl": rc_ppl,
                "avg_sentiment_loss": avg_sentiment_loss,
                "avg_sentiment_acc": avg_sentiment_acc,
                "avg_bow_loss": avg_bow_loss}

    @staticmethod
    def gaussian_kld(recog_mu, recog_logvar, prior_mu, prior_logvar):
        kld = -0.5 * torch.sum(1 + (recog_logvar - prior_logvar)
                               - torch.div(torch.pow(prior_mu - recog_mu, 2),
                                           torch.exp(prior_logvar))
                               - torch.div(torch.exp(recog_logvar), torch.exp(prior_logvar)), 1)
        return kld

    def calculate_kl_loss(self, recog_mu, recog_logvar, prior_mu, prior_logvar,
                          is_valid, current_step):
        """
        Calculate KLD (Kullback Leibler Divergence)
        between the recognition network and the prior network.

        if is_valid is True, then kl_weights are fixed to 1.
        else, then kl_weights will be linearly increased to 1 with given step limit value.


        :param recog_mu: The mean value of Recognition Network.
        :param recog_logvar: The variance of Recognition Network.
        :param prior_mu: The prior value of Recognition Network.
        :param prior_logvar: The variance of Recognition Network.
        :param is_valid: Is valid mode or not.
        :param current_step: Current step in training process.
        :return:
        """

        kld = self.gaussian_kld(recog_mu, recog_logvar, prior_mu, prior_logvar)
        avg_kld = torch.mean(kld)
        if not is_valid:
            kl_weights = min((current_step / self.full_kl_step), 1.0)
        else:
            kl_weights = 1.0

        return {"avg_kl_loss": avg_kld, "kl_weights": kl_weights}

    def calculate_loss(self, dec_out, bow_logit, da_logit,
                       out_token, out_das, model_output, is_train, is_valid, current_step):

        losses = {}
        seq_loss = self.calculate_seq_loss(dec_out, bow_logit, da_logit, out_token, out_das)
        losses.update(seq_loss)

        if is_train:
            recog_mulogvar = model_output["recog_mulogvar"]
            prior_mulogvar = model_output["prior_mulogvar"]

            recog_mu, recog_logvar = torch.chunk(recog_mulogvar, 2, 1)
            prior_mu, prior_logvar = torch.chunk(prior_mulogvar, 2, 1)

            kl_loss = self.calculate_kl_loss(recog_mu, recog_logvar, prior_mu, prior_logvar,
                                             is_valid, current_step)
            losses.update(kl_loss)
            elbo = losses["avg_rc_loss"] + losses["kl_weights"] * losses["avg_kl_loss"]
            aug_elbo = losses["avg_bow_loss"] + losses["avg_sentiment_loss"] + elbo

        else:
            elbo = losses["avg_rc_loss"]
            aug_elbo = losses["avg_bow_loss"] + losses["avg_sentiment_loss"] + elbo

        losses["main_loss"] = aug_elbo
        return losses
