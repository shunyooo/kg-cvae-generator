import trainer.trainer as trainer
import trainer.cvae.criterion as criterion

from typing import Dict, List, Optional, Union

import time
import torch
import torch.nn as nn
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import StepLR


class CVAETrainer(trainer.Trainer):
    def get_step_metric(self,
                        model_output: dict,
                        model_input,
                        loss: Optional[Union[float, torch.Tensor]] = None):
        """
        calculate metrics from step output
        :param model_output: forward output from model
        :param model_input: model input which feed into forward method
        :param loss: loss score from criterion
        :return: dictionary of metrics
        """
        return {}

    def get_epoch_metric(self, epoch_output: List[dict]) -> dict:
        """
        calculate metrics from epoch step outputs
        :param epoch_output: list of each step output (dict)
        :return: dictionary of metrics
        """

        return {}

    def calculate_loss(self,
                       model_output,
                       model_input,
                       current_step,
                       is_train,
                       is_valid):
        """
        calculate the loss using criterion with model_output, model_input
        :param model_output: forward output from model
        :param model_input: model input which feed into forward method
        :param current_step: current step.
        :param is_train: if model is in train_mode or not.
        :param is_valid: if model is in valid_mode or not.
        :return: loss score
        """
        return self.criterion(model_output, model_input, current_step, is_train, is_valid)

    def report(self, metrics: List[dict], is_train: Optional[bool] = True):
        """
        report metrics during total training or evaluation process
        :param metrics: list of metric(dict) from process
        :param is_train: is it currently train status
        """
        if is_train:
            print("Mode: Training")
        else:
            print("Mode: Testing")

    def report_per_epoch(
        self,
        metrics: List[dict],
        mode_name: str,
        epoch: Optional[int] = None,
        elapsed_time: float = 0.0,
        is_train: Optional[bool] = True,
    ):
        """
        report metrics during one-epoch training or evaluation
        :param metrics: list of metric(dict) from one-epoch
        :param mode_name: a name of report set (train, test, valid...)
        :param epoch: current epoch index
        :param elapsed_time: elapsed time
        :param is_train: is it currently train status
        """
        def print_with_logging(file_printer, log_mesg):
            file_printer.write(log_mesg + "\n")
            print(log_mesg)

        log_file_name = self.log_path.format(epoch)
        log_writer = open(log_file_name, "w")

        losses = [metrics[i]["loss"] for i in range(len(metrics))]
        loss = {}
        for key in losses[0].keys():
            loss[key] = 0.0
            for l in losses:
                if type(l[key]) != float:
                    loss[key] += l[key].item()
                else:
                    loss[key] += l[key]
            loss[key] /= len(losses)

        metric = []
        for key, value in loss.items():
            metric.append("{0}: {1:.3f}". format(key, value))
        metric_str = ", ".join(metric)

        elapsed_time_str = mode_name + " elapsed Time for Epoch {0} %H:%M:%S".format(epoch)
        elapsed_time_str = str(time.strftime(elapsed_time_str, time.gmtime(elapsed_time)))
        print_with_logging(log_writer, elapsed_time_str)

        metric_title = "Metric in {0} set for Epoch #{1}: {2}".format(mode_name, epoch, metric_str)
        print_with_logging(log_writer, metric_title)

        epoch_ex_str = "========== {0} Examples for Epoch #{1} ==========".format(mode_name, epoch)
        print_with_logging(log_writer, epoch_ex_str)
        for i in range(self.num_samples):
            context_sents = metrics[0]["model_output"]["context_sents"][i]
            for turn_id, turn in enumerate(context_sents):
                context_turn_str = "Context Turn #{0}: {1}".format(turn_id, turn)
                print_with_logging(log_writer, context_turn_str)

            generated_sent = metrics[0]["model_output"]["output_sents"][i]
            generated_sent_str = "Generated (Sample #1): {0}".format(generated_sent)
            print_with_logging(log_writer, generated_sent_str)

            if not is_train:
                for j in range(self.num_samples - 1):
                    sampled_sent = metrics[0]["model_output"]["sampled_output_sents"][j][i]
                    sampled_sent_str = "Sample #{0}: {1}".format(j+2, sampled_sent)
                    print_with_logging(log_writer, sampled_sent_str)
                if self.is_test_multi_da:
                    for da in self.da_types:
                        multi_da_result = metrics[0]["model_output"]["ctrl_output_sents"][da][i]
                        multi_da_str = "{0} : {1}".format(da, multi_da_result)
                        print_with_logging(log_writer, multi_da_str)

            real_str = metrics[0]["model_output"]["real_output_sents"][i]
            predicted_da = metrics[0]["model_output"]["output_das"][i]
            real_da = metrics[0]["model_output"]["real_output_das"][i]

            print_with_logging(log_writer, "Real Response: {0}".format(real_str))
            print_with_logging(log_writer, "Predicted DA: {0}".format(predicted_da))
            print_with_logging(log_writer, "Real DA: {0}".format(real_da))
        log_writer.close()

    def report_per_step(
        self,
        metric: dict,
        step: int,
        mode_name: str,
        epoch: Optional[int] = None,
        is_train: Optional[bool] = True,
    ):
        """
        report metric per step during training or evaluation
        :param metric: single metric(dict) from step
        :param step: current step index
        :param mode_name:
        :param epoch: current epoch index
        :param is_train: is it currently train status
        """

    def _set_optimizer(self, model) -> Optimizer:
        """
        setting which optimizer are gonna be used
        :return: optimizer(torch.optim.Optimizer) for training
        """
        learning_rate = self.config["learning_rate"]
        return torch.optim.Adam(model.parameters(), lr=learning_rate)

    def _set_scheduler(self):
        decay_gamma = self.config["learning_decay_rate"]
        decay_step_size = self.config["learning_decay_step"]
        scheduler = StepLR(self.optimizer, step_size=decay_step_size, gamma=decay_gamma)
        return scheduler

    def _set_criterion(self, config) -> Union[nn.Module, Dict[str, nn.Module]]:
        """
        setting which criterion are gonna be used
        :return: single criterion(nn.Module) or multiple criterion with dictionary
        """
        return criterion.CVAELoss(config)

    def update_gradient(self, loss: torch.Tensor):
        """
        1. back-propagation using auto-grad backward()
        2. update model using optimizer step
        :param loss: loss tensor from criterion output
        """
        loss.backward()
        self.optimizer.step()