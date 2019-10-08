from abc import ABC, abstractmethod
from typing import Dict, Iterable, List, Optional, Union

from datetime import datetime

import os
import tqdm
import torch
import torch.nn as nn
import time
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader


class Trainer(ABC):
    def __init__(self, config, model: nn.Module):
        """
        wrapped class of pytorch model training process
        :param config: config object for training
        :param model: main model for training
        """
        self.config = config
        self.model = model
        self.da_types = model.da_vocab
        loss_config = self.config["loss"]
        self.criterion: Union[nn.Module, Dict[str, nn.Module]] = self._set_criterion(loss_config)

        self.train_outputs: List[str] = config["train_output"]
        self.valid_outputs: List[str] = config["valid_output"]
        self.test_outputs: List[str] = config["test_output"]

        self.num_samples: int = config["num_samples"]

        self.optimizer: Optimizer = self._set_optimizer(model)
        self.scheduler = self._set_scheduler()
        self.is_learning_decay: bool = config["is_learning_decay"]
        self.is_valid_true: bool = config["is_valid_train"]
        self.is_test_multi_da = config["is_test_multi_da"]

        self.save_epoch_step = config["save_epoch_step"]

        output_dir_path = config["output_dir_path"]
        log_dirname = config["log_dirname"]
        model_dirname = config["model_dirname"]
        test_dirname = config["test_dirname"]

        self.log_dir_path = os.path.join(output_dir_path, log_dirname)
        self.model_dir_path = os.path.join(output_dir_path, model_dirname)
        self.test_dir_path = os.path.join(output_dir_path, test_dirname)
        dir_paths = [output_dir_path, self.log_dir_path, self.model_dir_path, self.test_dir_path]
        for dir_path in dir_paths:
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)

        self.model_path = os.path.join(self.model_dir_path, config["model_name"])
        self.log_path = os.path.join(self.log_dir_path, config["log_name"])

    def experiment(
        self,
        train_data_loader: Union[DataLoader, Iterable],
        valid_data_loader: Optional[Union[DataLoader, Iterable]] = None,
        test_data_loader: Optional[Union[DataLoader, Iterable]] = None,
        epoch_start_point: int = 0
    ) -> List[dict]:
        """
        run multiple epochs training with train_data_loader batch input
        input model forward. if eval_data_loader is exist, then run evaluation
        when every epoch finished.
        config.train.epochs is require to run this process. if not exist,
        then run with epochs=1
        :param train_data_loader: train step batch data loader
        :param valid_data_loader: if valid_data_loader given, run validation process
        :param test_data_loader: if test_data_loader given, run test process
        :param epoch_start_point: if start point
        :return: results(dict) of total training process
        """

        output_reports = []

        try:
            epochs = self.config["epoch"]
        except AttributeError:
            epochs = 1

        exp_epoch_start_point = 0
        if epoch_start_point > 0:
            model_path = self.model_path.format(epoch_start_point)
            # Check if parameter available
            if os.path.exists(model_path):
                self.model.load_state_dict(torch.load(model_path))
                device = torch.device("cuda")
                self.model.to(device)
                exp_epoch_start_point = epoch_start_point + 1
            else:
                exp_epoch_start_point = 0

        for epoch in range(exp_epoch_start_point, epochs):  # type: int
            print(datetime.today().strftime('%Y-%m-%d-%H:%M:%S'))
            start_time = time.time()
            train_output = self.train_one_epoch(train_data_loader, epoch)
            end_time = time.time()

            elapsed_time = end_time - start_time
            self.report_per_epoch(train_output, "Train", epoch, elapsed_time, is_train=True)
            output_report = {"train": train_output}

            if valid_data_loader:
                start_time = time.time()
                valid_output = self.valid(valid_data_loader, epoch=epoch)
                end_time = time.time()

                elapsed_time = end_time - start_time
                self.report_per_epoch(valid_output, "Valid", epoch, elapsed_time, is_train=False)
                output_report["valid"] = valid_output

            if test_data_loader:
                start_time = time.time()
                test_output = self.test(test_data_loader, epoch=epoch)
                end_time = time.time()

                elapsed_time = end_time - start_time
                self.report_per_epoch(test_output, "Test", epoch, elapsed_time, is_train=False)
                output_report["test"] = test_output

            output_reports.append(output_report)
            if epoch % self.save_epoch_step == 0:
                torch.save(self.model.state_dict(), self.model_path.format(epoch))

        return output_reports

    def _run_one_epoch(
        self,
        step_function,
        data_loader: Union[DataLoader, Iterable],
        mode: str,
        epoch: Optional[int] = None,
        is_train: bool = False,
    ) -> List[dict]:
        """
        run one-epoch with step_function(could be train, eval step) and data_loader.
        1. loading batch data from data loader
        2. call step_function() with batch data
        3. report step training output
        4. run until the data_loader is finished
        :param step_function: train or eval step function which takes model_input args
        :param data_loader: batch data loader for model forward
        :param epoch: indicate current epoch index
        :param is_train: pass to report about current epoch
        :return: return of step_outputs
        """

        step_outputs = []

        iterator = tqdm.tqdm(data_loader, desc=mode)
        current_step_num = epoch * len(iterator)

        for step_id, model_input in enumerate(iterator):
            current_step_num = current_step_num + 1
            step_output = step_function(model_input, current_step_num)
            self.report_per_step(step_output, step_id, mode, epoch, is_train=is_train)
            step_outputs.append(step_output)

        if is_train and self.is_learning_decay:
            self.scheduler.step()

        return step_outputs

    def valid(
        self, data_loader: Union[DataLoader, Iterable], epoch: Optional[int] = None
    ) -> List[dict]:
        """
        run evaluation process with data_loader batch data input.
        it does not run multiple epochs.
        :param data_loader: batch data loader for model forward
        :param epoch: optional, current eval epoch
        :return: results of evaluation from data_loader
        """

        return self._run_one_epoch(self.valid_step, data_loader, "valid", epoch, False)

    def test(
        self, data_loader: Union[DataLoader, Iterable], epoch: Optional[int] = None
    ) -> List[dict]:
        """
        run evaluation process with data_loader batch data input.
        it does not run multiple epochs.
        :param data_loader: batch data loader for model forward
        :param epoch: optional, current eval epoch
        :return: results of evaluation from data_loader
        """

        return self._run_one_epoch(self.test_step, data_loader, "test", epoch, False)

    def train_one_epoch(
        self, data_loader: Union[DataLoader, Iterable], epoch: Optional[int] = None,

    ) -> List[dict]:
        """
        one-epoch training process with batch data input
        :param data_loader: batch data loader for model forward
        :param epoch: optional, current train epoch
        :return: results of one epoch training
        """

        return self._run_one_epoch(self.train_step, data_loader, "train", epoch, True)

    @abstractmethod
    def get_step_metric(
        self,
        model_output: dict,
        model_input,
        loss: Optional[Union[float, torch.Tensor]] = None,
    ) -> dict:
        """
        calculate metrics from step output
        :param model_output: forward output from model
        :param model_input: model input which feed into forward method
        :param loss: loss score from criterion
        :return: dictionary of metrics
        """

        raise NotImplementedError

    @abstractmethod
    def get_epoch_metric(self, epoch_output: List[dict]) -> dict:
        """
        calculate metrics from epoch step outputs
        :param epoch_output: list of each step output (dict)
        :return: dictionary of metrics
        """

        raise NotImplementedError

    @abstractmethod
    def calculate_loss(
        self,
        model_output,
        model_input,
        current_step,
        is_train,
        is_valid
    ):
        """
        calculate the loss using criterion with model_output, model_input
        :param model_output: forward output from model
        :param model_input: model input which feed into forward method
        :param current_step: current step.
        :param is_train: if model is in train_mode or not.
        :param is_valid: if model is in valid_mode or not.
        :return: loss score
        """
        raise NotImplementedError

    @abstractmethod
    def report(self, metrics: List[dict], is_train: Optional[bool] = True):
        """
        report metrics during total training or evaluation process
        :param metrics: list of metric(dict) from process
        :param is_train: is it currently train status
        """
        raise NotImplementedError

    @abstractmethod
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
        raise NotImplementedError

    @abstractmethod
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
        :param mode_name: a name of report set (train, test, valid...)
        :param epoch: current epoch index
        :param is_train: is it currently train status
        """
        raise NotImplementedError

    @abstractmethod
    def _set_optimizer(self, model) -> Optimizer:
        """
        setting which optimizer are gonna be used
        :return: optimizer(torch.optim.Optimizer) for training
        """
        raise NotImplementedError

    @abstractmethod
    def _set_scheduler(self):
        """
        setting which scheduler are gonna be used.
        :return:
        """
        raise NotImplementedError

    @abstractmethod
    def _set_criterion(self, loss_config) -> Union[nn.Module, Dict[str, nn.Module]]:
        """
        setting which criterion are gonna be used
        :return: single criterion(nn.Module) or multiple criterion with dictionary
        """
        raise NotImplementedError

    def update_gradient(self, loss: torch.Tensor):
        """
        1. back-propagation using auto-grad backward()
        2. update model using optimizer step
        :param loss: loss tensor from criterion output
        """
        loss.backward()
        self.optimizer.step()

    def train_step(self, model_input, current_step) -> dict:
        """
        single training step with batch-input
        :param model_input: batch data input to model forward
        :param current_step: current_step.
        :return: return model output and loss as dictionary
        """
        self.optimizer.zero_grad()
        model_input["is_train"] = True
        model_input["num_samples"] = self.num_samples
        model_output = self.model.forward(model_input)

        targets = self.train_outputs

        recorded_model_output = {target: model_output[target] for target in targets}
        loss = self.calculate_loss(model_output, model_input, current_step, True, False)
        self.update_gradient(loss["main_loss"])

        return {"model_output": recorded_model_output, "loss": loss}

    def valid_step(self, model_input, current_step) -> dict:
        """
        single evaluation step with batch-input
        :param model_input: batch data input to model forward
        :param current_step: current_step.
        :return: return model output and loss as dictionary
        """
        model_input["is_train"] = self.is_valid_true
        model_input["is_train_multiple"] = True
        model_input["is_test_multi_da"] = self.is_test_multi_da
        model_input["num_samples"] = self.num_samples

        targets = self.valid_outputs
        with torch.no_grad():
            model_output = self.model.forward(model_input)
            loss = self.calculate_loss(model_output, model_input, current_step, False, True)
        recorded_model_output = {target: model_output[target] for target in targets}

        return {"model_output": recorded_model_output, "loss": loss}

    def test_step(self, model_input, current_step) -> dict:
        """
        single evaluation step with batch-input
        :param model_input: batch data input to model forward
        :param current_step: current_step.
        :return: return model output and loss as dictionary
        """
        model_input["is_train"] = False
        model_input["is_test_multi_da"] = self.is_test_multi_da
        model_input["num_samples"] = self.num_samples

        targets = self.test_outputs

        with torch.no_grad():
            model_output = self.model.forward(model_input)
            loss = self.calculate_loss(model_output, model_input, current_step, False, False)
        recorded_model_output = {target: model_output[target] for target in targets}

        return {"model_input": model_input, "model_output": recorded_model_output, "loss": loss}
