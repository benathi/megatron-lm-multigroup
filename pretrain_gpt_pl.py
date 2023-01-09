# (c) 2022
# AWS CodeWhisperer
#
#

import torch
import json
import sys
import os
import time
from collections import Counter
from typing import cast, Dict, List, Optional, Union
import pytorch_lightning as pl
from megatron import (
    initialize_megatron,
    get_args,
    print_rank_0,
    get_timers,
)
from megatron.training import (
    print_datetime,
    setup_model_and_optimizer,
    build_train_valid_test_data_iterators,
    evaluate_and_print_results
)
from megatron.training import train as megatron_train
from megatron.utils import (
    found_kill_switch,
    get_parameters_in_billions,
)
from megatron.checkpointing import (
    save_checkpoint
)
from pretrain_gpt import (
    model_provider,
    train_valid_test_datasets_provider,
    forward_step,
)
from megatron.data.dataset_utils import (
    analyze_data_prefix
)
#from megatron.lightning_utils import (
#    MStarEKSLogger
#)


try:
    from torch.distributed.elastic.multiprocessing.errors import record
except ImportError:
    # noop
    def record(fn):
        return fn


_TRAIN_START_TIME = time.time()


class PlModel(pl.LightningModule):
    def __init__(
            self,
    ):
        super().__init__()
        self.automatic_optimization = False # for custom loop
        # Initalize and get arguments, timers, and Tensorboard writer.
        args = get_args()
        self.args = args

        if found_kill_switch():
            print_datetime(f"Detected kill switch at {args.kill_switch_path}. Exiting")
            sys.exit()

        self.mark_start_time()
        self.timers = get_timers()
        if args.deepspeed:
            args.deepspeed_configuration = json.load(
                open(args.deepspeed_config, 'r', encoding='utf-8'))

        # Model, optimizer, and learning rate.
        self.timers('model-and-optimizer-setup').start()
        self.model, self.optimizer, self.lr_scheduler = setup_model_and_optimizer(model_provider)
        args.parameters_in_billions_no_embedding = get_parameters_in_billions(self.model, exclude_embeddings=True)
        print_rank_0(f'estimated model parameters: {get_parameters_in_billions(self.model)}')
        print_rank_0(
            f'estimated model parameters without embeddings: {get_parameters_in_billions(self.model, exclude_embeddings=True)}')
        self.timers('model-and-optimizer-setup').stop()
        print_datetime('after model, optimizer, and learning rate '
                       'scheduler are built')

        self.total_loss_dict = {}
        self.timers('train/valid/test-data-iterators-setup').start()

        if self.args.virtual_pipeline_model_parallel_size is None:
            self.train_data_iterator, self.valid_data_iterator, self.test_data_iterator = build_train_valid_test_data_iterators(
                train_valid_test_datasets_provider)
        else:
            # rare case when we use virtual_pipeline_model_parallel_size
            all_data_iterators = [
                build_train_valid_test_data_iterators(train_valid_test_datasets_provider)
                for _ in range(self.model)
            ]
            self.train_data_iterator = [data_iterators[0] for data_iterators in all_data_iterators]
            self.valid_data_iterator = [data_iterators[1] for data_iterators in all_data_iterators]
            self.test_data_iterator = [data_iterators[2] for data_iterators in all_data_iterators]

        if args.data_path is not None and len(args.data_path) > 1:
            prefixes, weights = analyze_data_prefix(args.data_path)
            setattr(args, "data_prefixes", prefixes)
            setattr(args, "data_weights", weights)
        elif args.train_weighted_split_paths is not None and len(args.train_weighted_split_paths[0]) > 1:
            paths = args.train_weighted_split_paths[0]
            weights = args.train_weighted_split_weights[0]
            data_prefix = [j for i in [[w, p] for w, p in zip(weights, paths)] for j in i]
            prefixes, weights = analyze_data_prefix(data_prefix)
            setattr(args, "data_prefixes", prefixes)
            setattr(args, "data_weights", weights)
        else:
            setattr(args, "data_prefixes", None)
            setattr(args, "data_weights", None)

        self.timers('train/valid/test-data-iterators-setup').stop()
        print_datetime('after dataloaders are built')

        # Print setup timing.
        print_rank_0('done with setup ...')
        self.timers.log(['model-and-optimizer-setup', 'train/valid/test-data-iterators-setup'])

    def mark_start_time(self):
        # Adjust the startup time so it reflects the largest value.
        # This will be closer to what scheduler will see (outside of
        # image ... launches.
        global _TRAIN_START_TIME
        start_time_tensor = torch.cuda.FloatTensor([_TRAIN_START_TIME])
        torch.distributed.all_reduce(start_time_tensor,
                                     op=torch.distributed.ReduceOp.MIN)
        _TRAIN_START_TIME = start_time_tensor.item()
        print_rank_0('time to initialize megatron (seconds): {:.3f}'.format(
            time.time() - _TRAIN_START_TIME))
        print_datetime('after megatron is initialized')
        self.timers = get_timers()


# define custom PlTrainer
class CustomTrainer(pl.Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.args = get_args()
        # self.setup_mstar_eks_logger()

    def setup_mstar_eks_logger(self):
        if "KUBERNETES_SERVICE_HOST" in os.environ:
            if not self.args.run_name:
                run_name = "{}-input-{}".format(
                    self.args.model_type, self.args.max_seq_length
                )
            else:
                run_name = self.args.run_name
            eks_logger = MStarEKSLogger(
                experiment_name="gpt2-experiment",
                run_name=run_name,
                tags={"mode": "Training"},
            )
            # TODO -- consume this logger

    def reset_port(self):
        port = str(1337)
        if "MASTER_PORT" in os.environ:
            if os.environ["MASTER_PORT"] != port:
                print("Setting master port to", port)
                os.environ["MASTER_PORT"] = port

    def fit(self, module, datamodule=None):
        print("Perform training @ fit function")
        print_rank_0('training ...')
        #self.reset_port()
        self.iteration = 0
        # train
        if self.args.do_train and self.args.train_iters > 0:
            self.iteration = megatron_train(forward_step, module.model, module.optimizer, module.lr_scheduler,
                       module.train_data_iterator, module.valid_data_iterator)
        print_datetime('after training is done')

        # eval after training
        print(f"args.do_valid = {self.args.do_valid} args.do_test = {self.args.do_test}")
        if self.args.do_valid:
            self.eval(module, module.valid_data_iterator, is_test=False, iteration=self.iteration)
        self.save(module)
        if self.args.do_test:
            self.eval(module, module.test_data_iterator, is_test=True, iteration=0)

    def eval(self, module, valid_data_iterator, is_test, iteration):

        names = self.args.valid_weighted_split_names
        names = names if names is not None else ['valid'] * len(valid_data_iterator)
        for iterator, name in zip(valid_data_iterator, names):
            prefix = 'the end of training for val data'
            evaluate_and_print_results(prefix, forward_step,
                                       iterator, module.model,
                                       iteration, is_test, data_group_name=name)

    def save(self, module):
        if self.args.save and self.iteration != 0:
            save_checkpoint(self.iteration, module.model, module.optimizer, module.lr_scheduler)

@record
def pl_main():
    print("initializing megatron")
    # TODO -- change option to configure tokenizer
    initialize_megatron(
        args_defaults={'tokenizer_type': 'GPT2BPETokenizer'}
    )
    module = PlModel()
    trainer = CustomTrainer()
    trainer.fit(module)


if __name__ == "__main__":
    pl_main()
