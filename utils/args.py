import os
import torch
import warnings
import argparse
from abc import ABC, abstractmethod


class ArgumentsManager(ABC):
    r"""This class defines options used during both training and test time.

    It also implements several helper functions such as parsing, printing, and saving the options.

    """

    def __init__(self):
        self.parser = argparse.ArgumentParser(description=__doc__)
        self.args = None
        self.initialized = False

        self.parser.add_argument(
            'experiment',
            help='name of experiment, possible are:'
                 '{"mnist", cifar10", "cifar100", "femnist", "shakespeare"}',
            type=str
        )
        self.parser.add_argument(
            '--model_name',
            help='the name of the model to be used, only used when experiment is CIFAR-10, CIFAR-100 or FEMNIST'
                 'possible are {"mobilenet"}, default is mobilenet',
            type=str,
            default="mobilenet"
        )
        self.parser.add_argument(
            '--bz',
            help='batch_size; default is 1',
            type=int,
            default=1
        )
        self.parser.add_argument(
            '--device',
            help='device to use, either cpu or cuda; default is cpu',
            type=str,
            default="cpu"
        )
        self.parser.add_argument(
            "--input_dimension",
            help='input dimension ofr two layers linear model',
            type=int,
            default=150
        )
        self.parser.add_argument(
            "--hidden_dimension",
            help='hidden dimension for two layers linear model',
            type=int,
            default=10
        )
        self.parser.add_argument(
            "--seed",
            help='random seed; if not specified the system clock is used to generate the seed',
            type=int,
            default=argparse.SUPPRESS
        )

    def parse_arguments(self, args_list=None):
        if args_list:
            args = self.parser.parse_args(args_list)
        else:
            args = self.parser.parse_args()

        self.args = args

        if self.args.device == "cuda" and not torch.cuda.is_available():
            self.args.device = "cpu"
            warnings.warn("CUDA is not available, device is automatically set to \"CPU\"!", RuntimeWarning)

        self.initialized = True

    @abstractmethod
    def args_to_string(self):
        pass


class TrainArgumentsManager(ArgumentsManager):
    def __init__(self):
        super(TrainArgumentsManager, self).__init__()

        self.parser.add_argument(
            '--aggregator_type',
            help='aggregator type; possible are "centralized"',
            type=str,
            default="centralized"
        )
        self.parser.add_argument(
            "--participation_probs",
            nargs="+",
            help="list of probabilities controlling the participation of clients from each group;"
                 "should be a list of the same size as `n_groups`;"
                 "default is [1.0, 0.1]",
            type=float,
            default=[1.0, 0.1]
        )
        self.parser.add_argument(
            '--unknown_participation_probs',
            help='if True, client participation probabilities are estimated through FedAU; default is "False"',
            action='store_true'
        )
        self.parser.add_argument(
            '--n_rounds',
            help='number of communication rounds; default is 1',
            type=int,
            default=1
        )
        self.parser.add_argument(
            '--local_steps',
            help='number of local steps before communication; default is 1',
            type=int,
            default=1
        )
        self.parser.add_argument(
            '--log_freq',
            help='frequency of writing logs; defaults is 1',
            type=int,
            default=1
        )
        self.parser.add_argument(
            '--optimizer',
            help='optimizer to be used for the training; default is sgd',
            type=str,
            default="sgd"
        )
        self.parser.add_argument(
            '--server_optimizer',
            help='optimizer to be used for the training; default is sgd',
            type=str,
            default="sgd"
        )
        self.parser.add_argument(
            "--lr",
            type=float,
            help='learning rate; default is 1e-3',
            default=1e-3
        )
        self.parser.add_argument(
            "--server_lr",
            type=float,
            help='learning rate; default is 1.',
            default=1.
        )
        self.parser.add_argument(
            "--lr_scheduler",
            help='learning rate decay scheme to be used;'
                 ' possible are "sqrt", "linear", "cosine_annealing" and "constant"(no learning rate decay);'
                 'default is "constant"',
            type=str,
            default="constant"
        )
        self.parser.add_argument(
            "--server_lr_scheduler",
            help='learning rate decay scheme to be used;'
                 ' possible are "sqrt", "linear", "cosine_annealing" and "constant"(no learning rate decay);'
                 'default is "constant"',
            type=str,
            default="constant"
        )
        self.parser.add_argument(
            "--mu",
            help='proximal / penalty term weight, used when --optimizer=`prox_sgd` also used with L2SGD; '
                 'default is `0.`',
            type=float,
            default=0
        )
        self.parser.add_argument(
            "--history_coefficient",
            help='history term weight, used when --optimizer=`history`; '
                 'default is `1.`',
            type=float,
            default=1.
        )
        self.parser.add_argument(
            '--validation',
            help='if chosen the validation part will be used instead of test part;'
                 ' make sure to use `val_frac > 0` in `generate_data.py`;',
            action='store_true'
        )
        self.parser.add_argument(
            "--logs_dir",
            help='directory to write logs; if not passed, it is set using arguments',
            default=argparse.SUPPRESS
        )
        self.parser.add_argument(
            "--chkpts_dir",
            help='directory to save checkpoints once the training is over; if not specified checkpoints are not saved',
            default=argparse.SUPPRESS
        )
        self.parser.add_argument(
            "--verbose",
            help='verbosity level, `0` to quiet, `1` to show global logs and `2` to show local logs; default is `0`;',
            type=int,
            default=0
        )
        self.parser.add_argument(
            "--availability_matrix_path",
            help='path to the availability matrix',
            type=str,
            default='data_availability/availability_matrix.csv'
        )

    def args_to_string(self):
        """
        Transform experiment's arguments into a string

        :return: string

        """
        args_string = ""

        args_to_show = ["experiment"]
        for arg in args_to_show:
            args_string = os.path.join(args_string, str(getattr(self.args, arg)))

        return args_string


