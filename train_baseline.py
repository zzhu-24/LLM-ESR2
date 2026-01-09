# here put the import lib
import os
import argparse
import torch

from generators.generator import GeneratorAllUser
from generators.generator import Seq2SeqGeneratorAllUser
from generators.bert_generator import BertGeneratorAllUser
from trainers.sequence_trainer import SeqTrainer
from utils.utils import set_seed
from utils.logger import Logger
from models.SASRec import SASRec, SASRec_seq
from models.Bert4Rec import Bert4Rec
from models.GRU4Rec import GRU4Rec, GRU4Rec_seq


parser = argparse.ArgumentParser()

# Required parameters
parser.add_argument("--model_name", 
                    default='sasrec',
                    choices=["sasrec", "bert4rec", "gru4rec"],
                    type=str, 
                    required=False,
                    help="model name")
parser.add_argument("--dataset", 
                    default="yelp", 
                    choices=["yelp", "fashion", "beauty", "beauty2014", "games","toys","sports","appliances","musical"],
                    help="Choose the dataset")
parser.add_argument("--inter_file",
                    default="inter",
                    type=str,
                    help="the name of interaction file")
parser.add_argument("--demo", 
                    default=False, 
                    action='store_true', 
                    help='whether run demo')
parser.add_argument("--output_dir",
                    default='./saved/',
                    type=str,
                    required=False,
                    help="The output directory where the model checkpoints will be written.")
parser.add_argument("--check_path",
                    default='',
                    type=str,
                    help="the save path of checkpoints for different running")
parser.add_argument("--do_test",
                    default=False,
                    action="store_true",
                    help="whether run the test on the well-trained model")
parser.add_argument("--do_emb",
                    default=False,
                    action="store_true",
                    help="save the user embedding derived from the SRS model")
parser.add_argument("--do_group",
                    default=False,
                    action="store_true",
                    help="conduct the group test")
parser.add_argument("--keepon",
                    default=False,
                    action="store_true",
                    help="whether keep on training based on a trained model")
parser.add_argument("--keepon_path",
                    type=str,
                    default="normal",
                    help="the path of trained model for keep on training")
parser.add_argument("--use_seq2seq",
                    default=False,
                    action="store_true",
                    help="whether use seq2seq loss for SASRec and GRU4Rec")
parser.add_argument("--ts_user",
                    type=int,
                    default=10,
                    help="the threshold to split the short and long seq")
parser.add_argument("--ts_item",
                    type=int,
                    default=20,
                    help="the threshold to split the long-tail and popular items")

# Model parameters
parser.add_argument("--hidden_size",
                    default=64,
                    type=int,
                    help="the hidden size of embedding")
parser.add_argument("--trm_num",
                    default=2,
                    type=int,
                    help="the number of transformer layer")
parser.add_argument("--num_heads",
                    default=1,
                    type=int,
                    help="the number of heads in Trm layer")
parser.add_argument("--num_layers",
                    default=1,
                    type=int,
                    help="the number of GRU layers")
parser.add_argument("--dropout_rate",
                    default=0.5,
                    type=float,
                    help="the dropout rate")
parser.add_argument("--max_len",
                    default=200,
                    type=int,
                    help="the max length of input sequence")
parser.add_argument("--mask_prob",
                    type=float,
                    default=0.4,
                    help="the mask probability for training Bert model")
parser.add_argument("--train_neg",
                    default=1,
                    type=int,
                    help="the number of negative samples for training")
parser.add_argument("--test_neg",
                    default=100,
                    type=int,
                    help="the number of negative samples for test")
parser.add_argument("--aug",
                    default=False,
                    action="store_true",
                    help="whether augment the sequence data")
parser.add_argument("--aug_seq",
                    default=False,
                    action="store_true",
                    help="whether use the augmented data")
parser.add_argument("--aug_seq_len",
                    default=0,
                    type=int,
                    help="the augmented length for each sequence")
parser.add_argument("--aug_file",
                    default="inter",
                    type=str,
                    help="the augmentation file name")
parser.add_argument('--enable_id', 
                    default=False,
                    action='store_true', 
                    help='Enable ID in user embeddings')
parser.add_argument("--sim_user_num",
                    default=10,
                    type=int,
                    help="the number of similar users for enhancement")

# Other parameters
parser.add_argument("--train_batch_size",
                    default=512,
                    type=int,
                    help="Total batch size for training.")
parser.add_argument("--lr",
                    default=0.001,
                    type=float,
                    help="The initial learning rate for Adam.")
parser.add_argument("--l2",
                    default=0,
                    type=float,
                    help='The L2 regularization')
parser.add_argument("--num_train_epochs",
                    default=100,
                    type=float,
                    help="Total number of training epochs to perform.")
parser.add_argument("--lr_dc_step",
                    default=1000,
                    type=int,
                    help='every n step, decrease the lr')
parser.add_argument("--lr_dc",
                    default=0,
                    type=float,
                    help='how many learning rate to decrease')
parser.add_argument("--patience",
                    type=int,
                    default=20,
                    help='How many steps to tolerate the performance decrease while training')
parser.add_argument("--watch_metric",
                    type=str,
                    default='NDCG@10',
                    help="which metric is used to select model.")
parser.add_argument('--seed',
                    type=int,
                    default=42,
                    help="random seed for different data split")
parser.add_argument("--no_cuda",
                    action='store_true',
                    help="Whether not to use CUDA when available")
parser.add_argument('--gpu_id',
                    default=0,
                    type=int,
                    help='The device id.')
parser.add_argument('--num_workers',
                    default=0,
                    type=int,
                    help='The number of workers in dataloader')
parser.add_argument("--log", 
                    default=False,
                    action="store_true",
                    help="whether create a new log file")

torch.autograd.set_detect_anomaly(True)

args = parser.parse_args()
set_seed(args.seed) # fix the random seed
args.output_dir = os.path.join(args.output_dir, args.dataset)
args.output_dir = os.path.join(args.output_dir, args.model_name)
args.keepon_path = os.path.join(args.output_dir, args.keepon_path)
args.output_dir = os.path.join(args.output_dir, args.check_path)    # if check_path is none, then without check_path


class BaselineTrainer(SeqTrainer):
    """Trainer for baseline models (SASRec, Bert4Rec, GRU4Rec)"""
    
    def __init__(self, args, logger, writer, device, generator):
        # Initialize parent class but override model creation
        self.args = args
        self.logger = logger
        self.writer = writer
        self.device = device
        self.user_num, self.item_num = generator.get_user_item_num()
        self.start_epoch = 0

        self.logger.info('Loading Model: ' + args.model_name)
        self._create_model()
        from utils.utils import get_n_params
        logger.info('# of model parameters: ' + str(get_n_params(self.model)))

        self._set_optimizer()
        self._set_scheduler()
        self._set_stopper()

        if args.keepon:
            self._load_pretrained_model()

        self.loss_func = torch.nn.BCEWithLogitsLoss()
        
        self.train_loader = generator.make_trainloader()
        self.valid_loader = generator.make_evalloader()
        self.test_loader = generator.make_evalloader(test=True)
        self.generator = generator

        # get item pop and user len
        self.item_pop = generator.get_item_pop()
        self.user_len = generator.get_user_len()

        self.watch_metric = args.watch_metric
        self.enable_id = False  # baseline models don't use ID
    
    def _create_model(self):
        '''create baseline model'''
        if self.args.model_name == "sasrec":
            if self.args.use_seq2seq:
                self.model = SASRec_seq(self.user_num, self.item_num, self.device, self.args)
            else:
                self.model = SASRec(self.user_num, self.item_num, self.device, self.args)
        elif self.args.model_name == "bert4rec":
            self.model = Bert4Rec(self.user_num, self.item_num, self.device, self.args)
        elif self.args.model_name == "gru4rec":
            if self.args.use_seq2seq:
                self.model = GRU4Rec_seq(self.user_num, self.item_num, self.device, self.args)
            else:
                self.model = GRU4Rec(self.user_num, self.item_num, self.device, self.args)
        else:
            raise ValueError(f"Unknown model name: {self.args.model_name}")
        
        self.model.to(self.device)
    
    def _set_optimizer(self):
        self.optimizer = torch.optim.Adam(self.model.parameters(), 
                                          lr=self.args.lr,
                                          weight_decay=self.args.l2)
    
    def _set_scheduler(self):
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer,
                                                         step_size=self.args.lr_dc_step,
                                                         gamma=self.args.lr_dc)
    
    def _set_stopper(self):
        from utils.earlystop import EarlyStoppingNew
        self.stopper = EarlyStoppingNew(patience=self.args.patience, 
                                     verbose=False,
                                     path=self.args.output_dir,
                                     trace_func=self.logger)
    
    def _load_pretrained_model(self):
        """Load pretrained model for continue training"""
        self.logger.info("Loading the trained model for keep on training ... ")
        checkpoint_path = os.path.join(self.args.keepon_path, 'pytorch_model.bin')

        model_dict = self.model.state_dict()
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        pretrained_dict = checkpoint['state_dict']

        # filter out required parameters
        new_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict.keys()}
        model_dict.update(new_dict)
        # Print: how many parameters are loaded from the checkpoint
        self.logger.info('Total loaded parameters: {}, update: {}'.format(len(pretrained_dict), len(new_dict)))
        self.model.load_state_dict(model_dict)  # load model parameters
        if 'optimizer' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer']) # load optimizer
        if 'scheduler' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler']) # load scheduler
        if 'epoch' in checkpoint:
            self.start_epoch = checkpoint['epoch']  # load epoch


def main():

    log_manager = Logger(args)  # initialize the log manager
    logger, writer = log_manager.get_logger()    # get the logger
    args.now_str = log_manager.get_now_str()

    device = torch.device("cuda:"+str(args.gpu_id) if torch.cuda.is_available()
                          and not args.no_cuda else "cpu")


    os.makedirs(args.output_dir, exist_ok=True)

    # generator is used to manage dataset
    if args.model_name == 'gru4rec':
        if args.use_seq2seq:
            generator = Seq2SeqGeneratorAllUser(args, logger, device)
        else:
            generator = GeneratorAllUser(args, logger, device)
    elif args.model_name == "bert4rec":
        generator = BertGeneratorAllUser(args, logger, device)
    elif args.model_name == "sasrec":
        if args.use_seq2seq:
            generator = Seq2SeqGeneratorAllUser(args, logger, device)
        else:
            # Use base Generator for point-wise loss (not GeneratorAllUser)
            from generators.generator import Generator
            generator = Generator(args, logger, device)
    else:
        raise ValueError(f"Unknown model name: {args.model_name}")

        

    trainer = BaselineTrainer(args, logger, writer, device, generator)

    if args.do_test:
        trainer.test()
    elif args.do_emb:
        trainer.save_user_emb()
    elif args.do_group:
        trainer.test_group()
    else:
        trainer.train()

    log_manager.end_log()   # delete the logger threads



if __name__ == "__main__":

    main()
