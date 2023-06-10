import torch
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--selector', type=str, default="WENGSYX/Deberta-Chinese-Large")
parser.add_argument('--lr_sel', type=float, default=2.2e-5)
parser.add_argument('--bsz_sel', type=int, default=12)
parser.add_argument('--sd_sel', type=int, default=52)

parser.add_argument('--generator', type=str, default="fnlp/cpt-large")
parser.add_argument('--lr_gen', type=float, default=2.1e-5)
parser.add_argument('--bsz_gen', type=int, default=8)
parser.add_argument('--sd_gen', type=int, default=44)
parser.add_argument('--num_beams_gen', type=int, default=1)
parser.add_argument('--do_sample', action='store_true', default=False)

args = parser.parse_args()

data_dir = '/remote-home/cktan/space/qlora/SpaCE2022/processed_data'
jsonl_dir = '/jsonl'
train_dir = data_dir + '/processed_task2_train.jsonl'
dev_dir = data_dir +  '/processed_task2_dev.jsonl'
test_dir = data_dir +  '/task3_test.jsonl'

# pretrained_model_dir = '../pretrained_model/'
selector_name, generator_name = args.selector, args.generator
selector_dir, generator_dir =  selector_name,  generator_name

# selector
selector_params = {
    'lr': args.lr_sel,
    'batch_size': args.bsz_sel,
    'seed': args.sd_sel,
    'weight_decay': 0.01,
    'clip_grad': 5,
    'epoch': 30,
    'patience': 0.0002,
    'patience_num': 5,
    'min_epoch_num': 5
}
# generator
generator_params = {
    'lr': args.lr_gen,
    'batch_size': args.bsz_gen,
    'seed': args.sd_gen,
    'weight_decay': 0.01,
    'num_beams': args.num_beams_gen,
    'do_sample': args.do_sample,
    'clip_grad': 5,
    'epoch': 50,
    'patience': 0.0002,
    'patience_num': 5,
    'min_epoch_num': 5
}
# device
device = torch.device('cuda')

selector_model_dir = '/remote-home/cktan/space/SpaCE2022/final/experiments/' + selector_name \
            + '_lr_' + str(args.lr_sel) + '_bsz_' + str(args.bsz_sel) + '_sd_' + str(args.sd_sel)
generator_model_dir = './experiments/' + generator_name \
            + '_lr_' + str(args.lr_gen) + '_bsz_' + str(args.bsz_gen) + '_sd_' + str(args.sd_gen)
generation_model_dir = './experiments/deberata_cpt'

selector_log_dir = selector_model_dir + '/train.log'
generator_log_dir = generator_model_dir + '/train.log'
generation_log_dir = generation_model_dir + '/train.log'
prediction_dir = generation_model_dir + '/prediction_test.jsonl'

