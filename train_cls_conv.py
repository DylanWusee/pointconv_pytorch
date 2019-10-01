import argparse
import os
import torch
import torch.nn.parallel
import torch.utils.data
import torch.nn.functional as F
from data_utils.ModelNetDataLoader import ModelNetDataLoader, load_data
import datetime
import logging
from pathlib import Path
from tqdm import tqdm
from utils.utils import test, save_checkpoint
from model.pointconv import PointConvDensityClsSsg2 as PointConvClsSsg


def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('PointConv')
    parser.add_argument('--batchsize', type=int, default=16, help='batch size in training')
    parser.add_argument('--epoch',  default=400, type=int, help='number of epoch in training')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='learning rate in training')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--train_metric', type=str, default=False, help='whether evaluate on training dataset')
    parser.add_argument('--optimizer', type=str, default='SGD', help='optimizer for training')
    parser.add_argument('--pretrain', type=str, default=None,help='whether use pretrain model')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='decay rate of learning rate')
    parser.add_argument('--model_name', default='pointconv', help='model name')
    return parser.parse_args()

def main(args):
    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    datapath = './data/ModelNet/'

    '''CREATE DIR'''
    experiment_dir = Path('./experiment/')
    experiment_dir.mkdir(exist_ok=True)
    file_dir = Path(str(experiment_dir) + '/%sModelNet40-'%args.model_name + str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')))
    file_dir.mkdir(exist_ok=True)
    checkpoints_dir = file_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir = file_dir.joinpath('logs/')
    log_dir.mkdir(exist_ok=True)

    '''LOG'''
    args = parse_args()
    logger = logging.getLogger(args.model_name)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler(str(log_dir) + 'train_%s_cls.txt'%args.model_name)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.info('---------------------------------------------------TRANING---------------------------------------------------')
    logger.info('PARAMETER ...')
    logger.info(args)

    '''DATA LOADING'''
    logger.info('Load dataset ...')
    train_data, train_label, test_data, test_label = load_data(datapath, classification=True)
    logger.info("The number of training data is: %d",train_data.shape[0])
    logger.info("The number of test data is: %d", test_data.shape[0])
    trainDataset = ModelNetDataLoader(train_data, train_label)
    testDataset = ModelNetDataLoader(test_data, test_label)
    trainDataLoader = torch.utils.data.DataLoader(trainDataset, batch_size=args.batchsize, shuffle=True)
    testDataLoader = torch.utils.data.DataLoader(testDataset, batch_size=args.batchsize, shuffle=False)

    '''MODEL LOADING'''
    num_class = 40
    classifier = PointConvClsSsg(num_class).cuda()
    if args.pretrain is not None:
        print('Use pretrain model...')
        logger.info('Use pretrain model')
        checkpoint = torch.load(args.pretrain)
        start_epoch = checkpoint['epoch']
        classifier.load_state_dict(checkpoint['model_state_dict'])
    else:
        print('No existing model, starting training from scratch...')
        start_epoch = 0


    if args.optimizer == 'SGD':
        optimizer = torch.optim.SGD(classifier.parameters(), lr=0.01, momentum=0.9)
    elif args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(
            classifier.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=args.decay_rate
        )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)
    global_epoch = 0
    global_step = 0
    best_tst_accuracy = 0.0
    blue = lambda x: '\033[94m' + x + '\033[0m'

    '''TRANING'''
    logger.info('Start training...')
    for epoch in range(start_epoch,args.epoch):
        print('Epoch %d (%d/%s):' % (global_epoch + 1, epoch + 1, args.epoch))
        logger.info('Epoch %d (%d/%s):' ,global_epoch + 1, epoch + 1, args.epoch)

        scheduler.step()
        for batch_id, data in tqdm(enumerate(trainDataLoader, 0), total=len(trainDataLoader), smoothing=0.9):
            points, target = data
            target = target[:, 0]
            points = points.transpose(2, 1)
            points, target = points.cuda(), target.cuda()
            optimizer.zero_grad()
            classifier = classifier.train()
            pred = classifier(points)
            loss = F.nll_loss(pred, target.long())

            loss.backward()
            optimizer.step()
            global_step += 1

        train_acc = test(classifier.eval(), trainDataLoader) if args.train_metric else None
        acc = test(classifier, testDataLoader)


        print('\r Loss: %f' % loss.data)
        logger.info('Loss: %.2f', loss.data)
        if args.train_metric:
            print('Train Accuracy: %f' % train_acc)
            logger.info('Train Accuracy: %f', (train_acc))
        print('\r Test %s: %f   ***  %s: %f' % (blue('Accuracy'),acc, blue('Best Accuracy'),best_tst_accuracy))
        logger.info('Test Accuracy: %f  *** Best Test Accuracy: %f', acc, best_tst_accuracy)

        if (acc >= best_tst_accuracy) and epoch > 5:
            best_tst_accuracy = acc
            logger.info('Save model...')
            save_checkpoint(
                global_epoch + 1,
                train_acc if args.train_metric else 0.0,
                acc,
                classifier,
                optimizer,
                str(checkpoints_dir),
                args.model_name)
            print('Saving model....')
        global_epoch += 1
    print('Best Accuracy: %f'%best_tst_accuracy)

    logger.info('End of training...')

if __name__ == '__main__':
    args = parse_args()
    main(args)
