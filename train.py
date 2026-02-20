import random
import torch
import torch.nn as nn
import torch.utils.data as Data
from tensorboardX import SummaryWriter
from argparse import ArgumentParser
from tqdm import tqdm
import os
import os.path as ops
import numpy as np
import time

from utils.Adan import Adan
from utils.data import SirstDataset, IRSTD1K_Dataset, NUDT_Dataset
from utils.lr_scheduler import adjust_learning_rate, create_lr_scheduler

from model.loss import SoftIoULoss, criterion
from model.metrics import IoUMetric, nIoUMetric, PD_FA
from model.FCCFNet_L import FCCFNet_L
from model.FCCFNet_M import FCCFNet_M
from model.FCCFNet_S import FCCFNet_S

from torch.amp import autocast, GradScaler

#原实验参数 batchSize = 8 lr =0.001

def parse_args():
    parser = ArgumentParser(description='Implement of MY model')

    parser.add_argument('--img_size', type=int, default=128, help='image size')
    parser.add_argument('--batch_size', type=int, default=8, help='batch_size for training')
    parser.add_argument('--epochs', type=int, default=600, help='number of epochs')
    parser.add_argument('--warm_up_epochs', type=int, default=10, help='warm up epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate')

    parser.add_argument('--dataset', type=str, default='sirst', help='datasets: sirst or IRSTD-1k or NUDT-SIRST')
    parser.add_argument('--mode', type=str, default='L', help='mode: L, M, S')

    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=1, type=int, help='start epoch')

    parser.add_argument('--amp', default=True, help='Use torch.cuda.amp for mixed precision training')

    args = parser.parse_args()
    return args

class Trainer(object):
    def __init__(self, args):
        self.args = args

        # datasets
        if args.dataset == 'sirst':
            self.train_set = SirstDataset(args, mode='train')
            self.val_set = SirstDataset(args, mode='val')
        elif args.dataset == 'IRSTD-1k':
            self.train_set = IRSTD1K_Dataset(args, mode='train')
            self.val_set = IRSTD1K_Dataset(args, mode='val')
        elif args.dataset == 'NUDT-SIRST':
            self.train_set = NUDT_Dataset(args, mode='train')
            self.val_set = NUDT_Dataset(args, mode='val')
        else:
            NameError

        self.train_data_loader = Data.DataLoader(self.train_set, batch_size=args.batch_size, shuffle=True,
                                                 num_workers=0, pin_memory=True)
        self.val_data_loader = Data.DataLoader(self.val_set, batch_size=args.batch_size, num_workers=0, pin_memory=True)

        assert args.mode in ['L', 'M', 'S']
        if args.mode == 'L':
            self.net = FCCFNet_L()
        elif args.mode == 'M':
            self.net = FCCFNet_M()
        elif args.mode == 'S':
            self.net = FCCFNet_S()
        else:
            NameError

        self.net = self.net.cuda()
        self.scaler = GradScaler()

        # self.criterion = SoftIoULoss()
        self.criterion = criterion

        # self.optimizer = torch.optim.Adagrad(self.net.parameters(), lr=args.learning_rate, weight_decay=1e-4)
        # self.optimizer = torch.optim.AdamW(self.net.parameters(), lr=args.learning_rate, weight_decay=0)
        self.optimizer = Adan(self.net.parameters(), lr=args.learning_rate, weight_decay=1e-4)

        #self.lr_scheduler = create_lr_scheduler(self.optimizer, len(self.train_data_loader), args.epochs, warmup=True,
                                                #warmup_epochs=args.warm_up_epochs)
        self.lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(self.optimizer,max_lr=args.learning_rate,total_steps=args.epochs,
                                                                pct_start=0.1,div_factor=25,final_div_factor=100)

        if args.resume:
            checkpoint = torch.load(args.resume)
            self.net.load_state_dict(checkpoint['net'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.start_epoch = checkpoint['epoch'] + 1
            if args.amp:
                self.scaler.load_state_dict(checkpoint["scaler"])

        self.iou_metric = IoUMetric()
        self.nIoU_metric = nIoUMetric(1, score_thresh=0.5)
        self.best_iou = 0
        self.best_nIoU = 0
        self.best_PD = 0
        self.best_FA = 1
        self.PD_FA = PD_FA(args.img_size)
        #加入早停计数器
        self.early_stopping_counter = 0
        self.early_stopping_patience = 100

        if args.resume:
            folder_name = os.path.abspath(
                os.path.dirname(os.path.abspath(os.path.dirname(args.resume) + os.path.sep + "."))
                + os.path.sep + ".")
        else:
            folder_name = '%s_bs%s_lr%s' % (time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time())),
                                            args.batch_size, args.learning_rate)

        folder_name = "FCCFNet_" + folder_name
        if self.train_set.__class__.__name__ == 'SirstDataset':
            self.save_folder = ops.join('results_sirst/', folder_name)  # sirst
            self.save_pth = ops.join(self.save_folder, 'checkpoint')
            if not ops.exists('results_sirst'):
                os.mkdir('results_sirst')
            if not ops.exists(self.save_folder):
                os.mkdir(self.save_folder)
            if not ops.exists(self.save_pth):
                os.mkdir(self.save_pth)
        if self.train_set.__class__.__name__ == 'IRSTD1K_Dataset':
            self.save_folder2 = ops.join('results_IRSTD-1k/', folder_name)  # IRSTD-1k
            self.save_pth2 = ops.join(self.save_folder2, 'checkpoint')
            if not ops.exists('results_IRSTD-1k'):
                os.mkdir('results_IRSTD-1k')
            if not ops.exists(self.save_folder2):
                os.mkdir(self.save_folder2)
            if not ops.exists(self.save_pth2):
                os.mkdir(self.save_pth2)
        elif args.dataset == 'NUDT-SIRST':
            self.save_folder3 = ops.join('results_NUDT-SIRST/', folder_name)  # NUDT-SIRST
            self.save_pth3 = ops.join(self.save_folder3, 'checkpoint')
            if not ops.exists('results_NUDT-SIRST'):
                os.mkdir('results_NUDT-SIRST')
            if not ops.exists(self.save_folder3):
                os.mkdir(self.save_folder3)
            if not ops.exists(self.save_pth3):
                os.mkdir(self.save_pth3)



        # Tensorboard SummaryWriter
        if self.train_set.__class__.__name__ == 'SirstDataset':
            self.writer = SummaryWriter(log_dir=self.save_folder)
            self.writer.add_text(folder_name, 'Args:%s, ' % args)
            # self.writer.add_graph(self.net, next(iter(self.train_data_loader))[0].cuda())
        if self.train_set.__class__.__name__ == 'IRSTD1K_Dataset':
            self.writer = SummaryWriter(log_dir=self.save_folder2)
            self.writer.add_text(folder_name, 'Args:%s, ' % args)
            # self.writer.add_graph(self.net, next(iter(self.train_data_loader))[0].cuda())
        elif args.dataset == 'NUDT-SIRST':
            self.writer = SummaryWriter(log_dir=self.save_folder3)
            self.writer.add_text(folder_name, 'Args:%s, ' % args)
            # self.writer.add_graph(self.net, next(iter(self.train_data_loader))[0].cuda())
        
        if self.train_set.__class__.__name__ == 'SirstDataset':
            print('folder: %s' % self.save_folder)
        if self.train_set.__class__.__name__ == 'IRSTD1K_Dataset':
            print('folder: %s' % self.save_folder2)
        elif args.dataset == 'NUDT-SIRST':
            print('folder: %s' % self.save_folder3)
        print('Args: %s' % args)

    def training(self, epoch):

        losses = []
        self.net.train()
        tbar = tqdm(self.train_data_loader)
        for i, (data, labels) in enumerate(tbar):
            data, labels = data.cuda(), labels.cuda()

            with autocast(device_type="cuda"):
                output = self.net(data)   
                loss = self.criterion(output,labels)

            self.optimizer.zero_grad()
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()

            losses.append(loss.item())
            tbar.set_description('Epoch:%3d, lr:%f, train loss:%f'
                                 % (epoch, trainer.optimizer.param_groups[0]['lr'], np.mean(losses)))
        # adjust_learning_rate(self.optimizer, epoch, args.epochs, args.learning_rate,
        #                      args.warm_up_epochs, 1e-6)

        self.lr_scheduler.step()
        self.writer.add_scalar('Losses/train loss', np.mean(losses), epoch)
        self.writer.add_scalar('Learning rate/', trainer.optimizer.param_groups[0]['lr'], epoch)

    def validation(self, epoch):

        self.iou_metric.reset()
        self.nIoU_metric.reset()
        self.PD_FA.reset()

        eval_losses = []
        self.net.eval()
        tbar = tqdm(self.val_data_loader)
        for i, (data, labels) in enumerate(tbar):
            with torch.no_grad():
                output = self.net(data.cuda())
                output = output.cpu()

            loss = self.criterion(output, labels)
            eval_losses.append(loss.item())

            self.iou_metric.update(output, labels)
            self.nIoU_metric.update(output, labels)
            self.PD_FA.update(output, labels)

            _, IoU = self.iou_metric.get()
            _, nIoU = self.nIoU_metric.get()
            Fa, Pd = self.PD_FA.get(len(self.val_set))

            tbar.set_description('  Epoch:%3d, eval loss:%f, IoU:%f, nIoU:%f, Fa:%.8f, Pd:%.5f'
                                 % (epoch, np.mean(eval_losses), IoU, nIoU, Fa, Pd))

        pkl_name = 'Epoch-%3d_IoU-%.4f_nIoU-%.4f_Fa-%.8f_Pd-%.5f.pth' % (epoch, IoU, nIoU, Fa, Pd)
        
        save_file = {"net": self.net.state_dict(),
                     "optimizer": self.optimizer.state_dict(),
                     "lr_scheduler": self.lr_scheduler.state_dict(),
                     "epoch": epoch,
                     "args": args}

        if args.amp:
            save_file["scaler"] = self.scaler.state_dict()

        if self.train_set.__class__.__name__ == 'SirstDataset':
            save_pth = self.save_pth
        if self.train_set.__class__.__name__ == 'IRSTD1K_Dataset':
            save_pth = self.save_pth2
        if IoU > self.best_iou:
            torch.save(save_file, ops.join(save_pth,"best_iou.pth"))
            self.best_iou = IoU
            self.early_stopping_counter = 0
        if nIoU > self.best_nIoU:
            torch.save(save_file, ops.join(save_pth, "best_niou.pth"))
            self.best_nIoU = nIoU
            self.early_stopping_counter = 0
        if Pd > self.best_PD:
            torch.save(save_file, ops.join(save_pth, "best_pd.pth"))
            self.best_PD = Pd
            self.early_stopping_counter = 0
        if Fa < self.best_FA:
            torch.save(save_file, ops.join(save_pth, "best_fa.pth"))
            self.best_FA = Fa
            self.early_stopping_counter = 0
        if self.early_stopping_counter >= self.early_stopping_patience:
            torch.save(save_file, ops.join(save_pth, "early_stop.pth"))
            print(f"early_stop:{self.early_stopping_patience}epoch")
            return False
        self.early_stopping_counter = self.early_stopping_counter + 1
        print("early_stopping_counter:",self.early_stopping_counter,'/',self.early_stopping_patience)

        self.writer.add_scalar('Losses/eval_loss', np.mean(eval_losses), epoch)
        self.writer.add_scalar('Eval/IoU', IoU, epoch)
        self.writer.add_scalar('Eval/nIoU', nIoU, epoch)
        self.writer.add_scalar('Best/IoU', self.best_iou, epoch)
        self.writer.add_scalar('Best/nIoU', self.best_nIoU, epoch)
        self.writer.add_scalar('Eval/Pd', Pd, epoch)
        self.writer.add_scalar('Eval/Fa', Fa, epoch)
        self.writer.add_scalar('Best/Pd', self.best_PD, epoch)
        self.writer.add_scalar('Best/Fa', self.best_FA, epoch)

        return True

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

if __name__ == '__main__':
    set_seed(666)
    args = parse_args()

    trainer = Trainer(args)
    for epoch in range(args.start_epoch, args.epochs + 1):
        trainer.training(epoch)
            # if epoch > 5:
            #     trainer.validation(epoch)
        flag = trainer.validation(epoch)
        if flag == False:
            break
    print('Best IoU: %.5f, best nIoU: %.5f, Best Pd: %.5f, best Fa: %.5f' %
            (trainer.best_iou, trainer.best_nIoU, trainer.best_PD, trainer.best_FA))