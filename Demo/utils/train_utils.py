from utils.mysummary import summary
from Demo.Datasets.datasets_utils.generalfunciton import indexed_dataset
from Demo.utils.general_func import find_all_index_N
import Demo.Models as models
import Demo.Datasets as datasets
from torch import nn, optim
import torch
import seaborn as sns
import scipy.io as io
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mplb
import warnings
import time
import pickle
import logging
import copy
import sys
import os


def root_path_k(x, k): return os.path.abspath(
    os.path.join(x, *([os.pardir] * (k + 1))))


projecht_dir = root_path_k(__file__, 2)
# add the project directory to the system path
if projecht_dir not in sys.path:
    sys.path.insert(0, projecht_dir)


class train_utils(object):
    def __init__(self, args, save_dir: str):
        self.args = args
        self.save_dir = save_dir

    def setup(self):
        args = self.args

        # Consider the gpu or cpu condition
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.device_count = 1
            self.device_count = torch.cuda.device_count()
            logging.info('using {} gpus'.format(self.device_count))
            assert args.batch_size % self.device_count == 0, "batch size should be divided by device count"
        else:
            warnings.warn("gpu is not available")
            self.device = torch.device("cpu")
            self.device_count = 1
            logging.info('using {} cpu'.format(self.device_count))

        # Load the datasets
        Dataset = getattr(datasets, args.data_name)
        self.Dataset_num_classes, self.Dataset_Fs = Dataset.num_classes, Dataset.Fs
        self.datasets = {}
        subargs = {k: getattr(args, k) for k in [
            'data_dir', 'data_type', 'normlizetype', 'test_size']}

        (self.datasets['train'], self.datasets['val']), self.label_name = Dataset(subargs).data_preprare(
            signal_size=args.data_signalsize, SNR=args.SNR, try_preload=True)

        # load the datasets
        self.dataloaders = {x: torch.utils.data.DataLoader(self.datasets[x], batch_size=args.batch_size,
                                                           shuffle=(
                                                               True if x == 'train' else False),
                                                           num_workers=args.num_workers,
                                                           pin_memory=(True if self.device == 'cuda' else False))
                            for x in ['train', 'val']}
        logging.info('train_datasize: {} | val_datasize: {}'.format(len(self.datasets['train']),
                                                                    len(self.datasets['val'])))

        # Define the models
        self.model = getattr(models, args.model_name)(
            in_channel=Dataset.inputchannel, out_channel=Dataset.num_classes)

        # Invert the models and define the loss
        self.model.to(self.device)
        try:
            info = summary(
                self.model, self.datasets['train'][0][0].shape, batch_size=-1, device="cuda")
            for item in info.split('\n'):
                logging.info(item)
        except:
            print('summary does not work!')

        self.criterion = nn.CrossEntropyLoss(reduction='mean')

        # Define the optimizer
        if args.opt == 'sgd':
            self.optimizer = optim.SGD(self.model.parameters(), lr=args.lr,
                                       momentum=args.momentum, weight_decay=args.weight_decay)
        elif args.opt == 'adam':
            self.optimizer = optim.Adam(
                self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        else:
            raise Exception("optimizer not implement")

        # Define the learning rate decay
        if args.lr_scheduler == 'step':
            steps = [int(step) for step in args.steps.split(',')]
            self.lr_scheduler = optim.lr_scheduler.MultiStepLR(
                self.optimizer, steps, gamma=args.gamma)
        elif args.lr_scheduler == 'exp':
            self.lr_scheduler = optim.lr_scheduler.ExponentialLR(
                self.optimizer, args.gamma)
        elif args.lr_scheduler == 'stepLR':
            steps = int(args.steps)
            self.lr_scheduler = optim.lr_scheduler.StepLR(
                self.optimizer, steps, args.gamma)
        elif args.lr_scheduler == 'fix':
            self.lr_scheduler = None
        else:
            raise Exception("lr schedule not implement")

        # Load the checkpoint
        self.start_epoch = 0

        # confusion matrix
        self.c_matrix = {phase: np.zeros(
            [Dataset.num_classes, Dataset.num_classes]) for phase in ['train', 'val']}
        self.c_matrix_best = {}

    def train(self):
        time_start_train = time.time()
        args = self.args

        best_acc = 0.0
        step_start = time.time()

        self.Records = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], 'best_epoch': 0,
                        "train_confusion_matrix": [], "val_confusion_matrix": []}
        self.MinorRecords = {"train_loss": [], "train_acc": []}

        for epoch in range(self.start_epoch, args.max_epoch):
            logging.info('-' * 5 + 'Epoch {}/{}'.format(epoch,
                         args.max_epoch - 1) + '-' * 5)
            # Update the learning rate
            if self.lr_scheduler is not None:
                # self.lr_scheduler.step(epoch)
                logging.info('current lr: {}'.format(
                    [float('%.8f' % item) for item in self.lr_scheduler.get_last_lr()]))
            else:
                logging.info('current lr: {}'.format([args.lr,]))

            # Each epoch has a training and val phase
            for phase in ['train', 'val']:
                # Define the temp variable
                epoch_start = time.time()
                epoch_acc, epoch_loss = 0, 0
                batch_acc, batch_loss = 0, 0
                batch_count = 0
                self.c_matrix[phase] = np.zeros_like(
                    self.c_matrix[phase])  # confusion matrix
                # Set models to train mode or test mode
                if phase == 'train':
                    self.model.train()
                else:
                    self.model.eval()

                for batch_idx, (inputs, labels_, info) in enumerate(self.dataloaders[phase]):
                    inputs = inputs.to(self.device)
                    labels = labels_.to(self.device)
                    info = info.to(self.device)
                    # Do the learning process, in val, we do not care about the gradient for relaxing
                    with torch.set_grad_enabled(phase == 'train'):
                        logits = self.model(inputs)
                        loss = self.criterion(logits, labels)
                    pred = logits.argmax(dim=1)
                    correct = torch.eq(pred, labels).sum().item()
                    epoch_loss += loss.item() * inputs.size(0)
                    epoch_acc += correct

                    # confusion matrix
                    for i, j in zip(labels.detach().cpu().numpy(), pred.detach().cpu().numpy()):
                        self.c_matrix[phase][i][j] += 1

                    # Calculate the training information
                    if phase == 'train':
                        # backward
                        self.optimizer.zero_grad()
                        loss.backward()
                        self.optimizer.step()

                        batch_loss += loss.item() * inputs.size(0)
                        batch_acc += correct
                        batch_count += inputs.size(0)
                        self.MinorRecords['train_loss'].append(loss.item())
                        self.MinorRecords['train_acc'].append(
                            correct/inputs.size(0))

                        # Print the training information
                        if (batch_idx + 1) % args.print_step == 0:
                            batch_loss = batch_loss / batch_count
                            batch_acc = batch_acc / batch_count
                            temp_time = time.time()
                            train_time = temp_time - step_start
                            step_start = temp_time
                            batch_time = train_time / args.print_step
                            sample_per_sec = 1.0 * batch_count / train_time
                            temp_info = f'<tempinfo> Epoch: {epoch:3d} [{(batch_idx + 1) * len(inputs):4d}/{len(self.dataloaders[phase].dataset):4d}],  step: {batch_idx + 1:3d}' + \
                                f'  {sample_per_sec:.1f} samples/sec  {batch_time:.4f} sec/batch  Train Acc: {batch_acc * 100:.2f},  Train Loss: {batch_loss:.4f}'
                            logging.info(temp_info)

                            batch_acc, batch_loss = 0, 0
                            batch_count = 0

                # Print the train and val information via each epoch
                epoch_loss /= len(self.dataloaders[phase].dataset)
                epoch_acc /= len(self.dataloaders[phase].dataset)
                temp_info = f'<info> Epoch: {epoch:d},  Cost {time.time() - epoch_start:.4f} sec,' + \
                    f'  {phase:s}-Acc: {epoch_acc * 100:.4f},  {phase}-Loss: {epoch_loss:.4f}'
                logging.info(temp_info)

                self.Records["%s_loss" % phase].append(epoch_loss)
                self.Records["%s_acc" % phase].append(epoch_acc)
                self.Records["%s_confusion_matrix" %
                             phase].append(self.c_matrix[phase])

                # save the models
                if phase == 'val':

                    if epoch_acc >= best_acc:
                        best_acc = epoch_acc
                        self.Records['best_epoch'] = epoch
                        logging.info(
                            "save best epoch {}, best acc {:.4f}".format(epoch, epoch_acc))
                        save_best_data_dir = os.path.join(self.save_dir,
                                                          'epoch{}-acc{:.4f}-best_model.pth'.format(epoch,
                                                                                                    epoch_acc * 100))
                        save_best_data = {'epoch': epoch, 'state_dict': self.model.state_dict(),
                                          'optimizer': self.optimizer.state_dict(), 'opt': self.args.opt}
                        self.c_matrix_best = copy.deepcopy(self.c_matrix)

                    if epoch == args.max_epoch - 1:
                        if args.save_model:
                            # save the best models according to the val accuracy
                            torch.save(save_best_data, save_best_data_dir)
                            # save the final models
                            logging.info(
                                "save final epoch {}, final acc {:.4f}".format(epoch, epoch_acc))
                            save_data = {'epoch': epoch, 'state_dict': self.model.state_dict(),
                                         'optimizer': self.optimizer.state_dict(), 'opt': self.args.opt}
                            torch.save(save_data,
                                       os.path.join(self.save_dir,
                                                    'epoch{}-acc{:.4f}-final_model.pth'.format(epoch, epoch_acc * 100)))

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

        # After train
        self.Records.update({'Dataset_num_classes': self.Dataset_num_classes,
                            'Dataset_Fs': self.Dataset_Fs, 'label_name': self.label_name})
        # stacks Record list to numpy when finished train process
        for k in self.Records.keys():
            self.Records[k] = np.array(self.Records[k])
        for k in self.MinorRecords.keys():
            self.MinorRecords[k] = np.array(self.MinorRecords[k])
        io.savemat(os.path.join(self.save_dir, "Records.mat"), self.Records)
        io.savemat(os.path.join(self.save_dir, "MinorRecords.mat"),
                   self.MinorRecords)

        # log the best and final acc and loss
        info = "max train acc in epoch {:2d}: {:10.6f}\n".format(self.Records['train_acc'].argmax(),
                                                                 self.Records['train_acc'].max()) \
               + "max val acc in epoch {:2d}: {:10.6f}\n".format(self.Records['val_acc'].argmax(),
                                                                 self.Records['val_acc'].max()) \
               + "final train acc: %.6f\n final val acc: %.6f\n" \
               % (self.Records['train_acc'][-1], self.Records['val_acc'][-1])
        for item in info.split('\n'):
            logging.info(item)
        with open(os.path.join(self.save_dir, 'acc output.txt'), 'w') as f:
            f.write(info)

        # training time
        logging.info("<training time>: {:.3f}".format(
            time.time() - time_start_train))

        # check data save
        if self.args.save_N_data > 0:
            self._collect_data(self.args.save_N_data)
            
    def _collect_data(self, Target_number=100):

        subargs = {k: getattr(self.args, k) for k in ['data_dir', 'data_type', 'normlizetype', 'test_size'] +
                   ['data_name', 'data_signalsize', 'SNR']}
        for temp_phase in ['train', 'val']:  # do both
            ToBeCollect = self.datasets[temp_phase]
            temp_data_path = os.path.join(
                self.save_dir, f'all_dataset_{temp_phase:s}_SavedInTraining.pkl')
            data, label, infos = [], [], []
            for i in range(len(ToBeCollect)):
                item, label_, *info = ToBeCollect[i]
                data.append(item)
                label.append(label_)
                infos.append(info)
            data, label, infos = np.array(data).squeeze(), np.array(
                label).squeeze(), np.array(infos)
            Index = find_all_index_N(label, Target_number)
            temp = {'data': data[Index, ...], 'label': label[Index, ...], 'infos': infos[Index, ...],
                    'label_name': [item.strip().capitalize() for item in self.label_name],
                    'Fs': self.Dataset_Fs, 'subargs': subargs}
            with open(temp_data_path, 'wb') as f:
                pickle.dump(temp, f)
        print('data load done! (by dataset)')

    def plot_save(self):
        # set color
        current_cmap = sns.color_palette("husl", 10)
        sns.set(style="white")
        sns.set(style="ticks", context="notebook",
                font='Times New Roman', palette=current_cmap, font_scale=1)
        mplb.rcParams['font.size'] = 8
        mplb.rcParams['font.family'] = 'Times New Roman'
        mplb.rcParams['figure.dpi'] = 300
        # make dir
        self.save_dir_sub = os.path.join(self.save_dir, "postprosess")
        if not os.path.exists(self.save_dir_sub):
            os.makedirs(self.save_dir_sub)

        # plot confusion matrix
        for phase in ['train', 'val']:
            f, ax = plt.subplots(figsize=(5, 4))
            sns.heatmap(self.c_matrix_best[phase].astype(int), annot=True, ax=ax, fmt="d",
                        cmap=sns.color_palette("light:b", as_cmap=True),
                        # Blues | RdBu_r | light:b
                        linewidths=0.3, annot_kws={'size': 12})
            ax.invert_yaxis()
            ax.set_xticklabels(self.label_name)
            ax.set_yticklabels(self.label_name)
            ax.set_xlabel('Prediction')
            ax.set_ylabel('Truth')
            # ax.set_title('Confusion matrix: %s'%phase)
            f.tight_layout()
            f.savefig(os.path.join(self.save_dir_sub,
                      "Confusion_matrix_%s.jpg" % phase))
        plt.close('all')

        # plot Records
        Ls = ['-', '--', '-.', ':']
        Markers = ['.', ',', 'o', 'v', '^', '<',
                   '>', '1', '2', '3', '4', '8', 's', ]
        epochs = np.arange(self.Records['train_loss'].shape[0]) + 1
        fig = plt.figure(figsize=[5, 4])
        ax1 = fig.add_subplot(2, 1, 1)
        ax1.plot(epochs, self.Records['train_loss'],
                 "b", markersize=2, linewidth=2)
        ax1.plot(epochs, self.Records['val_loss'],
                 "r", markersize=2, linewidth=2)
        ax1.legend(['train loss', 'val loss'], loc=5)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        # ax1.grid()
        ax2 = fig.add_subplot(2, 1, 2)
        ax2.plot(epochs, self.Records['train_acc'] * 100, "b-.d",
                 epochs, self.Records['val_acc'] * 100, "r-.d", markersize=2, linewidth=2)
        ax2.legend(['train acc', 'val acc'], loc=5)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Acc.')
        # ax2.set_ylim([80,100])
        fig.tight_layout()
        # ax2.grid()
        fig.savefig(os.path.join(self.save_dir_sub, "loss_acc.jpg"))

        # plot MinorRecords
        steps = np.arange(self.MinorRecords['train_loss'].shape[0]) + 1
        steps = steps / len(steps) * self.args.max_epoch
        fig = plt.figure(figsize=[5, 4])
        ax1 = fig.add_subplot(2, 1, 1)
        ax1.plot(
            steps, self.MinorRecords['train_loss'], "b-", markersize=3, linewidth=1)
        # ax1.grid()
        # ax1.legend('train loss', loc=5)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax2 = fig.add_subplot(2, 1, 2)
        ax2.plot(steps, self.MinorRecords['train_acc']
                 * 100, "b-", markersize=3, linewidth=1)
        # ax2.legend('train acc')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('acc.')
        # ax2.set_ylim([80,100])
        fig.tight_layout()
        # ax2.grid()
        fig.savefig(os.path.join(self.save_dir_sub, "Minor_loss_acc.jpg"))

        # plt data spectrum
        collected_data_for_spectrum = {}
        for i in range(len(self.datasets['train'])):
            item, label, *info = self.datasets['train'][i]
            if type(label) == list:
                label = '_'.join([str(item) for item in label])
            if str(label) not in collected_data_for_spectrum.keys():
                collected_data_for_spectrum[label] = item

        temp_label = sorted(collected_data_for_spectrum.keys())
        A = np.array([collected_data_for_spectrum[k].squeeze()
                     for k in temp_label])

        if self.args.data_type == 'time':
            Z = np.fft.fft(A.squeeze(), axis=-1) / A.shape[-1]
            f = np.arange(Z.shape[1])/Z.shape[1] * 2
            fig = plt.figure(figsize=[5, 3])
            ax = fig.add_subplot()
            ax.plot(f, np.abs(Z.T))
            ax.set_xlim(0, 1)
            ax.legend([self.label_name[int(l)] for l in temp_label])
            fig.tight_layout()
            fig.savefig(os.path.join(self.save_dir_sub, "data_spectrum.jpg"))
            fig = plt.figure(figsize=[5, 3])
            ax = fig.add_subplot()
            ax.plot(range(A.squeeze().shape[1]), A.squeeze().T)
            fig.tight_layout()
            fig.savefig(os.path.join(self.save_dir_sub, "time.jpg"))
            # water fall (time)
            fig, (ax1) = plt.subplots(figsize=(5, 4),
                                      subplot_kw={'projection': '3d'})
            fig.tight_layout(pad=0)
            X, Y = np.meshgrid(range(A.shape[1]), range(A.shape[0]))
            # X, Y, Z = axes3d.get_test_data(0.05)
            a = ax1.plot_wireframe(X, Y, A, rstride=1, cstride=0)
            a.set_linewidth(0.5)
            ax1.set_xlabel('Node')
            ax1.set_yticks(range(A.shape[0]),
                           [self.label_name[int(item)] for item in temp_label])
            ax1.tick_params(pad=0)
            ax1.set_xlim([0, A.shape[1]-1])
            ax1.set_ylim([0, A.shape[0]-1])
            fig.savefig(os.path.join(self.save_dir_sub, "time-waterfall.jpg"))

            # water fall (FFT)
            fig, (ax1) = plt.subplots(figsize=(5, 4),
                                      subplot_kw={'projection': '3d'})
            fig.tight_layout(pad=0)
            X, Y = np.meshgrid(f, range(Z.shape[0]))
            # X, Y, Z = axes3d.get_test_data(0.05)
            N = Z.shape[1]//2
            a = ax1.plot_wireframe(X[:, :N], Y[:, :N], np.abs(
                Z[:, :N]), rstride=1, cstride=0)
            a.set_linewidth(0.5)
            ax1.set_xticks([0, 0.5, 1], ['0', '0.5', '1'])
            ax1.set_xlabel('f')
            ax1.set_yticks(range(Z.shape[0]),
                           [self.label_name[int(item)] for item in temp_label])
            ax1.tick_params(pad=0)
            ax1.set_xlim([0, 1.05])
            ax1.set_ylim([0, Z.shape[0]-1])
            fig.savefig(os.path.join(self.save_dir_sub,
                        "data_spectrum-waterfall.jpg"))

        elif self.args.data_type == 'fft':
            fig = plt.figure(figsize=[8, 5])
            ax = fig.add_subplot()
            ax.plot(range(A.squeeze().shape[1]), A.squeeze().T)
            try:
                ax.legend([self.label_name[int(item)] for item in list(
                    self.collected_data_for_spectrum.keys())])
            except:
                ax.legend(list(self.collected_data_for_spectrum.keys()))
            fig.savefig(os.path.join(self.save_dir_sub, "data_spectrum.jpg"))
