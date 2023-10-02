import logging
import numpy as np
import torch
from torch import nn
from torch.serialization import load
from tqdm import tqdm
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from utils.inc_net import IncrementalNet,SimpleCosineIncrementalNet,MultiBranchCosineIncrementalNet,SimpleVitNet
from models.base import BaseLearner
from utils.toolkit import target2onehot, tensor2numpy

from torchvision.ops.focal_loss import sigmoid_focal_loss

num_workers = 8
# batch_size=128

class Learner(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        self._network = SimpleVitNet(args, True)
        self.batch_size= args["batch_size"]
        self.init_lr=args["init_lr"]
        self.weight_decay=args["weight_decay"] if args["weight_decay"] is not None else 0.0005
        self.min_lr=args['min_lr'] if args['min_lr'] is not None else 1e-8
        self.args=args
        self.state_dict = None

    def load_checkpoint(self, state_dict):
        self.state_dict = state_dict
        # if len(self._multiple_gpus) > 1:
        #     # replace "module." in state_dict
        #     for k in list(state_dict["model_state_dict"].keys()):
        #         if k.startswith("module."):
        #             state_dict["model_state_dict"][k[7:]] = state_dict["model_state_dict"].pop(k)
        self._network.load_state_dict(state_dict["model_state_dict"], strict=True)
        
    def after_task(self):
        self._known_classes = self._total_classes
    
    def replace_fc(self, trainloader, model, args):
        model = model.eval()
        embedding_list = []
        label_list = []
        with torch.no_grad():
            for i, batch in enumerate(trainloader):
                (_,data,label)=batch
                data=data.cuda()
                label=label.cuda()
                # import pdb; pdb.set_trace()
                embedding = model(data)['features']
                # embedding=model.convnet(data)
                embedding_list.append(embedding.cpu())
                label_list.append(label.cpu())
        embedding_list = torch.cat(embedding_list, dim=0)
        label_list = torch.cat(label_list, dim=0)

        class_list=np.unique(self.train_dataset.labels)
        proto_list = []
        for class_index in class_list:
            # print('Replacing...',class_index)
            data_index=(label_list==class_index).nonzero().squeeze(-1)
            embedding=embedding_list[data_index]
            proto=embedding.mean(0)
            self._network.fc.weight.data[class_index]=proto
        return model
   
    def incremental_train(self, data_manager, mode="train"):
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(self._cur_task)
        self._network.update_fc(self._total_classes)
        logging.info("Learning on {}-{}".format(self._known_classes, self._total_classes))

        # ! Data augmentation, using random flip
        train_dataset = data_manager.get_dataset(np.arange(self._known_classes, self._total_classes),source="train", mode="flip")
        self.train_dataset=train_dataset
        self.data_manager=data_manager
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=num_workers)
        
        test_dataset = data_manager.get_dataset(np.arange(0, self._total_classes), source="test", mode="test" )
        self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=num_workers)
        
        # ! Data augmentation, using random flip
        train_dataset_for_protonet=data_manager.get_dataset(np.arange(self._known_classes, self._total_classes),source="train", mode="flip")
        self.train_loader_for_protonet = DataLoader(train_dataset_for_protonet, batch_size=self.batch_size, shuffle=True, num_workers=num_workers)
        if len(self._multiple_gpus) > 1:
            print('Multiple GPUs')
            self._network = nn.DataParallel(self._network, self._multiple_gpus)
        self._train(
            self.train_loader,
            self.test_loader, 
            self.train_loader_for_protonet,
            mode=mode,)
            # if len(self._multiple_gpus) > 1:
            #     self._network = self._network.module

    def _train(self, train_loader, test_loader, train_loader_for_protonet, mode="train"):
        # self._network.to(self._device)
        
        # mihghao's solution
        if self._cur_task == 0:
            if mode=="eval":
                logging.info("Load model from state dict")
                self.load_checkpoint(self.state_dict)
            elif mode=="train":
                logging.info("Init train")
                # show total parameters and trainable parameters
                total_params = sum(p.numel() for p in self._network.parameters())
                print(f'{total_params:,} total parameters.')
                total_trainable_params = sum(
                    p.numel() for p in self._network.parameters() if p.requires_grad)
                print(f'{total_trainable_params:,} training parameters.')
                if total_params != total_trainable_params:
                    for name, param in self._network.named_parameters():
                        if param.requires_grad:
                            print(name, param.numel())
                
                # ! Optimizer
                if self.args['optimizer']=='sgd':
                    optimizer = optim.SGD(
                        self._network.parameters(), 
                        momentum=0.9, 
                        lr=self.init_lr,
                        weight_decay=self.weight_decay)
                elif self.args['optimizer']=='adam':
                    optimizer=optim.AdamW(
                        self._network.parameters(), 
                        lr=self.init_lr, 
                        weight_decay=self.weight_decay)
                scheduler=optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.args['tuned_epoch'], eta_min=self.min_lr)
                self._init_train(train_loader, test_loader, optimizer, scheduler)
            self.construct_dual_branch_network()
        else:
            self._network = self._network.module
        self.replace_fc(train_loader_for_protonet, self._network, None)

    def construct_dual_branch_network(self):
        network = MultiBranchCosineIncrementalNet(self.args, True)
        network.construct_dual_branch_network(self._network.module)
        # self._network=network.to(self._device)
        self._network=network.cuda()

    def _init_train(self, train_loader, test_loader, optimizer, scheduler):
        prog_bar = tqdm(range(self.args['tuned_epoch']))
        for _, epoch in enumerate(prog_bar):
            self._network.train()
            losses = 0.0
            correct, total = 0, 0
            for i, (_, inputs, targets) in enumerate(train_loader):
                # inputs, targets = inputs.to(self._device), targets.to(self._device)
                inputs, targets = inputs.cuda(), targets.cuda()
                logits = self._network(inputs)["logits"]

                # ! Loss function, using focal loss
                loss = F.cross_entropy(logits, targets)
                # loss = sigmoid_focal_loss(logits, targets, alpha=0.25, gamma=2.0, reduction="mean")
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()

                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)

            scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)

            test_acc = self._compute_accuracy(self._network, test_loader)
            info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}".format(
                self._cur_task,
                epoch + 1,
                self.args['tuned_epoch'],
                losses / len(train_loader),
                train_acc,
                test_acc,
            )
            prog_bar.set_description(info)
        # ! save ckpt
        self.save_checkpoint(f'checkpoints/minghao_lr({self.init_lr})_wd({self.weight_decay})_opt({self.args["optimizer"]})')
        logging.info(info)
