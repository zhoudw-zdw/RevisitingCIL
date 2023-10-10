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
from convs.vpt import build_promptmodel, VPT_ViT

num_workers = 8
# batch_size=128

class Learner(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        self._network = SimpleVitNet(args, True)
        
        # ! VPT
        model = build_promptmodel(
            modelname='vit_base_patch16_224_in21k',  
            Prompt_Token_num=args["prompt_token_num"], 
            VPT_type=args["vpt_type"])
        prompt_state_dict = model.obtain_prompt()
        model.load_prompt(prompt_state_dict)
        model.eval()
        model.out_dim=768
        self._network.convnet = model
        self.proto_list = []
        self.class_list = []
        self.radius = 0
        
        # configs
        self.batch_size= args["batch_size"]
        self.init_lr=args["init_lr"]
        self.weight_decay=args["weight_decay"] if args["weight_decay"] is not None else 0.0005
        self.min_lr=args['min_lr'] if args['min_lr'] is not None else 1e-8
        self.loss_fn=args['loss_fn'] if args['loss_fn'] is not None else 'cross_entropy'
        self.args=args
        self.data_augmentation=args['data_augmentation'] if args['data_augmentation'] is not None else 'train'
        self.alpha = args['alpha'] if args['alpha'] is not None else 0.5
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

        # ! Using PASS
        class_list=np.unique(self.train_dataset.labels)
        proto_list = []
        radius = []
        for class_index in class_list:
            # print('Replacing...',class_index)
            data_index=(label_list==class_index).nonzero().squeeze(-1)
            embedding=embedding_list[data_index]
            proto=embedding.mean(0)
            proto_list.append(proto)
            # self._network.fc.weight.data[class_index]=proto
            # ! Maintain partial fc weight
            self._network.fc.weight.data[class_index] = (1 - self.alpha) * self._network.fc.weight.data[class_index] + self.alpha * proto
            if self._cur_task == 0:
                cov = np.cov(embedding.T)
                radius.append(np.trace(cov) / embedding_list.shape[1])
                
        if self._cur_task == 0:
            self.proto_list = proto_list
            self.class_list = class_list
            self.radius = np.sqrt(np.mean(radius))
            print('Radius:', self.radius) 
        else:
            self.proto_list = np.concatenate((self.proto_list, proto_list), axis=0)
            self.class_list = np.concatenate((self.class_list, class_list), axis=0)
        return model
   
    def _get_proto_list(self, trainloader, model, args):
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
        proto_list = []
        class_list=np.unique(self.train_dataset.labels)
        for class_index in class_list:
            data_index=(label_list==class_index).nonzero().squeeze(-1)
            embedding=embedding_list[data_index]
            proto=embedding.mean(0)
            proto_list.append(proto)
        proto_tensor = torch.stack(proto_list, dim=0).cuda()
        return proto_tensor
        
        
    def incremental_train(self, data_manager, mode="train"):
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(self._cur_task)
        self._network.update_fc(self._total_classes)
        logging.info("Learning on {}-{}".format(self._known_classes, self._total_classes))

        # ! Data augmentation, using random flip
        train_dataset = data_manager.get_dataset(np.arange(self._known_classes, self._total_classes),source="train", mode=self.data_augmentation, flip_prob=0.3)
        self.train_dataset=train_dataset
        self.data_manager=data_manager
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=num_workers)
        
        test_dataset = data_manager.get_dataset(np.arange(0, self._total_classes), source="test", mode="test_norm" )
        self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=num_workers)
        
        # ! Data augmentation, using random flip
        train_dataset_for_protonet=data_manager.get_dataset(np.arange(self._known_classes, self._total_classes),source="train", mode='test_norm')
        self.train_loader_for_protonet = DataLoader(train_dataset_for_protonet, batch_size=self.batch_size, shuffle=True, num_workers=num_workers)
        
        # ! Before DataParallel
        # if len(self._multiple_gpus) > 1:
        #     print('Multiple GPUs')
        #     self._network = nn.DataParallel(self._network, self._multiple_gpus)
        self._train(
            self.train_loader,
            self.test_loader, 
            self.train_loader_for_protonet,
            mode=mode,)
            # if len(self._multiple_gpus) > 1:
            #     self._network = self._network.module

    def _train(self, train_loader, test_loader, train_loader_for_protonet, mode="train"):
        self._network.cuda()
        
        # mihghao's solution
        if self._cur_task == 0:
            self.construct_dual_branch_network()
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
                else:
                    raise NotImplementedError
                scheduler=optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.args['tuned_epoch'], eta_min=self.min_lr)
                # scheduler=optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
                self._init_train(train_loader, test_loader, optimizer, scheduler)  
        else:
            if len(self._multiple_gpus) > 1:
                self._network = self._network.module
        self.replace_fc(train_loader_for_protonet, self._network, None)

    def construct_dual_branch_network(self):
        network = MultiBranchCosineIncrementalNet(self.args, True)
        # ! After
        # if len(self._multiple_gpus) > 1:
        #     network.construct_dual_branch_network(self._network.module)
        # else:
        #     network.construct_dual_branch_network(self._network)
        # self._network=network.cuda()
        
        # ! Before
        network.construct_dual_branch_network(self._network)
        if len(self._multiple_gpus) > 1:
            print('Multiple GPUs')
            self._network = nn.DataParallel(network, self._multiple_gpus).cuda()

    def _compute_prototype_loss(self, fc_weights, prototypes):
        # import pdb; pdb.set_trace()
        loss = ((fc_weights - prototypes) ** 2).sum(1).mean()
        return loss
    
    def _compute_loss(self, inputs, targets, proto_tensor=None, ifrot=False) -> torch.Tensor:
        # import pdb; pdb.set_trace()
        logits = self._network(inputs)["logits"]
        # ! Loss function, using focal loss
        if self.loss_fn == "cross_entropy":
            loss_cls = F.cross_entropy(logits, targets)
        elif self.loss_fn == "focal_loss":
            targets_onehot = torch.zeros(targets.size(0), 30).to(targets.device).scatter_(1, targets.unsqueeze(1), 1)
            loss_cls = sigmoid_focal_loss(logits, targets_onehot, alpha=0.25, gamma=2.0, reduction="mean")
        else:
            raise NotImplementedError
        
        if proto_tensor is not None:
            # Compute prototype loss
            proto_loss = self._compute_prototype_loss(self._network.module.fc.weight.data, proto_tensor)                
            
            # Combine main loss and prototype loss
            beta = 0.1  # This is a hyperparameter you can tune
            loss = loss_cls + beta * proto_loss
        
        return loss, logits
        
    def _init_train(self, train_loader, test_loader, optimizer, scheduler):
        prog_bar = tqdm(range(self.args['tuned_epoch']))
        for _, epoch in enumerate(prog_bar):
            self._network.train()
            losses = 0.0
            correct, total = 0, 0
            
            proto_tensor = self._get_proto_list(train_loader, self._network, None)
            for i, (_, inputs, targets) in enumerate(train_loader):
                # inputs, targets = inputs.to(self._device), targets.to(self._device)
                inputs, targets = inputs.cuda(), targets.cuda()
                # logits = self._network(inputs)["logits"]
                
                # ! PASS Loss
                loss, logits = self._compute_loss(inputs, targets, proto_tensor)
                
                # ! Regularization
                l2_regularization = torch.tensor(0.).cuda()
                for param in self._network.parameters():
                    l2_regularization += torch.norm(param, 2).cuda()
                loss += self.weight_decay * l2_regularization
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()

                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)

            scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)

            if epoch % 5 == 0:
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    self.args['tuned_epoch'],
                    losses / len(train_loader),
                    train_acc,
                )
            else:
                test_acc = self._compute_accuracy(self._network, test_loader)
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    self.args['tuned_epoch'],
                    losses / len(train_loader),
                    train_acc,
                    test_acc,
                )
            logging.info(info)
            prog_bar.set_description(info)
        # ! save ckpt
        self.save_checkpoint(f'checkpoints/minghao_lr({self.init_lr})_wd({self.weight_decay})_opt({self.args["optimizer"]})_vt({self.args["vpt_type"]})_loss({self.loss_fn})_epoch({self.args["tuned_epoch"]})')
        logging.info(info)
