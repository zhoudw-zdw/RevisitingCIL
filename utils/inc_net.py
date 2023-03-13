import copy
import logging
import torch
from torch import nn
from convs.linears import SimpleLinear, SplitCosineLinear, CosineLinear
import timm


def get_convnet(args, pretrained=False):

    name = args["convnet_type"].lower()
    #Resnet
    if name=="pretrained_resnet18":
        from convs.resnet import resnet18, resnet34, resnet50, resnet101, resnet152
        model=resnet18(pretrained=False,args=args)
        model.load_state_dict(torch.load("./pretrained_models/resnet18-f37072fd.pth"),strict=False)
        return model.eval()
    elif name=="pretrained_resnet50":
        from convs.resnet import resnet18, resnet34, resnet50, resnet101, resnet152
        model=resnet50(pretrained=False,args=args)
        model.load_state_dict(torch.load("./pretrained_models/resnet50-11ad3fa6.pth"),strict=False)
        return model.eval()
    elif name=="pretrained_resnet101":
        from convs.resnet import resnet18, resnet34, resnet50, resnet101, resnet152
        model=resnet101(pretrained=False,args=args)
        model.load_state_dict(torch.load("./pretrained_models/resnet101-cd907fc2.pth"),strict=False)
        return model.eval()
    elif name=="pretrained_resnet152":
        from convs.resnet import resnet18, resnet34, resnet50, resnet101, resnet152
        model=resnet152(pretrained=False,args=args)
        model.load_state_dict(torch.load("./pretrained_models/resnet152-f82ba261.pth"),strict=False)
        return model.eval()
    
    #SimpleCIL or SimpleCIL w/ Finetune
    elif name=="pretrained_vit_b16_224" or name=="vit_base_patch16_224":
        model=timm.create_model("vit_base_patch16_224",pretrained=True, num_classes=0)
        model.out_dim=768
        return model.eval()
    elif name=="pretrained_vit_b16_224_in21k" or name=="vit_base_patch16_224_in21k":
        model=timm.create_model("vit_base_patch16_224_in21k",pretrained=True, num_classes=0)
        model.out_dim=768
        return model.eval()
    
    # SSF 
    elif '_ssf' in name:
        if args["model_name"]=="adam_ssf":
            from convs import vision_transformer_ssf
            if name=="pretrained_vit_b16_224_ssf":
                model = timm.create_model("vit_base_patch16_224_ssf", pretrained=True, num_classes=0)
                model.out_dim=768
            elif name=="pretrained_vit_b16_224_in21k_ssf":
                model=timm.create_model("vit_base_patch16_224_in21k_ssf",pretrained=True, num_classes=0)
                model.out_dim=768
            return model.eval()
        else:
            raise NotImplementedError("Inconsistent model name and model type")
    
    # VPT
    elif '_vpt' in name:
        if args["model_name"]=="adam_vpt":
            from convs.vpt import build_promptmodel
            if name=="pretrained_vit_b16_224_vpt":
                basicmodelname="vit_base_patch16_224" 
            elif name=="pretrained_vit_b16_224_in21k_vpt":
                basicmodelname="vit_base_patch16_224_in21k"
            
            print("modelname,",name,"basicmodelname",basicmodelname)
            VPT_type="Deep"
            if args["vpt_type"]=='shallow':
                VPT_type="Shallow"
            Prompt_Token_num=args["prompt_token_num"]

            model = build_promptmodel(modelname=basicmodelname,  Prompt_Token_num=Prompt_Token_num, VPT_type=VPT_type)
            prompt_state_dict = model.obtain_prompt()
            model.load_prompt(prompt_state_dict)
            model.out_dim=768
            return model.eval()
        else:
            raise NotImplementedError("Inconsistent model name and model type")

    elif '_adapter' in name:
        ffn_num=args["ffn_num"]
        if args["model_name"]=="adam_adapter" :
            from convs import vision_transformer_adapter
            from easydict import EasyDict
            tuning_config = EasyDict(
                # AdaptFormer
                ffn_adapt=True,
                ffn_option="parallel",
                ffn_adapter_layernorm_option="none",
                ffn_adapter_init_option="lora",
                ffn_adapter_scalar="0.1",
                ffn_num=ffn_num,
                d_model=768,
                # VPT related
                vpt_on=False,
                vpt_num=0,
            )
            if name=="pretrained_vit_b16_224_adapter":
                model = vision_transformer_adapter.vit_base_patch16_224_adapter(num_classes=0,
                    global_pool=False, drop_path_rate=0.0, tuning_config=tuning_config)
                model.out_dim=768
            elif name=="pretrained_vit_b16_224_in21k_adapter":
                model = vision_transformer_adapter.vit_base_patch16_224_in21k_adapter(num_classes=0,
                    global_pool=False, drop_path_rate=0.0, tuning_config=tuning_config)
                model.out_dim=768
            else:
                raise NotImplementedError("Unknown type {}".format(name))
            return model.eval()
        else:
            raise NotImplementedError("Inconsistent model name and model type")

    else:
        raise NotImplementedError("Unknown type {}".format(name))


def load_state_vision_model(model, ckpt_path):
    ckpt_state = torch.load(ckpt_path, map_location='cpu')
    if 'state_dict' in ckpt_state:
        # our upstream converted checkpoint
        ckpt_state = ckpt_state['state_dict']
        prefix = ''
    elif 'model' in ckpt_state:
        # prototype checkpoint
        ckpt_state = ckpt_state['model']
        prefix = 'module.'
    else:
        # official checkpoint
        prefix = ''

    logger = logging.getLogger('global')
    if ckpt_state:
        logger.info('==> Loading model state "{}XXX" from pre-trained model..'.format(prefix))
        
        own_state = model.state_dict()
        state = {}
        for name, param in ckpt_state.items():
            if name.startswith(prefix):
                state[name[len(prefix):]] = param
        success_cnt = 0
        for name, param in state.items():
            if name in own_state:
                if isinstance(param, torch.nn.Parameter):
                    # backwards compatibility for serialized parameters
                    param = param.data
                try:
                    if isinstance(param, bool):
                        own_state[name] = param
                    else:
                        # normal version 
                        own_state[name].copy_(param)
                    success_cnt += 1
                except Exception as err:
                    logger.warn(err)
                    logger.warn('while copying the parameter named {}, '
                                         'whose dimensions in the model are {} and '
                                         'whose dimensions in the checkpoint are {}.'
                                         .format(name, own_state[name].size(), param.size()))
                    logger.warn("But don't worry about it. Continue pretraining.")
        ckpt_keys = set(state.keys())
        own_keys = set(model.state_dict().keys())
        missing_keys = own_keys - ckpt_keys
        logger.info('Successfully loaded {} key(s) from {}'.format(success_cnt, ckpt_path))
        for k in missing_keys:
            logger.warn('Caution: missing key from checkpoint: {}'.format(k))
        redundancy_keys = ckpt_keys - own_keys
        for k in redundancy_keys:
            logger.warn('Caution: redundant key from checkpoint: {}'.format(k))


class BaseNet(nn.Module):
    def __init__(self, args, pretrained):
        super(BaseNet, self).__init__()


        print('This is for the BaseNet initialization.')
        self.convnet = get_convnet(args, pretrained)
        print('After BaseNet initialization.')
        self.fc = None

    @property
    def feature_dim(self):
        return self.convnet.out_dim

    def extract_vector(self, x):
        return self.convnet(x)["features"]

    def forward(self, x):
        x = self.convnet(x)
        out = self.fc(x["features"])
        """
        {
            'fmaps': [x_1, x_2, ..., x_n],
            'features': features
            'logits': logits
        }
        """
        out.update(x)

        return out

    def update_fc(self, nb_classes):
        pass

    def generate_fc(self, in_dim, out_dim):
        pass

    def copy(self):
        return copy.deepcopy(self)

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
        self.eval()

        return self


class IncrementalNet(BaseNet):
    def __init__(self, args, pretrained, gradcam=False):
        super().__init__(args, pretrained)
        self.gradcam = gradcam
        if hasattr(self, "gradcam") and self.gradcam:
            self._gradcam_hooks = [None, None]
            self.set_gradcam_hook()

    def update_fc(self, nb_classes):
        fc = self.generate_fc(self.feature_dim, nb_classes)
        if self.fc is not None:
            nb_output = self.fc.out_features
            weight = copy.deepcopy(self.fc.weight.data)
            bias = copy.deepcopy(self.fc.bias.data)
            fc.weight.data[:nb_output] = weight
            fc.bias.data[:nb_output] = bias

        del self.fc
        self.fc = fc

    def weight_align(self, increment):
        weights = self.fc.weight.data
        newnorm = torch.norm(weights[-increment:, :], p=2, dim=1)
        oldnorm = torch.norm(weights[:-increment, :], p=2, dim=1)
        meannew = torch.mean(newnorm)
        meanold = torch.mean(oldnorm)
        gamma = meanold / meannew
        print("alignweights,gamma=", gamma)
        self.fc.weight.data[-increment:, :] *= gamma

    def generate_fc(self, in_dim, out_dim):
        fc = SimpleLinear(in_dim, out_dim)

        return fc

    def forward(self, x):
        x = self.convnet(x)
        out = self.fc(x["features"])
        out.update(x)
        if hasattr(self, "gradcam") and self.gradcam:
            out["gradcam_gradients"] = self._gradcam_gradients
            out["gradcam_activations"] = self._gradcam_activations

        return out

    def unset_gradcam_hook(self):
        self._gradcam_hooks[0].remove()
        self._gradcam_hooks[1].remove()
        self._gradcam_hooks[0] = None
        self._gradcam_hooks[1] = None
        self._gradcam_gradients, self._gradcam_activations = [None], [None]

    def set_gradcam_hook(self):
        self._gradcam_gradients, self._gradcam_activations = [None], [None]

        def backward_hook(module, grad_input, grad_output):
            self._gradcam_gradients[0] = grad_output[0]
            return None

        def forward_hook(module, input, output):
            self._gradcam_activations[0] = output
            return None

        self._gradcam_hooks[0] = self.convnet.last_conv.register_backward_hook(
            backward_hook
        )
        self._gradcam_hooks[1] = self.convnet.last_conv.register_forward_hook(
            forward_hook
        )

class IL2ANet(IncrementalNet):

    def update_fc(self, num_old, num_total, num_aux):
        fc = self.generate_fc(self.feature_dim, num_total+num_aux)
        if self.fc is not None:
            weight = copy.deepcopy(self.fc.weight.data)
            bias = copy.deepcopy(self.fc.bias.data)
            fc.weight.data[:num_old] = weight[:num_old]
            fc.bias.data[:num_old] = bias[:num_old]
        del self.fc
        self.fc = fc

class CosineIncrementalNet(BaseNet):
    def __init__(self, args, pretrained, nb_proxy=1):
        super().__init__(args, pretrained)
        self.nb_proxy = nb_proxy

    def update_fc(self, nb_classes, task_num):
        fc = self.generate_fc(self.feature_dim, nb_classes)
        if self.fc is not None:
            if task_num == 1:
                fc.fc1.weight.data = self.fc.weight.data
                fc.sigma.data = self.fc.sigma.data
            else:
                prev_out_features1 = self.fc.fc1.out_features
                fc.fc1.weight.data[:prev_out_features1] = self.fc.fc1.weight.data
                fc.fc1.weight.data[prev_out_features1:] = self.fc.fc2.weight.data
                fc.sigma.data = self.fc.sigma.data

        del self.fc
        self.fc = fc

    def generate_fc(self, in_dim, out_dim):
        if self.fc is None:
            fc = CosineLinear(in_dim, out_dim, self.nb_proxy, to_reduce=True)
        else:
            prev_out_features = self.fc.out_features // self.nb_proxy
            # prev_out_features = self.fc.out_features
            fc = SplitCosineLinear(
                in_dim, prev_out_features, out_dim - prev_out_features, self.nb_proxy
            )

        return fc


class BiasLayer(nn.Module):
    def __init__(self):
        super(BiasLayer, self).__init__()
        self.alpha = nn.Parameter(torch.ones(1, requires_grad=True))
        self.beta = nn.Parameter(torch.zeros(1, requires_grad=True))

    def forward(self, x, low_range, high_range):
        ret_x = x.clone()
        ret_x[:, low_range:high_range] = (
            self.alpha * x[:, low_range:high_range] + self.beta
        )
        return ret_x

    def get_params(self):
        return (self.alpha.item(), self.beta.item())


class IncrementalNetWithBias(BaseNet):
    def __init__(self, args, pretrained, bias_correction=False):
        super().__init__(args, pretrained)

        # Bias layer
        self.bias_correction = bias_correction
        self.bias_layers = nn.ModuleList([])
        self.task_sizes = []

    def forward(self, x):
        x = self.convnet(x)
        out = self.fc(x["features"])
        if self.bias_correction:
            logits = out["logits"]
            for i, layer in enumerate(self.bias_layers):
                logits = layer(
                    logits, sum(self.task_sizes[:i]), sum(self.task_sizes[: i + 1])
                )
            out["logits"] = logits

        out.update(x)

        return out

    def update_fc(self, nb_classes):
        fc = self.generate_fc(self.feature_dim, nb_classes)
        if self.fc is not None:
            nb_output = self.fc.out_features
            weight = copy.deepcopy(self.fc.weight.data)
            bias = copy.deepcopy(self.fc.bias.data)
            fc.weight.data[:nb_output] = weight
            fc.bias.data[:nb_output] = bias

        del self.fc
        self.fc = fc

        new_task_size = nb_classes - sum(self.task_sizes)
        self.task_sizes.append(new_task_size)
        self.bias_layers.append(BiasLayer())

    def generate_fc(self, in_dim, out_dim):
        fc = SimpleLinear(in_dim, out_dim)

        return fc

    def get_bias_params(self):
        params = []
        for layer in self.bias_layers:
            params.append(layer.get_params())

        return params

    def unfreeze(self):
        for param in self.parameters():
            param.requires_grad = True


class DERNet(nn.Module):
    def __init__(self, args, pretrained):
        super(DERNet, self).__init__()
        self.convnet_type = args["convnet_type"]
        self.convnets = nn.ModuleList()
        self.pretrained = pretrained
        self.out_dim = None
        self.fc = None
        self.aux_fc = None
        self.task_sizes = []
        self.args = args

    @property
    def feature_dim(self):
        if self.out_dim is None:
            return 0
        return self.out_dim * len(self.convnets)

    def extract_vector(self, x):
        features = [convnet(x)["features"] for convnet in self.convnets]
        features = torch.cat(features, 1)
        return features

    def forward(self, x):
        features = [convnet(x)["features"] for convnet in self.convnets]
        features = torch.cat(features, 1)

        out = self.fc(features)  # {logics: self.fc(features)}

        aux_logits = self.aux_fc(features[:, -self.out_dim :])["logits"]

        out.update({"aux_logits": aux_logits, "features": features})
        return out
        """
        {
            'features': features
            'logits': logits
            'aux_logits':aux_logits
        }
        """

    def update_fc(self, nb_classes):
        if len(self.convnets) == 0:
            self.convnets.append(get_convnet(self.args))
        else:
            self.convnets.append(get_convnet(self.args))
            self.convnets[-1].load_state_dict(self.convnets[-2].state_dict())

        if self.out_dim is None:
            self.out_dim = self.convnets[-1].out_dim
        fc = self.generate_fc(self.feature_dim, nb_classes)
        if self.fc is not None:
            nb_output = self.fc.out_features
            weight = copy.deepcopy(self.fc.weight.data)
            bias = copy.deepcopy(self.fc.bias.data)
            fc.weight.data[:nb_output, : self.feature_dim - self.out_dim] = weight
            fc.bias.data[:nb_output] = bias

        del self.fc
        self.fc = fc

        new_task_size = nb_classes - sum(self.task_sizes)
        self.task_sizes.append(new_task_size)

        self.aux_fc = self.generate_fc(self.out_dim, new_task_size + 1)

    def generate_fc(self, in_dim, out_dim):
        fc = SimpleLinear(in_dim, out_dim)

        return fc

    def copy(self):
        return copy.deepcopy(self)

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
        self.eval()

        return self

    def freeze_conv(self):
        for param in self.convnets.parameters():
            param.requires_grad = False
        self.convnets.eval()

    def weight_align(self, increment):
        weights = self.fc.weight.data
        newnorm = torch.norm(weights[-increment:, :], p=2, dim=1)
        oldnorm = torch.norm(weights[:-increment, :], p=2, dim=1)
        meannew = torch.mean(newnorm)
        meanold = torch.mean(oldnorm)
        gamma = meanold / meannew
        print("alignweights,gamma=", gamma)
        self.fc.weight.data[-increment:, :] *= gamma


class SimpleCosineIncrementalNet(BaseNet):
    def __init__(self, args, pretrained):
        super().__init__(args, pretrained)

    def update_fc(self, nb_classes, nextperiod_initialization=None):
        fc = self.generate_fc(self.feature_dim, nb_classes).cuda()
        if self.fc is not None:
            nb_output = self.fc.out_features
            weight = copy.deepcopy(self.fc.weight.data)
            fc.sigma.data = self.fc.sigma.data
            if nextperiod_initialization is not None:
                weight = torch.cat([weight, nextperiod_initialization])
            else:
                weight = torch.cat([weight, torch.zeros(nb_classes - nb_output, self.feature_dim).cuda()])
            fc.weight = nn.Parameter(weight)
        del self.fc
        self.fc = fc

    def generate_fc(self, in_dim, out_dim):
        fc = CosineLinear(in_dim, out_dim)
        return fc


class SimpleVitNet(BaseNet):
    def __init__(self, args, pretrained):
        super().__init__(args, pretrained)

    def update_fc(self, nb_classes, nextperiod_initialization=None):
        fc = self.generate_fc(self.feature_dim, nb_classes).cuda()
        if self.fc is not None:
            nb_output = self.fc.out_features
            weight = copy.deepcopy(self.fc.weight.data)
            fc.sigma.data = self.fc.sigma.data
            if nextperiod_initialization is not None:
                weight = torch.cat([weight, nextperiod_initialization])
            else:
                weight = torch.cat([weight, torch.zeros(nb_classes - nb_output, self.feature_dim).cuda()])
            fc.weight = nn.Parameter(weight)
        del self.fc
        self.fc = fc

    def generate_fc(self, in_dim, out_dim):
        fc = CosineLinear(in_dim, out_dim)
        return fc

    def extract_vector(self, x):
        return self.convnet(x)

    def forward(self, x):
        x = self.convnet(x)
        out = self.fc(x)
        # out.update(x)
        return out


class MultiBranchCosineIncrementalNet(BaseNet):
    def __init__(self, args, pretrained):
        super().__init__(args, pretrained)
        
        # no need the convnet.
        
        print('Clear the convnet in MultiBranchCosineIncrementalNet, since we are using self.convnets with dual branches')
        self.convnet=torch.nn.Identity()
        for param in self.convnet.parameters():
            param.requires_grad = False

        self.convnets = nn.ModuleList()
        self.args=args
        
        if 'resnet' in args['convnet_type']:
            self.modeltype='cnn'
        else:
            self.modeltype='vit'

    def update_fc(self, nb_classes, nextperiod_initialization=None):
        fc = self.generate_fc(self._feature_dim, nb_classes).cuda()
        if self.fc is not None:
            nb_output = self.fc.out_features
            weight = copy.deepcopy(self.fc.weight.data)
            fc.sigma.data = self.fc.sigma.data
            if nextperiod_initialization is not None:
                weight = torch.cat([weight, nextperiod_initialization])
            else:
                weight = torch.cat([weight, torch.zeros(nb_classes - nb_output, self._feature_dim).cuda()])
            fc.weight = nn.Parameter(weight)
        del self.fc
        self.fc = fc

    def generate_fc(self, in_dim, out_dim):
        fc = CosineLinear(in_dim, out_dim)
        return fc
    

    def forward(self, x):
        if self.modeltype=='cnn':
            features = [convnet(x)["features"] for convnet in self.convnets]
            features = torch.cat(features, 1)
            # import pdb; pdb.set_trace()
            out = self.fc(features)
            out.update({"features": features})
            return out
        else:
            features = [convnet(x) for convnet in self.convnets]
            features = torch.cat(features, 1)
            # import pdb; pdb.set_trace()
            out = self.fc(features)
            out.update({"features": features})
            return out

    
    def construct_dual_branch_network(self, tuned_model):
        if 'ssf' in self.args['convnet_type']:
            newargs=copy.deepcopy(self.args)
            newargs['convnet_type']=newargs['convnet_type'].replace('_ssf','')
            print(newargs['convnet_type'])
            self.convnets.append(get_convnet(newargs)) #pretrained model without scale
        elif 'vpt' in self.args['convnet_type']:
            newargs=copy.deepcopy(self.args)
            newargs['convnet_type']=newargs['convnet_type'].replace('_vpt','')
            print(newargs['convnet_type'])
            self.convnets.append(get_convnet(newargs)) #pretrained model without vpt
        elif 'adapter' in self.args['convnet_type']:
            newargs=copy.deepcopy(self.args)
            newargs['convnet_type']=newargs['convnet_type'].replace('_adapter','')
            print(newargs['convnet_type'])
            self.convnets.append(get_convnet(newargs)) #pretrained model without adapter
        else:
            self.convnets.append(get_convnet(self.args)) #the pretrained model itself

        self.convnets.append(tuned_model.convnet) #adappted tuned model
    
        self._feature_dim = self.convnets[0].out_dim * len(self.convnets) 
        self.fc=self.generate_fc(self._feature_dim,self.args['init_cls'])
        

    

class FOSTERNet(nn.Module):
    def __init__(self, args, pretrained):
        super(FOSTERNet, self).__init__()
        self.convnet_type = args["convnet_type"]
        self.convnets = nn.ModuleList()
        self.pretrained = pretrained
        self.out_dim = None
        self.fc = None
        self.fe_fc = None
        self.task_sizes = []
        self.oldfc = None
        self.args = args

    @property
    def feature_dim(self):
        if self.out_dim is None:
            return 0
        return self.out_dim * len(self.convnets)

    def extract_vector(self, x):
        features = [convnet(x)["features"] for convnet in self.convnets]
        features = torch.cat(features, 1)
        return features

    def forward(self, x):
        features = [convnet(x)["features"] for convnet in self.convnets]
        features = torch.cat(features, 1)
        out = self.fc(features)
        fe_logits = self.fe_fc(features[:, -self.out_dim :])["logits"]

        out.update({"fe_logits": fe_logits, "features": features})

        if self.oldfc is not None:
            old_logits = self.oldfc(features[:, : -self.out_dim])["logits"]
            out.update({"old_logits": old_logits})

        out.update({"eval_logits": out["logits"]})
        return out

    def update_fc(self, nb_classes):
        self.convnets.append(get_convnet(self.args))
        if self.out_dim is None:
            self.out_dim = self.convnets[-1].out_dim
        fc = self.generate_fc(self.feature_dim, nb_classes)
        if self.fc is not None:
            nb_output = self.fc.out_features
            weight = copy.deepcopy(self.fc.weight.data)
            bias = copy.deepcopy(self.fc.bias.data)
            fc.weight.data[:nb_output, : self.feature_dim - self.out_dim] = weight
            fc.bias.data[:nb_output] = bias
            self.convnets[-1].load_state_dict(self.convnets[-2].state_dict())

        self.oldfc = self.fc
        self.fc = fc
        new_task_size = nb_classes - sum(self.task_sizes)
        self.task_sizes.append(new_task_size)
        self.fe_fc = self.generate_fc(self.out_dim, nb_classes)

    def generate_fc(self, in_dim, out_dim):
        fc = SimpleLinear(in_dim, out_dim)
        return fc

    def copy(self):
        return copy.deepcopy(self)

    def copy_fc(self, fc):
        weight = copy.deepcopy(fc.weight.data)
        bias = copy.deepcopy(fc.bias.data)
        n, m = weight.shape[0], weight.shape[1]
        self.fc.weight.data[:n, :m] = weight
        self.fc.bias.data[:n] = bias

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
        self.eval()
        return self

    def freeze_conv(self):
        for param in self.convnets.parameters():
            param.requires_grad = False
        self.convnets.eval()

    def weight_align(self, old, increment, value):
        weights = self.fc.weight.data
        newnorm = torch.norm(weights[-increment:, :], p=2, dim=1)
        oldnorm = torch.norm(weights[:-increment, :], p=2, dim=1)
        meannew = torch.mean(newnorm)
        meanold = torch.mean(oldnorm)
        gamma = meanold / meannew * (value ** (old / increment))
        logging.info("align weights, gamma = {} ".format(gamma))
        self.fc.weight.data[-increment:, :] *= gamma
