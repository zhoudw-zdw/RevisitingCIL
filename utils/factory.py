def get_model(model_name, args):
    name = model_name.lower()
    if name=="simplecil":
        from models.simplecil import Learner
        return Learner(args)
    elif name=="aper_finetune":
        from models.aper_finetune import Learner
        return Learner(args)
    elif name=="aper_ssf":
        from models.aper_ssf import Learner
        return Learner(args)
    elif name=="aper_vpt":
        from models.aper_vpt import Learner
        return Learner(args) 
    elif name=="aper_adapter":
        from models.aper_adapter import Learner
        return Learner(args)
    elif name=="aper_bn":
        from models.aper_bn import Learner
        return Learner(args)
    else:
        assert 0
