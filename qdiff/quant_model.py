import logging
import torch.nn as nn
from qdiff.quant_block import get_specials, BaseQuantBlock
from qdiff.quant_block import QuantBasicTransformerBlock, QuantResBlock
from qdiff.quant_block import QuantQKMatMul, QuantSMVMatMul, QuantBasicTransformerBlock, QuantAttnBlock
from qdiff.quant_layer import QuantModule, StraightThrough
from ldm.modules.attention import BasicTransformerBlock

logger = logging.getLogger(__name__)


class QuantModel(nn.Module):

    def __init__(self, model: nn.Module, weight_quant_params: dict = {}, act_quant_params: dict = {}, **kwargs):
        super().__init__()
        self.model = model
        self.sm_abit = kwargs.get('sm_abit', 8)
        self.in_channels = model.in_channels
        if hasattr(model, 'image_size'):
            self.image_size = model.image_size
        self.specials = get_specials(act_quant_params['leaf_param'])
        self.quant_module_refactor(self.model, weight_quant_params, act_quant_params)
        self.quant_block_refactor(self.model, weight_quant_params, act_quant_params)

    def quant_module_refactor(self, module: nn.Module, weight_quant_params: dict = {}, act_quant_params: dict = {}):
        """
        Recursively replace the normal layers (conv2D, conv1D, Linear etc.) to QuantModule
        :param module: nn.Module with nn.Conv2d, nn.Conv1d, or nn.Linear in its children
        :param weight_quant_params: quantization parameters like n_bits for weight quantizer
        :param act_quant_params: quantization parameters like n_bits for activation quantizer
        """
        prev_quantmodule = None
        for name, child_module in module.named_children():
            if isinstance(child_module, (nn.Conv2d, nn.Conv1d, nn.Linear)): # nn.Conv1d
                # if isinstance(child_module, (nn.Conv2d, nn.Conv1d)) and name.split('.')[-1] != 'proj_out':
                if isinstance(child_module, (nn.Conv2d, nn.Conv1d)):
                    sd = True
                else:
                    sd = False
                setattr(module, name, QuantModule(
                    child_module, weight_quant_params, act_quant_params, sd=sd))
                prev_quantmodule = getattr(module, name)

            elif isinstance(child_module, StraightThrough):
                continue

            else:
                self.quant_module_refactor(child_module, weight_quant_params, act_quant_params)

    def quant_block_refactor(self, module: nn.Module, weight_quant_params: dict = {}, act_quant_params: dict = {}):
        for name, child_module in module.named_children():
            if type(child_module) in self.specials:
                if self.specials[type(child_module)] in [QuantBasicTransformerBlock, QuantAttnBlock]:
                    setattr(module, name, self.specials[type(child_module)](child_module,
                        act_quant_params, sm_abit=self.sm_abit))
                elif self.specials[type(child_module)] == QuantSMVMatMul:
                    setattr(module, name, self.specials[type(child_module)](
                        act_quant_params, sm_abit=self.sm_abit))
                elif self.specials[type(child_module)] == QuantQKMatMul:
                    setattr(module, name, self.specials[type(child_module)](
                        act_quant_params))
                else:
                    setattr(module, name, self.specials[type(child_module)](child_module, 
                        act_quant_params))
            else:
                self.quant_block_refactor(child_module, weight_quant_params, act_quant_params)

    def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False):
        for m in self.model.modules():
            if isinstance(m, (QuantModule, BaseQuantBlock)):
                m.set_quant_state(weight_quant, act_quant)

    def set_use_sd(self, use_sd):
        for m in self.model.modules():
            if isinstance(m, (QuantModule, BaseQuantBlock)):
                m.set_use_sd(use_sd)

    def set_real_time(self, real_time):
        for m in self.model.modules():
            if isinstance(m, (QuantModule, QuantAttnBlock, QuantQKMatMul, QuantSMVMatMul)):
                m.set_real_time(real_time)

    def set_full_prec(self, full_prec):
        for m in self.model.modules():
            if isinstance(m, (QuantModule, QuantAttnBlock, QuantQKMatMul, QuantSMVMatMul)):
                m.set_full_prec(full_prec)

    def reset_sd(self):
        for m in self.model.modules():
            if isinstance(m, (QuantModule, QuantAttnBlock, QuantQKMatMul, QuantSMVMatMul)):
                m.reset_sd()

    def copy_sd(self, model):
        for name1, module1 in self.model.named_modules():
            for name2, module2 in model.model.named_modules():
                if isinstance(module1, (QuantModule)) and isinstance(module2, (QuantModule)):
                    if name1 == name2 and module2.delta != None:
                        module1.delta = module2.delta.clone()
                        module1.sigma = module2.sigma.clone()
                if isinstance(module1, (QuantAttnBlock)) and isinstance(module2, (QuantAttnBlock)):
                    if name1 == name2 and module2.delta != None:
                        module1.delta = module2.delta.clone()
                        module1.sigma = module2.sigma.clone()
                if isinstance(module1, (QuantQKMatMul)) and isinstance(module2, (QuantQKMatMul)):
                    if name1 == name2 and module2.delta != None:
                        module1.delta = module2.delta.clone()
                        module1.sigma = module2.sigma.clone()
                if isinstance(module1, (QuantSMVMatMul)) and isinstance(module2, (QuantSMVMatMul)):
                    if name1 == name2 and module2.delta != None:
                        module1.delta = module2.delta.clone()
                        module1.sigma = module2.sigma.clone()

    def forward(self, x, timesteps=None, context=None):
        return self.model(x, timesteps, context)
    
    def set_running_stat(self, running_stat: bool, sm_only=False):
        for m in self.model.modules():
            if isinstance(m, QuantBasicTransformerBlock):
                if sm_only:
                    m.attn1.act_quantizer_w.running_stat = running_stat
                    m.attn2.act_quantizer_w.running_stat = running_stat
                else:
                    m.attn1.act_quantizer_q.running_stat = running_stat
                    m.attn1.act_quantizer_k.running_stat = running_stat
                    m.attn1.act_quantizer_v.running_stat = running_stat
                    m.attn1.act_quantizer_w.running_stat = running_stat
                    m.attn2.act_quantizer_q.running_stat = running_stat
                    m.attn2.act_quantizer_k.running_stat = running_stat
                    m.attn2.act_quantizer_v.running_stat = running_stat
                    m.attn2.act_quantizer_w.running_stat = running_stat
            if isinstance(m, QuantModule) and not sm_only:
                m.set_running_stat(running_stat)

    def set_grad_ckpt(self, grad_ckpt: bool):
        for name, m in self.model.named_modules():
            if isinstance(m, (QuantBasicTransformerBlock, BasicTransformerBlock)):
                # logger.info(name)
                m.checkpoint = grad_ckpt
            # elif isinstance(m, QuantResBlock):
                # logger.info(name)
                # m.use_checkpoint = grad_ckpt

