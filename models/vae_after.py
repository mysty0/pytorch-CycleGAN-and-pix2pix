import torch
from torch import nn
from torch.autograd import Variable

import importlib
import numpy as np

from util import ssim
from .base_model import BaseModel
from . import networks


import torchvision
import yaml
    
def instantiate_from_config(config):
    if not "target" in config:
        if config == '__is_first_stage__':
            return None
        elif config == "__is_unconditional__":
            return None
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))


def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)

def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self

class DCSModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        For pix2pix, we do not use image buffer
        The training objective is: GAN Loss + lambda_L1 * ||G(A)-B||_1
        By default, we use vanilla GAN loss, UNet with batchnorm, and aligned datasets.
        """
        # changing the default values to match the pix2pix paper (https://phillipi.github.io/pix2pix/)
        parser.set_defaults(norm='batch', netG='unet_128', dataset_mode='aligned')
        if is_train:
            parser.set_defaults(pool_size=0, gan_mode='vanilla')
            parser.add_argument('--lambda_L1', type=float, default=100.0, help='weight for L1 loss')

        return parser

    def __init__(self, opt):
        """Initialize the pix2pix class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['G', 'SSIM']#['G_GAN', 'G_L1', 'D_real', 'D_fake']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        self.visual_names = ['real_A', 'fake_B', 'real_B']

        self.model_names = ['G']

        self.vae = None

        input_nc, output_nc = opt.input_nc, opt.output_nc
        if opt.vae:
            config = yaml.safe_load(open('./checkpoints/vae2/config.yaml'))
            print("config:")
            print(config)
            self.vae = instantiate_from_config(config['model']).cuda()

            ckpt = torch.load('./checkpoints/vae2/animevae.pt')
            loss = []
            for i in ckpt["state_dict"].keys():
                if i[0:4] == "loss":
                    loss.append(i)
            for i in loss:
                del ckpt["state_dict"][i]

            self.vae.load_state_dict(ckpt["state_dict"])

            self.vae = self.vae.eval()
            self.vae.train = disabled_train
            for param in self.vae.parameters():
                param.requires_grad = False
        
        input_nc, output_nc = 4, 4

        self.netG = networks.define_G(input_nc, output_nc, opt.ngf, opt.netG, opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:  # define a discriminator; conditional GANs need to take both input and output images; Therefore, #channels for D is input_nc + output_nc
            if opt.netD == 'vgg':
                self.netVGG = VggLoss('vgg11')
                self.loss_names.append('VGG')
            if opt.netD == 'vgg19':
                self.netVGG = VggLoss('vgg19')
                self.loss_names.append('VGG')
            if opt.netD == 'perceptual':
                self.netVGG = VGGPerceptualLoss()
                self.loss_names.append('VGG')
        #    self.netD = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf, opt.netD,
        #                                  opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:
            # define loss functions
            #self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionL1 = torch.nn.L1Loss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            #self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            #self.optimizers.append(self.optimizer_D)

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap images in domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def encode_img_latents(self, imgs):

        img_arr = imgs#.float()#.permute(3, 1, 2, 0)#(0, 3, 1, 2)
        img_arr = 2 * (img_arr - 0.5)

        latent_dists = self.vae.encode(img_arr.to(self.device))
        latent_samples = latent_dists.sample()
        #latent_samples *= 0.18215

        return latent_samples

    def decode_img_latents(self, latents):
        latents = 1 / 0.18215 * latents

        with torch.no_grad():
            imgs = self.vae.decode(latents)

        imgs = (imgs / 2 + 0.5).clamp(0, 1)

        return imgs #pil_images

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""

        img = self.real_A
        if self.vae:
            img = self.vae.encode(self.real_A).sample()

        self.fake_B = self.netG(img)  # G(A)

        if self.vae:
            self.fake_B_latent = self.fake_B
            self.fake_B = self.vae.decode(self.fake_B)


    def backward_G(self):

        r_ssim = (1 - ssim.ssim(self.real_B[:,[0],:,:], self.fake_B[:,[0],:,:], data_range=1.0, size_average=False, win_size=9)) / 2
        g_ssim = (1- ssim.ssim(self.real_B[:,[1],:,:], self.fake_B[:,[1],:,:], data_range=1.0, size_average=False, win_size=9)) / 2
        b_ssim = (1 - ssim.ssim(self.real_B[:,[2],:,:], self.fake_B[:,[2],:,:], data_range=1.0, size_average=False, win_size=9)) / 2
        
        if hasattr(self, 'netVGG'):
            self.set_requires_grad(self.netVGG, False)
            self.loss_VGG = self.netVGG(self.real_B, self.fake_B)# * 4
        else:
            self.loss_VGG = 0

        self.loss_SSIM = (r_ssim + g_ssim + b_ssim) / 3 

        #self.loss_G = r_ssim + g_ssim + b_ssim
        self.loss_G = self.loss_VGG + self.loss_SSIM
        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()                   # compute fake images: G(A)
        
        # update D
        #self.set_requires_grad(self.netD, True)  # enable backprop for D
        #self.optimizer_D.zero_grad()     # set D's gradients to zero
        #self.backward_D()                # calculate gradients for D
        #self.optimizer_D.step()          # update D's weights
        # update G
       # self.set_requires_grad(self.netD, False)  # D requires no gradients when optimizing G
        self.optimizer_G.zero_grad()        # set G's gradients to zero
        self.backward_G()                   # calculate graidents for G
        self.optimizer_G.step()             # update G's weights
