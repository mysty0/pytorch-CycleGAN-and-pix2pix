from collections import OrderedDict
import torch
from torch import nn
from torch.autograd import Variable

import importlib
import numpy as np
from models.unet_orig import UNetModel

from util import ssim
from .base_model import BaseModel
from . import networks


import torchvision
import yaml

class VGGPerceptualLoss(torch.nn.Module):
    def __init__(self, resize=True):
        super(VGGPerceptualLoss, self).__init__()
        blocks = []
        blocks.append(torchvision.models.vgg16(pretrained=True).features[:4].cuda().eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[4:9].cuda().eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[9:16].cuda().eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[16:23].cuda().eval())
        for bl in blocks:
            for p in bl.parameters():
                p.requires_grad = False
        self.blocks = torch.nn.ModuleList(blocks)
        self.transform = torch.nn.functional.interpolate
        self.resize = resize
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).cuda())
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).cuda())

    def forward(self, input, target, feature_layers=[0, 1, 2, 3], style_layers=[]):
        if input.shape[1] != 3:
            input = input.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)
        input = (input-self.mean) / self.std
        target = (target-self.mean) / self.std
        if self.resize:
            input = self.transform(input, mode='bilinear', size=(224, 224), align_corners=False)
            target = self.transform(target, mode='bilinear', size=(224, 224), align_corners=False)
        loss = 0.0
        x = input
        y = target
        for i, block in enumerate(self.blocks):
            x = block(x)
            y = block(y)
            if i in feature_layers:
                loss += torch.nn.functional.l1_loss(x, y)
            if i in style_layers:
                act_x = x.reshape(x.shape[0], x.shape[1], -1)
                act_y = y.reshape(y.shape[0], y.shape[1], -1)
                gram_x = act_x @ act_x.permute(0, 2, 1)
                gram_y = act_y @ act_y.permute(0, 2, 1)
                loss += torch.nn.functional.l1_loss(gram_x, gram_y)
        return loss

class VggLoss(nn.Module):
    def __init__(self, name):
        super(VggLoss, self).__init__()
        #print("Loading VGG19 model from path: {}".format(model_paths["vgg"]))

        vgg_name = f"torchvision.models.vgg"
        print(f"vgg name {name}")
        lib = importlib.import_module(vgg_name)
        for lname, cls in lib.__dict__.items():
            if lname.lower() == name.lower():
                vgg = cls()

        state_dict = torch.load(f'./checkpoints/{name}.pth')
        vgg.load_state_dict(state_dict)

        self.vgg_model = torch.nn.Sequential(*(list(vgg.children())[:-1]))
        #self.vgg_model.load_state_dict(torch.load(model_paths['vgg']))
        self.vgg_model.cuda()
        self.vgg_model.eval()

        self.l1loss = torch.nn.L1Loss()

    def forward(self, input_photo, output):
        vgg_photo = self.vgg_model(input_photo)
        vgg_output = self.vgg_model(output)
        #n, c, h, w = vgg_photo.shape
        # h, w, c = vgg_photo.get_shape().as_list()[1:]
        loss = self.l1loss(vgg_photo, vgg_output)

        return loss

class DCSModel(BaseModel):
    """ This class implements the pix2pix model, for learning a mapping from input images to output images given paired data.

    The model training requires '--dataset_mode aligned' dataset.
    By default, it uses a '--netG unet256' U-Net generator,
    a '--netD basic' discriminator (PatchGAN),
    and a '--gan_mode' vanilla GAN loss (the cross-entropy objective used in the orignal GAN paper).

    pix2pix paper: https://arxiv.org/pdf/1611.07004.pdf
    """
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
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        #if self.isTrain:
        #    self.model_names = ['G', 'D']
        #else:  # during test time, only load G
        self.model_names = ['G']
        # define networks (both generator and discriminator)

        self.vae = None

        input_nc, output_nc = opt.input_nc, opt.output_nc
        if opt.vae:
            self.vae = networks.get_vae()
        
        input_nc, output_nc = 4, 4

        #self.netG = networks.define_G(input_nc, output_nc, opt.ngf, opt.netG, opt.norm,
        #                              not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        self.netG = UNetModel(
            in_channels=4,
            out_channels=4,
            channels=64,#320,
            n_res_blocks=2,
            attention_levels=[4,2,1],
            channel_multipliers=[1,2,4,4],
            n_heads=8,
            tf_layers=1,
            d_cond=128#768
        ).cuda()

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

    #@torch.no_grad()
    def encode_img_latents(self, imgs):
        #if not isinstance(imgs, list):
        #    imgs = [imgs]

       # img_arr = np.stack([np.array(img) for img in imgs], axis=0)
        #print(imgs.shape)
        #img_arr = imgs / 255.0
        img_arr = imgs#.float()#.permute(3, 1, 2, 0)#(0, 3, 1, 2)
        img_arr = 2 * (img_arr - 0.5)

        latent_dists = self.vae.encode(img_arr.to(self.device))
        latent_samples = latent_dists.sample()
        latent_samples *= 0.18215

        return latent_samples

    #@torch.no_grad()
    def decode_img_latents(self, latents):
        latents = 1 / 0.18215 * latents

        with torch.no_grad():
            imgs = self.vae.decode(latents)

        imgs = (imgs / 2 + 0.5).clamp(0, 1)
        #imgs = imgs.detach().cpu()#.permute(0, 2, 3, 1)#(0, 2, 3, 1)#.numpy()
        #imgs = (imgs * 255).round().astype('uint8')
        #pil_images = [Image.fromarray(image) for image in imgs]
        return imgs #pil_images

    # @torch.no_grad()
    # #@torch.autocast("cuda", enabled=True, dtype=torch.float16)
    # def encode_image(self, image, model):
    #     #if isinstance(image, Image.Image):
    #     #    image = np.array(image)
    #     #    image = torch.from_numpy(image)

    #     if isinstance(image, np.ndarray):
    #         image = torch.from_numpy(image)

    #     #dtype = image.dtype
    #     image = image.to(torch.float32)
    #     #gets image as numpy array and returns as tensor
    #     def preprocess_vqgan(x):
    #         x = x / 255.0
    #         x = 2.*x - 1.
    #         return x

    #     image = image.permute(2, 0, 1).unsqueeze(0).float().cuda()
    #     image = preprocess_vqgan(image)
    #     image = model.encode(image).sample()
    #     #image = image.to(dtype)

    #     return image

    # @torch.no_grad()
    # def decode_image(self, image, model):
    #     def custom_to_pil(x):
    #         x = x.detach().float().cpu()
    #         x = torch.clamp(x, -1., 1.)
    #         x = (x + 1.)/2.
    #         x = x.permute(0, 2, 3, 1)#.numpy()
    #         #x = (255*x).astype(np.uint8)
    #         #x = Image.fromarray(x)
    #         #if not x.mode == "RGB":
    #         #    x = x.convert("RGB")
    #         return x

    #     image = model.decode(image)
    #     image = custom_to_pil(image)
    #     return image

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""

        img = self.real_A
        if self.vae:
            img = self.vae.encode(self.real_A).sample()
            #img = Variable(img, requires_grad = True)

        self.fake_B = self.netG(img, torch.tensor([0]).cuda(), torch.zeros([1, 8, 8], dtype=torch.float32).cuda())  # G(A)

        if self.vae:
            #with torch.no_grad():
            self.fake_B_latent = self.fake_B
            self.fake_B = self.vae.decode(self.fake_B)


    #def get_current_visuals(self):
    #    """Return visualization images. train.py will display these images with visdom, and save the images to a HTML"""
        #visual_ret = OrderedDict()
        #for name in self.visual_names:
        #    if isinstance(name, str):
        #        visual_ret[name] = getattr(self, name)
        #return visual_ret



    # def backward_D(self):
    #     """Calculate GAN loss for the discriminator"""
    #     # Fake; stop backprop to the generator by detaching fake_B
    #     fake_AB = torch.cat((self.real_A, self.fake_B), 1)  # we use conditional GANs; we need to feed both input and output to the discriminator
    #     pred_fake = self.netD(fake_AB.detach())
    #     self.loss_D_fake = self.criterionGAN(pred_fake, False)
    #     # Real
    #     real_AB = torch.cat((self.real_A, self.real_B), 1)
    #     pred_real = self.netD(real_AB)
    #     self.loss_D_real = self.criterionGAN(pred_real, True)
    #     # combine loss and calculate gradients
    #     self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
    #     self.loss_D.backward()

    def backward_G(self):
        """Calculate GAN and L1 loss for the generator"""
        # First, G(A) should fake the discriminator
        #fake_AB = torch.cat((self.real_A, self.fake_B), 1)
        #pred_fake = self.netD(fake_AB)
        #self.loss_G_GAN = self.criterionGAN(pred_fake, True)
        # Second, G(A) = B
        #self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) * self.opt.lambda_L1
        # combine loss and calculate gradients
        #self.loss_G = self.loss_G_GAN + self.loss_G_L1
        #print(self.real_A.shape)
        if self.vae:
            img = self.vae.encode(self.real_B).sample()
            #loss = self.criterionL1(self.fake_B_latent, img)
            loss = torch.nn.functional.mse_loss(self.fake_B_latent, img)
            self.loss_SSIM = 0
            self.loss_VGG = 0
            self.loss_G = loss
            self.loss_G.backward()
            print(loss)
            return 

        
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
