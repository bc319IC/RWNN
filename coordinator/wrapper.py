import torch
from coordinator.wrapper_algorithms import WrapperAlgorithms
from models.transform_model import ImageTransform
from helpers.misc import CharbonnierLoss
from helpers.perlin import gen_noise_map
import numpy as np
import torch.nn.functional as F

class Wrapper(WrapperAlgorithms):
    def __init__(self, dae, num_channels = 1, test_nl = 0, decode_depth = 1):
        self.alg = self.dae if dae else self.fusion_denoise
        super().__init__(num_channels)
        self.transform_net = ImageTransform(num_channels = num_channels)

        self.max_nrm_train_nl = 55. / 255.

        self.nrm_test_nl = test_nl / 255.
        self.decode_depth = decode_depth

        self.mse = torch.nn.MSELoss()
        # smooth l1
        self.l1 = CharbonnierLoss()

    def get_training_loss(self, x):
        l = {}
        nrm_sigma = torch.rand(1, device = x.device) * self.max_nrm_train_nl #og

        #noise = torch.cuda.FloatTensor(x.size()).normal_(mean=0, std=self.nrm_test_nl) #AG CG
        #noise = np.zeros((x.size()[2],x.size()[3]),dtype=np.uint8) #SP
        #cv2.randu(noise,0,255) #SP
        #noisew = cv2.threshold(noise,220,255,cv2.THRESH_BINARY)[1]*(-1) #SP
        #cv2.randu(noise,0,255) #SP
        #noiseb = cv2.threshold(noise,220,255,cv2.THRESH_BINARY)[1] #SP
        #noise = noiseb+noisew #SP
        #noise = torch.cuda.FloatTensor(noise) #SP 
        #con_noise = np.concatenate([np.random.normal(0, 15/225, size=(x.size()[2]//2, x.size()[3])),np.random.normal(0, 50/255, size=(x.size()[2]//2, x.size()[3]))]) #CG change +1 accordingly
        #np.random.shuffle(con_noise) #CG
        #con_noise = torch.cuda.FloatTensor(con_noise) #CG
        #inp = torch.poisson(x)#P
        #inp = torch.cuda.FloatTensor(inp)#P
        #inp = x + noise #AG SP
        #inp = x + noise + con_noise #CG

        # Define the Prewitt filter kernels
        prewitt_x = torch.cuda.FloatTensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]])
        prewitt_y = torch.cuda.FloatTensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]])
        # Apply zero padding to the input tensor
        padded_input = F.pad(x, (1, 1, 1, 1), mode='reflect')
        # Apply the Prewitt filter using conv2d
        output_x = F.conv2d(padded_input, prewitt_x.unsqueeze(0).unsqueeze(0))
        output_y = F.conv2d(padded_input, prewitt_y.unsqueeze(0).unsqueeze(0))
        # Compute the gradient magnitude
        edges = torch.sqrt(output_x.pow(2) + output_y.pow(2))
        edges = (100*edges+1e-6)/edges.max()
        noise = torch.normal(0, edges/255).cuda()
        inp = x + noise
        inp = torch.cuda.FloatTensor(inp)

        #inp = x + nrm_sigma * torch.randn_like(x) #og
        res, ne = self.alg(inp, layer = self.decode_depth, show_ne = True)
        l['mse'] = self.mse(res, x)
        l['psnr'] = -10. * torch.log10(l['mse'])
        l['l1'] = self.l1(res, x)
        l['ne'] = self.l1(ne, nrm_sigma)
        if (self.alg == self.fusion_denoise):
            l['oloss'] = self.dn_net.orthogonal_loss()
        l['total loss'] = l['l1'] + l['ne'] + 10. * (l['oloss'] if (self.alg == self.fusion_denoise) else 0.)
        return l, l['total loss']

    def test(self, x, vis = False, map = False):
        l = {}
        map = gen_noise_map(x.shape, self.nrm_test_nl) if map else None

        #noise = torch.cuda.FloatTensor(x.size()).normal_(mean=0, std=self.nrm_test_nl) #AG CG
        #noise = np.zeros((x.size()[2],x.size()[3]),dtype=np.uint8) #SP
        #cv2.randu(noise,0,255) #SP
        #noisew = cv2.threshold(noise,220,255,cv2.THRESH_BINARY)[1]*(-1) #SP
        #cv2.randu(noise,0,255) #SP
        #noiseb = cv2.threshold(noise,220,255,cv2.THRESH_BINARY)[1] #SP
        #noise = noiseb+noisew #SP
        #noise = torch.cuda.FloatTensor(noise) #SP 
        #con_noise = np.concatenate([np.random.normal(0, 15/225, size=(x.size()[2]//2, x.size()[3])),np.random.normal(0, 50/255, size=(x.size()[2]//2, x.size()[3]))]) #CG change +1 accordingly
        #np.random.shuffle(con_noise) #CG
        #con_noise = torch.cuda.FloatTensor(con_noise) #CG
        #inp = x + nrm_sigma * torch.randn_like(x)  #og
        #inp = torch.poisson(x) #P
        #inp = x + noise #AG SP
        #inp = x + noise + con_noise #CG

        # Define the Prewitt filter kernels
        prewitt_x = torch.cuda.FloatTensor([[-1., 0., 1.], [-1., 0., 1.], [-1., 0., 1.]])
        prewitt_y = torch.cuda.FloatTensor([[-1., -1., -1.], [0., 0., 0.], [1., 1., 1.]])
        # Apply zero padding to the input tensor
        padded_input = F.pad(x, (1, 1, 1, 1), mode='constant', value=0)
        # Apply the Prewitt filter using conv2d
        output_x = F.conv2d(padded_input, prewitt_x.unsqueeze(0).unsqueeze(0))
        output_y = F.conv2d(padded_input, prewitt_y.unsqueeze(0).unsqueeze(0))
        # Compute the gradient magnitude
        edges = torch.sqrt(output_x.pow(2) + output_y.pow(2))
        edges = (100*edges+1e-3)/edges.max()
        noise = torch.normal(0, edges/255).cuda()
        inp = x + noise

        #inp = x + (self.nrm_test_nl if map is None else map) * torch.randn_like(x)
        res = self.alg(inp, layer = self.decode_depth, map = map).clip(0., 1.)
        l['mse'] = self.mse(res, x)
        l['psnr'] = -10. * torch.log10(l['mse'])
        l['l1'] = self.l1(res, x)
        l['total loss'] = l['l1']
        if not vis: return l, l['total loss']
        to_vis = {}
        to_vis['original'] = x
        xf, _ = self.transform_net(x, J = 1)
        to_vis['original transformed'] = self.transform_net.collect_pieces(xf)
        to_vis['input'] = inp
        inpf, _ = self.transform_net(inp, J = 1)
        to_vis['input transformed'] = self.transform_net.collect_pieces(inpf)
        to_vis['result'] = res
        return l, l['total loss'], to_vis
