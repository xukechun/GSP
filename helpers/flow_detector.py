import cv2
import numpy as np
import torch
import torch.optim
import torch.utils.data
from models.RAFT.core.raft import RAFT
from models.RAFT.core.utils import flow_viz
from models.RAFT.core.utils.utils import InputPadder


class OpticalFlowNet:
    def __init__(self, args):
        self.device = args.device 
        self.args = args

    def viz_magnitude(self, flow, stride=10):
        flow = flow[0].permute(1,2,0).cpu().numpy()
        flow = flow.copy()
        flow[:,:,0] = -flow[:,:,0]

        height, width, _ = flow.shape
        xx = np.arange(0,height,stride)
        yy = np.arange(0,width,stride)
        X, Y = np.meshgrid(xx,yy)
        X = X.flatten()
        Y = Y.flatten()

        # sample
        sample_0 = flow[:, :, 0][xx]
        sample_0 = sample_0.T
        sample_x = sample_0[yy]
        sample_x = sample_x.T
        sample_1 = flow[:, :, 1][xx]
        sample_1 = sample_1.T
        sample_y = sample_1[yy]
        sample_y = sample_y.T

        sample_x = sample_x[:,:,np.newaxis]
        sample_y = sample_y[:,:,np.newaxis]
        new_flow = np.concatenate([sample_x, sample_y], axis=2)
        flow_x = new_flow[:, :, 0].flatten()
        flow_y = new_flow[:, :, 1].flatten()
        
        import matplotlib.pyplot as plt
        # display
        ax = plt.gca()
        ax.xaxis.set_ticks_position('top')
        ax.invert_yaxis()
        # plt.quiver(X,Y, flow_x, flow_y, angles="xy", color="#666666")
        ax.quiver(X, Y, flow_x, flow_y, color="#666666")
        ax.grid()
        # ax.legend()
        plt.draw()
        plt.show()


    def viz(self, img1, img2, flo):
        img1 = img1[0].permute(1,2,0).cpu().numpy()
        img2 = img2[0].permute(1,2,0).cpu().numpy()
        flo = flo[0].permute(1,2,0).cpu().numpy()
        
        # map flow to rgb image
        flo = flow_viz.flow_to_image(flo)
        img_flo = np.concatenate([img1, img2, flo], axis=0)

        import matplotlib.pyplot as plt
        plt.imshow(img_flo / 255.0)
        plt.show()

        # cv2.imshow('image', img_flo[:, :, [2,1,0]]/255.0)
        # cv2.waitKey()

    def viz_uncertainty(self, img1, flo):

        flo = flo[0].permute(1,2,0).cpu().numpy()

        magnitude, _ = cv2.cartToPolar(flo[..., 0], flo[..., 1])
        print('max flow magnitude: ', np.max(magnitude))
        print('min flow magnitude: ', np.min(magnitude))
        print('mean flow magnitude: ', np.mean(magnitude))

        # map flow to rgb image
        flo = flow_viz.flow_to_image(flo)
        img_flo = np.concatenate([img1, flo], axis=0)

        import matplotlib.pyplot as plt
        plt.imshow(img_flo / 255.0)
        plt.show()

        # cv2.imshow('image', img_flo[:, :, [2,1,0]]/255.0)
        # cv2.waitKey()

    def preprocess(self, image):
        img = np.array(image).astype(np.uint8)
        img = torch.from_numpy(img).permute(2, 0, 1).float()
        return img[None].to(self.device)

    def run(self, image1, image2, vis=False):
        # load pretrained model
        model = torch.nn.DataParallel(RAFT(self.args))
        model.load_state_dict(torch.load(self.args.model))

        model = model.module
        model.to(self.device)
        model.eval()

        with torch.no_grad():
            
            image1 = self.preprocess(image1)
            image2 = self.preprocess(image2)

            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1, image2)
            
            flow_low, flow_up = model(image1, image2, iters=20, test_mode=True)

            if vis:
                self.viz(image1, image2, flow_up)

            return flow_up