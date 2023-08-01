from custom_model.baselayer import *
import math

class DGCN(nn.Module):
    def __init__(self, para, A, B, return_interpret=False, uncertainty=True):
        super(DGCN, self).__init__()
        self.N = para['nb_node']
        self.F = para['dim_feature']
        self.A = A
        self.B = B
        self.T = para['horizon']
        self.interpret = return_interpret
        self.uncertainty = uncertainty
        
        self.encoder = Encoder(self.N, self.F, self.A, self.B)
        self.decoder = Decoder(self.N, self.F, self.A, self.B, self.T, self.uncertainty)

    def forward(self, x, thred):
        h = self.encoder(x[:,:-self.T-1])
        prediction, mask1, mask2, demand = self.decoder(x[:,-self.T-1:-1], h, thred)

        if self.interpret:
            return prediction, mask1, mask2, demand
        return prediction

class SoftLoss(nn.Module):
    def __init__(self):
        super(SoftLoss, self).__init__()

    def forward(self, inputs, targets):
        error = torch.mean(100*torch.abs(inputs-targets), -1)
        weight = self.get_weights(targets[...,0])
        return torch.mean(error*weight)

    def get_weights(self, label):
        weights = torch.where(label>0.45, 1., 3.)
        return weights
    
class MSE_scale(nn.Module):
    def __init__(self):
        super(MSE_scale, self).__init__()
        self.main = nn.MSELoss()

    def forward(self, inputs, targets):
        return self.main(inputs*10, targets*10)

    def get_weights(self, label):
        weights = torch.where(label>0.45, 1., 3.)
        return weights
    
class DERLoss(nn.Module):
    def __init__(self, gamma):
        super(DERLoss, self).__init__()
        self.gamma = gamma
        self.main = nn.MSELoss()

    def forward(self, inputs, targets):
        mu = inputs[...,0]
        v = inputs[...,1]
        a = inputs[...,2]
        b = inputs[...,3]

        speed = inputs[...,4]
        flow = inputs[...,5]
        #sigma = inputs[...,-1]
        omega = 2*b*(1+v)

        vt = targets[...,0]
        qt = targets[...,1]

        weights = self.get_weights(vt)

        loss1 = torch.abs(flow-qt)*100*weights
        loss1 = torch.mean(loss1)

        # loss3 = torch.abs(speed-vt)*100*weights
        # loss3 = torch.mean(loss3)

        loss3 = torch.sqrt(self.main(speed*100, vt*100))

        #reg = torch.abs(torch.abs(mu-vt)*10 - torch.sqrt(b/(a-1)))/v
        reg = torch.abs(torch.abs(mu-vt)/torch.sqrt(b/(a-1))-1) * (2*v+a)
        #reg = (torch.sqrt(self.main(speed, vt)/b/(v+1)*v*(a-1))) * (2*v+a)
        reg = torch.mean(reg)

        lmain = 0.5*torch.log(math.pi/v) - a*torch.log(omega) + \
            (a+0.5)*torch.log((vt-mu)*(vt-mu)*v+omega) + torch.lgamma(a)-torch.lgamma(a+0.5)
        
        lmain = torch.mean(lmain)
        return loss1 + lmain + self.gamma*reg + loss3
        #print(loss1.size())
        #return lmain
    
    def get_weights(self, label):
        weights = torch.where(label>0.45, 1., 3.)
        return weights

class RegressionLoss(nn.Module):
    def __init__(self, ):
        super(RegressionLoss, self).__init__()
        # self.main = nn.MSELoss()

    def forward(self, inputs, targets):

        speed = inputs[...,0]
        flow = inputs[...,1]

        vt = targets[...,0]
        qt = targets[...,1]

        weights = self.get_weights(vt)

        loss1 = torch.abs(flow-qt)*100*weights
        loss1 = torch.mean(loss1)

        loss2 = torch.abs(speed-vt)*100*weights
        loss2 = torch.mean(loss2)

        return loss1 + loss2
    
    def get_weights(self, label):
        weights = torch.where(label>0.45, 1., 3.)
        return weights
    
class DGCN_vis(nn.Module):
    def __init__(self, para, A, B, return_interpret=False):
        super(DGCN_vis, self).__init__()
        self.N = para['nb_node']
        self.F = para['dim_feature']
        self.A = A
        self.B = B
        self.T = para['horizon']
        self.interpret = return_interpret
        
        self.encoder = Encoder_vis(self.N, self.F, self.A, self.B)
        self.decoder = Decoder_vis(self.N, self.F, self.A, self.B, self.T)

    def forward(self, x):
        h = self.encoder(x[:,:-self.T-1])
        prediction, mask1, mask2, demand = self.decoder(x[:,-self.T-1:-1], h)

        if self.interpret:
            return prediction, mask1, mask2, demand
        return prediction   