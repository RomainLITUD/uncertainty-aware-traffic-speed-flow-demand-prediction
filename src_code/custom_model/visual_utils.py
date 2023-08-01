import numpy as np
import matplotlib.pyplot as plt
from custom_model.custom_dataset import *
from shapely.geometry import Point, LineString
from matplotlib import cm
from skimage.transform import resize
from labellines import labelLine, labelLines

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def get_prediction(model, testset, index):

    groundtruth = testset.X[index]

    x = torch.Tensor(testset.X[[index]]).float().to(device)
    y, m1, m2, demand = model(x)
    y = y.to("cpu").detach().numpy()


    v = y[...,1]
    alpha = y[...,2]
    beta = y[...,3]

    alea = beta/(alpha-1)*130*130
    epis = beta/v/(alpha-1)*130*130

    speed = y[...,4]*130
    flow = y[...,5]*3000

    states = speed.copy()
    states[states>55] = 80
    states[states<=55] = -19

    m1 = m1.to("cpu").detach().numpy()
    m2 = m2.to("cpu").detach().numpy()
    demand = demand.to("cpu").detach().numpy()*3000/speed

    return groundtruth, speed, flow, alea, epis, m1, m2, demand

def plot_prediction(yt, yp, mode='speed'):
    if mode == 'speed':
        y = yt[...,0]*130
        vm = 130
    if mode == 'flow':
        y = yt[...,1]*3000
        vm = 3000
    yr = resize(y, (70, 193), anti_aliasing=True)
    y_pred = np.concatenate([resize(y[:20], (40, 193), anti_aliasing=True), resize(yp, (30, 193), anti_aliasing=True)], 0)

    fig, ax = plt.subplots(1, 2, constrained_layout=True, figsize=(8, 3))

    X = np.arange(0.5,194, 1)
    Y = np.arange(-40.5,30.5, 1)
    im1 = ax[0].pcolormesh(Y, X, yr.T, cmap='rainbow_r', vmin=0, vmax=vm)
    ax[0].vlines(0, ymin=0.5, ymax=193.5, color='white', lw=2)
    im2 = ax[1].pcolormesh(Y, X, y_pred.T, cmap='rainbow_r', vmin=0, vmax=vm)
    ax[1].vlines(0, ymin=0.5, ymax=193.5, color='white', lw=2)

    ax[0].set_xlabel('time (min)')
    ax[0].set_ylabel('location (nb.)')
    ax[1].set_xlabel('time (min)')

    ax[0].set_title('groundtruth', fontsize=14)
    ax[1].set_title('prediction', fontsize=14)

    if mode == 'speed':
        fig.colorbar(im2, orientation='vertical', label='speed (km/h)', fraction=0.046, pad=0.04)
    if mode == 'flow':
        fig.colorbar(im2, orientation='vertical', label='flow (veh/lane/h)', fraction=0.046, pad=0.04)
    return fig

def plot_state_on_map(links, state, mode='speed'):

    if mode == 'speed':
        vm = 130
    if mode == 'flow':
        vm = 3000
    fig, ax = plt.subplots(constrained_layout=True, figsize=(5,20/6))
    for i in range(len(links)):
        value = links[str(i)]
        if mode == 'speed':
            ax.plot(value[1], value[0], c=cm.rainbow_r(state[i]/vm), lw=2)
            #im = ax.scatter(value[1], value[0], c=state[i]*np.ones(len(value[1])), s=0., vmin=0, vmax=vm, cmap=cm.rainbow_r)
        if mode == 'flow':
            ax.plot(value[1], value[0], c=cm.rainbow(state[i]/vm), lw=2)
            #im = ax.scatter(value[1], value[0], c=state[i]*np.ones(len(value[1])), s=0., vmin=0, vmax=vm, cmap=cm.rainbow)
    if mode == 'speed':
        im = ax.scatter(value[1], value[0], c=65*np.ones(len(value[1])), s=0., vmin=0, vmax=vm, cmap=cm.rainbow_r)
    if mode == 'flow':
        im = ax.scatter(value[1], value[0], c=65*np.ones(len(value[1])), s=0., vmin=0, vmax=vm, cmap=cm.rainbow)
    #ax.set_xlabel('longitude')
    #ax.set_ylabel('latitude')
    ax.set_xticks([])
    ax.set_yticks([])
    if mode == 'speed':
        fig.colorbar(im, orientation='vertical', label='speed (km/h)', fraction=0.046, pad=0.04)
    if mode == 'flow':
        fig.colorbar(im, orientation='vertical', label='flow (veh/lane/h)', fraction=0.046, pad=0.04)
    #ax.set_aspect('equal')
    return fig

def plot_demand_on_map(links, state, vm):

    fig, ax = plt.subplots(constrained_layout=True, figsize=(5,20/6))
    for i in range(len(links)):
        value = links[str(i)]
        ax.plot(value[1], value[0], c=cm.bwr((state[i]+vm)/vm/2), lw=2)
        im = ax.scatter(value[1], value[0], c=state[i]*np.ones(len(value[1])), s=0., vmin=-vm, vmax=vm, cmap=cm.bwr)
    
    ax.set_xlabel('longitude')
    ax.set_ylabel('latitude')
    ax.set_xticks([])
    ax.set_yticks([])
    fig.colorbar(im, orientation='vertical', label='fluctuation (veh/lane)', fraction=0.046, pad=0.04)
    #ax.set_aspect('equal')
    return fig

def draw_interpretation(links, A, Attention, id, mode='flow', xlim=[], ylim=[], clear_axis=False, colorbar=False, show_lines=True):
    surrounding_id = np.where(A[id]>0)[0]
    atten = Attention[id][surrounding_id]
    vm = np.amax(atten)

    fig, ax = plt.subplots(constrained_layout=True, figsize=(5, 5))
    #im = ax.scatter([])
    for id_, at in zip(surrounding_id, atten):
        value = links[str(id_)]
        if id_ == id:
            ax.plot(value[1], value[0], c=cm.rainbow(at/vm), lw=5, label=format(at, '.2f'))
            ax.arrow(value[1][-3], value[0][-3], value[1][-1]-value[1][-3], value[0][-1]-value[0][-3], color=cm.rainbow(at/vm), 
                     width=0, head_width=9e-4)
            im = ax.scatter(value[1], value[0], c=at*np.ones(len(value[1])), s=0., vmin=0, vmax=vm, cmap=cm.rainbow)
        else:
            if mode=='speed':
                ax.plot(value[1], value[0], c=cm.rainbow(at/vm), lw=2, label=format(at, '.2f'))
                ax.arrow(value[1][-3], value[0][-3], value[1][-1]-value[1][-3], value[0][-1]-value[0][-3], color=cm.rainbow(at/vm), 
                     width=0, head_width=9e-4)
            if mode=='flow':
                ax.plot(value[1], value[0], c=cm.rainbow(at/vm), lw=2, label=format(at, '.1f'))
                ax.arrow(value[1][-3], value[0][-3], value[1][-1]-value[1][-3], value[0][-1]-value[0][-3], color=cm.rainbow(at/vm), 
                     width=0, head_width=9e-4)
    if mode=='speed':
        labelLines(ax.get_lines(), zorder=2.5, fontsize=10)
    if mode=='flow' and show_lines:
        labelLines(ax.get_lines(), zorder=2.5, fontsize=10)
    if clear_axis:
        ax.set_xticks([])
        ax.set_yticks([])
    if xlim:
        ax.set_xlim(xlim[0], xlim[1])
    if ylim:
        ax.set_ylim(ylim[0], ylim[1])
    # ax.set_xlabel('longitude')
    # ax.set_ylabel('latitude')
    if colorbar:
        fig.colorbar(im, orientation='vertical', label='impact coefficience', fraction=0.046, pad=0.04)
    ax.set_aspect('equal')
    return fig
