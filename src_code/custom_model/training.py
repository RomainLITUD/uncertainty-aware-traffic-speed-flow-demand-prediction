import torch
import math
from custom_model.custom_dataset import *
import gc
from tqdm import tqdm

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def train_one_epoch(model, optimizer, loss_2, training_loader, scheduler, beta, count, T):
    running_loss = 0.
    last_loss = 0.
    
    for j, data in enumerate(training_loader):
        x = data
        optimizer.zero_grad()
        p = model(x, math.exp(beta*count))
        #p = model(x, 0)
        loss = loss_2(p, x[:,-T:])
        print('loss:'+str(loss.item()),end='\r')
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if j % 400 == 399:
            scheduler.step()
            count += 1
            last_loss = running_loss / 400
            print('  batch {} loss: {}'.format(j + 1, last_loss))
            running_loss = 0.
                
    return last_loss, count
        
def train_model(epochs, batch_size, trainset, model, optimizer, validation_loader, loss_2, scheduler, T, beta=-0.03):
    training_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    epoch_number = 0

    EPOCHS = epochs
    count = 0

    for epoch in range(EPOCHS):
        print('EPOCH {}:'.format(epoch_number + 1))

        model.train(True)
        avg_loss, count = train_one_epoch(model, optimizer, loss_2, training_loader, scheduler, beta, count, T)
        model.train(False)
        if epoch % 5 == 4:
            torch.save(model.state_dict(), './distill/high70.pt')

        running_vloss = 0.0
        for i, vdata in enumerate(validation_loader):
            x = vdata
            p = model(x, 0)
            heatmaploss = loss_2(p, x[:,-T:])
            running_vloss += float(heatmaploss)

        avg_vloss = running_vloss / (i + 1)
        print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))
        scheduler.step()
        epoch_number += 1

def test_run_point(testset, model):
    model.eval()
    testloader = DataLoader(testset, batch_size=16, shuffle=False)

    prediction = []

    for j, data in tqdm(enumerate(testloader)):
        x = data
        p = model(x, 0)

        gc.disable()
        prediction.append(np.array(p.detach().to('cpu'))[...,-3:])
        gc.enable()

    prediction = np.concatenate(prediction, 0)

    return prediction

def test_run_point_distill(testset, model):
    model.eval()
    testloader = DataLoader(testset, batch_size=16, shuffle=False)

    prediction = []

    for j, data in tqdm(enumerate(testloader)):
        x = data
        p = model(x, 0)

        gc.disable()
        prediction.append(np.array(p.detach().to('cpu')))
        gc.enable()

    prediction = np.concatenate(prediction, 0)

    return prediction

def test_run_rarity(testset, model):
    model.eval()
    testloader = DataLoader(testset, batch_size=16, shuffle=False)

    Um = []

    for j, data in tqdm(enumerate(testloader)):
        x = data
        p = model(x, 0)

        v = p[...,1]
        a = p[...,2]
        b = p[...,3]

        alea = b/(a-1)
        epis = alea/v

        gc.disable()
        pred = np.array(epis.detach().to('cpu'))
        Um.append(np.mean(pred[:,-1:], (1,2)))
        gc.enable()

    Um = np.concatenate(Um)

    return Um

def test_run_uq(testset, model):
    model.eval()
    testloader = DataLoader(testset, batch_size=16, shuffle=False)

    Ud = []
    Um = []

    for j, data in tqdm(enumerate(testloader)):
        x = data
        p = model(x, 0)

        v = p[...,1]
        a = p[...,2]
        b = p[...,3]

        alea = b/(a-1)
        epis = alea/v

        gc.disable()
        Ud.append(np.array(alea.detach().to('cpu')))
        Um.append(np.array(epis.detach().to('cpu')))
        gc.enable()

    Ud = np.concatenate(Ud, 0)
    Um = np.concatenate(Um, 0)

    return Ud, Um

def get_epis(model, year):

    model.eval()
    dt = zarr.open('./datasets/'+year+'.zarr')
    V_morning = np.transpose(dt.speed_morning, (0,2,1))
    V_evening = np.transpose(dt.speed_evening, (0,2,1))
    Q_morning = np.transpose(dt.flow_morning, (0,2,1))
    Q_evening = np.transpose(dt.flow_evening, (0,2,1))
    
    V_morning[V_morning>130] = 100.
    V_evening[V_evening>130] = 100.

    V_morning = V_morning/130.
    V_evening = V_evening/130.

    Q_morning[Q_morning>3000] = 1000.
    Q_evening[Q_evening>3000] = 1000.

    Q_morning = Q_morning/3000.
    Q_evening = Q_evening/3000.

    Umorning = np.zeros((len(V_morning), 85, 193))
    Uevening = np.zeros((len(V_evening), 175, 193))

    print('evaluating morning:')

    for d in tqdm(range(0, len(V_morning), 10)):        
        for i in range(0, 85):
            x = np.stack([V_morning[d:d+10,i:i+35], Q_morning[d:d+10,i:i+35]], -1)
            x = torch.Tensor(x).float().to(device)

            y = model(x, 0)

            v = y[...,1]
            a = y[...,2]
            b = y[...,3]

            epis = torch.mean(b/(a-1), -2).detach().to('cpu').numpy()**0.5*130

            Umorning[d:d+10, i] = epis

    print('evaluating evening:')        

    for d in tqdm(range(0, len(V_evening), 10)):        
        for i in range(0, 175):
            x = np.stack([V_evening[d:d+10,i:i+35], Q_evening[d:d+10,i:i+35]], -1)
            x = torch.Tensor(x).float().to(device)

            y = model(x, 0)

            v = y[...,1]
            a = y[...,2]
            b = y[...,3]

            epis = torch.mean(b/(a-1), -2).detach().to('cpu').numpy()**0.5*130

            Uevening[d:d+10, i] = epis

    return Umorning, Uevening