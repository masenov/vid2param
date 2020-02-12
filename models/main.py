from model import *
from tools import *
from viz import *


# set random seed to 0
#np.random.seed(2)
#torch.manual_seed(0)
device = torch.device("cuda:0")
# load data and make training set
data, params = generate_data(T = 20, L = 2000, N = 100, sample=2000)
data = normalize_data(data)
viz_data(data[0,:,0],data[0,:,1])
input = torch.from_numpy(data[3:, :-1, :]).to(device)
target = torch.from_numpy(data[3:, 1:, :]).to(device)
test_input = torch.from_numpy(data[:6, :-1, :]).to(device)
test_input = torch.from_numpy(data[:6, :-1500, :]).to(device)
test_target = torch.from_numpy(data[:6, 1:, :]).to(device)
# build the model
seq = Sequence()
seq.double()
seq.to(device)
criterion = nn.MSELoss()
# use LBFGS as optimizer since we can load the whole data to train
optimizer = optim.LBFGS(seq.parameters(), lr=0.8)
#optimizer = optim.Adam(seq.parameters(), lr=1e-2)
#begin to train
for i in range(500):
    def closure():
        optimizer.zero_grad()
        out = seq(input, future=0)
        out = out.view(out.shape[0],out.shape[1],4)
        loss = criterion(out, target[:,:,:4])
        print('step: ',i,' loss:', loss.item())
        loss.backward()
        return loss
    optimizer.step(closure)
    if (i%1==0):
        # begin to predict, no need to track gradient here
        with torch.no_grad():
            future = 4000
            pred = seq(test_input, future=future)
            pred = pred.view(pred.shape[0],pred.shape[1],4)
            #loss = criterion(pred[:, :-future,:], test_target[:,:,:1])
            #print('test loss:', loss.item())
            y = pred.cpu().detach().numpy()
        viz_results(y, future, i)
