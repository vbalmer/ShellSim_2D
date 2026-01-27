import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel, Matern, ExpSineSquared


De = np.load("NN/newadd_data_De.pkl", allow_pickle=True)
eps = np.load("NN/newadd_data_eps.pkl", allow_pickle=True)
sig = np.load("NN/newadd_data_sig.pkl", allow_pickle=True)
t = np.load("NN/newadd_data_t.pkl", allow_pickle=True)

eps = np.array(eps)
sig = np.array(sig)#[:,3:6]
t = np.array(t)[:,0]

X = np.concatenate([eps, t.reshape(-1,1)], axis=-1)

### equally sample from N bins
ratio = 0.01
bins = 100
newX = []
newSig = []
hists, bin_edges_list = [], []
for i in range(sig.shape[1]):
    hist, bin_edges = np.histogram(sig[:,i], bins=bins)
    hists.append(hist)
    bin_edges_list.append(bin_edges)
hists = np.array(hists)
bin_edges_list = np.array(bin_edges_list)

sampleWeights = np.zeros(sig.shape[0])
for i in range(sig.shape[0]):
    for j in range(sig.shape[1]):
        freq = 0
        for k in range(bins):
            condition = (sig[i,j] >= bin_edges_list[j][k]) and (sig[i,j] <= bin_edges_list[j][k+1])
            if condition:
                freq = hists[j][k]
                break
        if freq == 0:
            assert False
        sampleWeights[i] += 1 / freq
sampleWeights /= sampleWeights.sum()

randomIdx = np.random.choice(eps.shape[0], int(ratio*eps.shape[0]), replace=False, p=sampleWeights)
# X = X[randomIdx]
# sig = sig[randomIdx]
print(f"X shape: {X.shape} - sig shape: {sig.shape}")

# histogram of sig
fig, axs = plt.subplots(sig.shape[1], 1, figsize=(2*sig.shape[1], 12))
for i in range(sig.shape[1]):
    axs[i].hist(sig[:,i], bins=30)
    axs[i].set_title(f"sig{i}")
plt.savefig("hist_sig.png")

normalizedX = (X - np.min(X, axis=0)) / (np.max(X, axis=0) - np.min(X, axis=0))
normalizedSig = (sig - np.min(sig, axis=0)) / (np.max(sig, axis=0) - np.min(sig, axis=0))

GP = False
if GP:
    kernel = DotProduct(sigma_0_bounds=(1e-9, 1e3)) + WhiteKernel(noise_level_bounds=(1e-9, 1e3)) + Matern(length_scale_bounds=(1e-9, 1e6), nu=1.5) + ExpSineSquared(length_scale_bounds=(1e-9, 1e6), periodicity_bounds=(1e-9, 1e9))
    model = GaussianProcessRegressor(kernel=kernel, random_state=0, n_restarts_optimizer=2)

    model.fit(normalizedX, normalizedSig)

    error = abs(model.predict(normalizedX) - normalizedSig)

    print(f"Rel Error Mean: {100*error.mean():.4f}%  - Error Std: {100*error.std():.4f}% - Error Max: {100*error.max():.4f}%")
    print(f"Rel Error Mean in each component: {100*error.mean(axis=0)}%")


    predictions = model.predict(normalizedX) * (np.max(sig, axis=0) - np.min(sig, axis=0)) + np.min(sig, axis=0)
    rmse = np.sqrt(np.mean((predictions - sig)**2))
    print(f"RMSE total: {rmse:.4f}")
    print(f"RMSE in each component: {np.sqrt(np.mean((predictions - sig)**2, axis=0))}")

    coolrmse = np.sqrt(np.mean((predictions - sig)**2, axis=0)) / np.sqrt(np.mean((sig - sig.mean(axis=0))**2, axis=0))
    print(f"Cool RMSE: {coolrmse}")

    def s (eps, t):
        nu = 0.3
        E = 210000

        D_p = (E/(1+nu**2))*np.array([[1, nu, 0], 
                                    [nu, 1, 0], 
                                    [0, 0, 0.5 * (1 - nu)]])

        Dse = (5/6)*t*(2*E)/(4*(1+nu))

        D_an_1 = np.hstack([t*D_p, 0*D_p, np.zeros((3,2))])
        D_an_2 = np.hstack([0*D_p, (1/12)*(t**3)*D_p, np.zeros((3,2))])
        D_an_3 = np.hstack([np.zeros((2,3)), np.zeros((2,3)), np.array([[Dse, 0], [0, Dse]])])
        D_analytical = np.vstack([D_an_1, D_an_2, D_an_3])


        sig_analytical = np.matmul(D_analytical, eps)
        if np.linalg.matrix_rank(D_analytical) != 8:
            print(f"Matrix Rank: {np.linalg.matrix_rank(D_analytical)}")

        return sig_analytical


    siga = []
    for i in range(eps.shape[0]):
        siga.append(s(eps[i], t[i]))
    siga = np.array(siga)

    # plotting
    nRows = sig.shape[1]
    nCols = eps.shape[1]
    fig, axs = plt.subplots(nRows, nCols, figsize=(2*nCols, 2*nRows))
    for i in range(nRows):
        for j in range(nCols):
            axs[i, j].plot(X[:,j], sig[:,i], 'o')
            axs[i, j].set_xlabel(f"eps{j}")
            axs[i, j].set_ylabel(f"sig{i}")
            axs[i, j].plot(X[:,j], predictions[:,i], 'ro')

    plt.savefig("testo.png")
    plt.close()



### Fitting an MLP
hiddenDim = 512
activation = nn.ELU()
modules = [
    nn.Linear(X.shape[1], hiddenDim), 
    activation,
    nn.Dropout(p=0.0),
]
for _ in range(4):
    modules.append(nn.Linear(hiddenDim, hiddenDim))
    modules.append(activation)
    modules.append(nn.Dropout(p=0.0))
modules.append(nn.Linear(hiddenDim, sig.shape[1]))

model = nn.Sequential(*modules)
# number of parameters
print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
lossFn = nn.L1Loss()
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.99)

randomIdx = np.random.permutation(normalizedX.shape[0])
trainRatio, testRatio = 0.8, 0.2
trainIdx = randomIdx[:int(trainRatio*normalizedX.shape[0])]
testIdx = randomIdx[int(trainRatio*normalizedX.shape[0]):]

model.to(device)

MAXEPOCHS = 10000
for epoch in range(MAXEPOCHS):
    model.train()
    optimizer.zero_grad()
    X_torch = torch.tensor(normalizedX[trainIdx], dtype=torch.float32).to(device)
    sig_torch = torch.tensor(normalizedSig[trainIdx], dtype=torch.float32).to(device)

    pred = model(X_torch)
    loss = lossFn(pred, sig_torch)
    loss.backward()
    optimizer.step()
    scheduler.step()

    if epoch % 100 == 0:
        print(f"Epoch [{epoch}/{MAXEPOCHS}] with LR {scheduler.get_last_lr()[0]:.2e}: \tLoss: {loss.item():.4e}")

model.eval()
normalizedPred = model(torch.tensor(normalizedX[testIdx], dtype=torch.float32).to(device)).cpu().detach().numpy()
predictions = normalizedPred * (np.max(sig, axis=0) - np.min(sig, axis=0)) + np.min(sig, axis=0)

error = abs(normalizedPred - normalizedSig[testIdx])

print(f"Rel Error Mean: {100*error.mean():.4f}%  - Error Std: {100*error.std():.4f}% - Error Max: {100*error.max():.4f}%")
print(f"Rel Error Mean in each component: {100*error.mean(axis=0)}%")

rmse = np.sqrt(np.mean((predictions - sig[testIdx])**2))
print(f"RMSE total: {rmse:.4f}")
print(f"RMSE in each component: {np.sqrt(np.mean((predictions - sig[testIdx])**2, axis=0))}")

coolrmse = np.sqrt(np.mean((predictions - sig[testIdx])**2, axis=0)) / np.sqrt(np.mean((sig - sig.mean(axis=0))**2, axis=0))
print(f"Cool RMSE: {coolrmse}")


# plotting
nRows = sig.shape[1]
nCols = eps.shape[1]
fig, axs = plt.subplots(nRows, nCols, figsize=(2*nCols, 2*nRows))
for i in range(nRows):
    for j in range(nCols):
        axs[i, j].plot(X[testIdx,j], sig[testIdx,i], 'o')
        axs[i, j].set_xlabel(f"eps{j}")
        axs[i, j].set_ylabel(f"sig{i}")
        axs[i, j].plot(X[testIdx,j], predictions[:,i], 'ro')

plt.savefig("testo.png")
plt.close()