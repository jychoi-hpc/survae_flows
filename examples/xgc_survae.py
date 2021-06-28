import torch
import argparse

# Data
from survae.data.loaders.image import DynamicallyBinarizedMNIST

# Model
from survae.flows import Flow
from survae.transforms import VAE
from survae.distributions import StandardNormal, ConditionalNormal, ConditionalBernoulli
from survae.nn.nets import MLP

# Optim
from torch.optim import Adam
from survae.utils import iwbo_nats

# Plot
import torchvision.utils as vutils

# VAPOR
import sys
import os
import numpy as np
import adios2 as ad2
import logging
from sklearn.model_selection import train_test_split

# %%
def read_f0(istep, expdir=None, iphi=None, inode=0, nnodes=None, average=False, randomread=0.0, nchunk=16, fieldline=False):
    """
    Read XGC f0 data
    """
    def adios2_get_shape(f, varname):
        nstep = int(f.available_variables()[varname]['AvailableStepsCount'])
        shape = f.available_variables()[varname]['Shape']
        lshape = None
        if shape == '':
            ## Accessing Adios1 file
            ## Read data and figure out
            v = f.read(varname)
            lshape = v.shape
        else:
            lshape = tuple([ int(x.strip(',')) for x in shape.strip().split() ])
        return (nstep, lshape)

    fname = os.path.join(expdir, 'restart_dir/xgc.f0.%05d.bp'%istep)
    if randomread > 0.0:
        ## prefetch to get metadata
        with ad2.open(fname, 'r') as f:
            nstep, nsize = adios2_get_shape(f, 'i_f')
            ndim = len(nsize)
            nphi = nsize[0]
            _nnodes = nsize[2] if nnodes is None else nnodes
            nmu = nsize[1]
            nvp = nsize[3]
        assert _nnodes%nchunk == 0
        _lnodes = list(range(inode, inode+_nnodes, nchunk))
        lnodes = random.sample(_lnodes, k=int(len(_lnodes)*randomread))
        lnodes = np.sort(lnodes)

        lf = list()
        li = list()
        for i in tqdm(lnodes):
            li.append(np.array(range(i,i+nchunk), dtype=np.int32))
            with ad2.open(fname, 'r') as f:
                nphi = nsize[0] if iphi is None else 1
                iphi = 0 if iphi is None else iphi
                start = (iphi,0,i,0)
                count = (nphi,nmu,nchunk,nvp)
                _f = f.read('i_f', start=start, count=count).astype('float64')
                lf.append(_f)
        i_f = np.concatenate(lf, axis=2)
        lb = np.concatenate(li)
    elif fieldline is True:
        import networkx as nx

        fname2 = os.path.join(expdir, 'xgc.mesh.bp')
        with ad2.open(fname2, 'r') as f:
            _nnodes = int(f.read('n_n', ))
            nextnode = f.read('nextnode')
        
        G = nx.Graph()
        for i in range(_nnodes):
            G.add_node(i)
        for i in range(_nnodes):
            G.add_edge(i, nextnode[i])
            G.add_edge(nextnode[i], i)
        cc = [x for x in list(nx.connected_components(G)) if len(x) >= 16]

        li = list()
        for k, components in enumerate(cc):
            DG = nx.DiGraph()
            for i in components:
                DG.add_node(i)
            for i in components:
                DG.add_edge(i, nextnode[i])
            
            cycle = list(nx.find_cycle(DG))
            DG.remove_edge(*cycle[-1])
            
            path = nx.dag_longest_path(DG)
            #print (k, len(components), path[0])
            for i in path[:len(path)-len(path)%16]:
                li.append(i)

        with ad2.open(fname, 'r') as f:
            nstep, nsize = adios2_get_shape(f, 'i_f')
            ndim = len(nsize)
            nphi = nsize[0] if iphi is None else 1
            iphi = 0 if iphi is None else iphi
            _nnodes = nsize[2]
            nmu = nsize[1]
            nvp = nsize[3]
            start = (iphi,0,0,0)
            count = (nphi,nmu,_nnodes,nvp)
            logging.info (f"Reading: {start} {count}")
            i_f = f.read('i_f', start=start, count=count).astype('float64')
        
        _nnodes = len(li)-inode if nnodes is None else nnodes
        lb = np.array(li[inode:inode+_nnodes], dtype=np.int32)
        logging.info (f"Fieldline: {len(lb)}")
        logging.info (f"{lb}")
        i_f = i_f[:,:,lb,:]
    else:
        with ad2.open(fname, 'r') as f:
            nstep, nsize = adios2_get_shape(f, 'i_f')
            ndim = len(nsize)
            nphi = nsize[0] if iphi is None else 1
            iphi = 0 if iphi is None else iphi
            _nnodes = nsize[2]-inode if nnodes is None else nnodes
            nmu = nsize[1]
            nvp = nsize[3]
            start = (iphi,0,inode,0)
            count = (nphi,nmu,_nnodes,nvp)
            logging.info (f"Reading: {start} {count}")
            i_f = f.read('i_f', start=start, count=count).astype('float64')
            #e_f = f.read('e_f')
        li = list(range(inode, inode+_nnodes))
        lb = np.array(li, dtype=np.int32)

    # if i_f.shape[3] == 31:
    #     i_f = np.append(i_f, i_f[...,30:31], axis=3)
    #     # e_f = np.append(e_f, e_f[...,30:31], axis=3)
    # if i_f.shape[3] == 39:
    #     i_f = np.append(i_f, i_f[...,38:39], axis=3)
    #     i_f = np.append(i_f, i_f[:,38:39,:,:], axis=1)

    Z0 = np.moveaxis(i_f, 1, 2)

    if average:
        Z0 = np.mean(Z0, axis=0)
        zlb = lb
    else:
        Z0 = Z0.reshape((-1,Z0.shape[2],Z0.shape[3]))
        _lb = list()
        for i in range(nphi):
            _lb.append( i*100_000_000 + lb)
        zlb = np.concatenate(_lb)
    
    #zlb = np.concatenate(li)
    zmu = np.mean(Z0, axis=(1,2))
    zsig = np.std(Z0, axis=(1,2))
    zmin = np.min(Z0, axis=(1,2))
    zmax = np.max(Z0, axis=(1,2))
    Zif = (Z0 - zmin[:,np.newaxis,np.newaxis])/(zmax-zmin)[:,np.newaxis,np.newaxis]

    return (Z0, Zif, zmu, zsig, zmin, zmax, zlb)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--num_epochs', help='num of epochs (default: %(default)s)', type=int, default=20)
    parser.add_argument('-b', '--batch_size', help='batch_size (default: %(default)s)', type=int, default=128)
    args = parser.parse_args()

    logging.info("Command: {0}\n".format(" ".join([x for x in sys.argv]))) 
    logging.debug("All settings used:") 
    for k,v in sorted(vars(args).items()): 
        logging.debug("\t{0}: {1}".format(k,v))

    ############
    ## Device ##
    ############

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    ##########
    ## Data ##
    ##########

    # data = DynamicallyBinarizedMNIST()
    # train_loader, test_loader = data.get_data_loaders(128)

    batch_size = args.batch_size
    Z0, Zif, zmu, zsig, zmin, zmax, zlb = read_f0(420, expdir='d3d_coarse_v2_colab', iphi=0)
    _, nx, ny = Z0.shape

    lx = list()
    ly = list()
    for i in range(len(zlb)):
        lx.append(Zif[i,np.newaxis,:,:])
        ly.append(Zif[i,np.newaxis,:,:])
    
    X_train, X_test, y_train, y_test = train_test_split(lx, ly, test_size=0.2)
    print (lx[0].shape, ly[0].shape, len(X_train), len(X_test))

    # X_train, y_train = rescale(X_train, grid), torch.tensor(y_train)
    # X_test, y_test = rescale(X_test, grid), torch.tensor(y_test)
    # X_full, y_full = rescale(lx, grid), torch.tensor(ly)
    X_train, y_train = torch.tensor(X_train), torch.tensor(y_train)
    X_test, y_test = torch.tensor(X_test), torch.tensor(y_test)
    X_full, y_full = torch.tensor(lx), torch.tensor(ly)

    training_data = torch.utils.data.TensorDataset(X_train, y_train)
    validation_data = torch.utils.data.TensorDataset(X_test, y_test)
    full_data = torch.utils.data.TensorDataset(X_full, y_full)

    train_loader = torch.utils.data.DataLoader(training_data, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(validation_data, batch_size=batch_size, shuffle=True)
    full_loader = torch.utils.data.DataLoader(full_data, batch_size=1, shuffle=False)

    ###########
    ## Model ##
    ###########

    latent_size = 20

    encoder = ConditionalNormal(MLP(nx*ny, 2*latent_size,
                                    hidden_units=[512,256],
                                    activation='relu',
                                    in_lambda=lambda x: 2 * x.view(x.shape[0], nx*ny).float() - 1))
    decoder = ConditionalBernoulli(MLP(latent_size, nx*ny,
                                    hidden_units=[512,256],
                                    activation='relu',
                                    out_lambda=lambda x: x.view(x.shape[0], 1, nx, ny)))

    model = Flow(base_dist=StandardNormal((latent_size,)),
                transforms=[
                    VAE(encoder=encoder, decoder=decoder)
                ]).to(device)

    ###########
    ## Optim ##
    ###########

    optimizer = Adam(model.parameters(), lr=1e-3)

    ###########
    ## Train ##
    ###########

    print('Training...')
    num_epchs = args.num_epochs
    for epoch in range(num_epchs):
        l = 0.0
        for i, (x, y) in enumerate(train_loader):
            optimizer.zero_grad()
            loss = -model.log_prob(x.to(device)).mean()
            loss.backward()
            optimizer.step()
            l += loss.detach().cpu().item()
            print('Epoch: {}/{}, Iter: {}/{}, Nats: {:.3f}'.format(epoch+1, num_epchs, i+1, len(train_loader), l/(i+1)), end='\r')
        print('')

    ##########
    ## Test ##
    ##########

    print('Testing...')
    with torch.no_grad():
        l = 0.0
        for i, (x, y) in enumerate(test_loader):
            loss = iwbo_nats(model, x.to(device), k=10)
            #loss = -model.log_prob(x.to(device)).mean()
            l += loss.detach().cpu().item()
            print('Iter: {}/{}, Nats: {:.3f}'.format(i+1, len(test_loader), l/(i+1)), end='\r')
        print('')

    ############
    ## Sample ##
    ############

    print('Sampling...')
    img, _ = next(iter(test_loader))
    img = img[:64]
    samples = model.sample(64)
    
    vutils.save_image(img.cpu().float(), fp='mnist_data.png', nrow=8)
    vutils.save_image(samples.cpu().float(), fp='mnist_vae.png', nrow=8)
