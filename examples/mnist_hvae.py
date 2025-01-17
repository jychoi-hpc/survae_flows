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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--num_epochs', help='num of epochs (default: %(default)s)', type=int, default=20)
    args = parser.parse_args()

    ############
    ## Device ##
    ############

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    ##########
    ## Data ##
    ##########

    data = DynamicallyBinarizedMNIST()
    train_loader, test_loader = data.get_data_loaders(128)

    ###########
    ## Model ##
    ###########

    latent_sizes = [20,10]

    encoder = ConditionalNormal(MLP(784, 2*latent_sizes[0],
                                    hidden_units=[512,256],
                                    activation='relu',
                                    in_lambda=lambda x: 2 * x.view(x.shape[0], 784).float() - 1))
    decoder = ConditionalBernoulli(MLP(latent_sizes[0], 784,
                                    hidden_units=[512,256],
                                    activation='relu',
                                    out_lambda=lambda x: x.view(x.shape[0], 1, 28, 28)))

    encoder2 = ConditionalNormal(MLP(latent_sizes[0], 2*latent_sizes[1],
                                    hidden_units=[256,128],
                                    activation='relu'))
    decoder2 = ConditionalNormal(MLP(latent_sizes[1], 2*latent_sizes[0],
                                    hidden_units=[256,128],
                                    activation='relu'))

    model = Flow(base_dist=StandardNormal((latent_sizes[-1],)),
                transforms=[
                    VAE(encoder=encoder, decoder=decoder),
                    VAE(encoder=encoder2, decoder=decoder2),
                ]).to(device)

    ###########
    ## Optim ##
    ###########

    optimizer = Adam(model.parameters(), lr=1e-3)

    ###########
    ## Train ##
    ###########

    print('Training...')
    for epoch in range(args.num_epochs):
        l = 0.0
        for i, x in enumerate(train_loader):
            optimizer.zero_grad()
            loss = -model.log_prob(x.to(device)).mean()
            loss.backward()
            optimizer.step()
            l += loss.detach().cpu().item()
            print('Epoch: {}/{}, Iter: {}/{}, Nats: {:.3f}'.format(epoch+1, args.num_epochs, i+1, len(train_loader), l/(i+1)), end='\r')
        print('')

    ##########
    ## Test ##
    ##########

    print('Testing...')
    with torch.no_grad():
        l = 0.0
        for i, x in enumerate(test_loader):
            loss = iwbo_nats(model, x.to(device), k=10)
            l += loss.detach().cpu().item()
            print('Iter: {}/{}, Nats: {:.3f}'.format(i+1, len(test_loader), l/(i+1)), end='\r')
        print('')

    ############
    ## Sample ##
    ############

    print('Sampling...')
    img = next(iter(test_loader))[:64]
    samples = model.sample(64)

    vutils.save_image(img.cpu().float(), fp='mnist_data.png', nrow=8)
    vutils.save_image(samples.cpu().float(), fp='mnist_hvae.png', nrow=8)
