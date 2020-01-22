
from model import Generator, Critic
import torch
from torch import optim, autograd
from torch.autograd import Variable
import pandas as pd
import torchvision
import argparse
import os


def one_hot(test, i=10):
    b = torch.zeros(test.shape[0], i)
    b[torch.arange(test.shape[0]), test] = 1
    return b.float()

def gradient_penalty(c, images, samples, y, lamb=10):
    assert images.size() == samples.size()
    jump = torch.rand(images.shape[0], 1).cuda()
    jump_ = jump.expand(images.shape[0], images.nelement()//images.shape[0]).contiguous().view(images.shape[0],1,28,28)
    interpolated = Variable(images*jump_ + (1-jump_)*samples, requires_grad = True)
    c_ = c(interpolated, y)
    gradients = autograd.grad(c_, interpolated, grad_outputs=(torch.ones(c_.size()).cuda()),create_graph = True, retain_graph = True)[0]
    return lamb*((1-gradients.norm(2,dim=1))**2).mean()

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, dest='lr', default=1e-4)
    parser.add_argument('--wd', type=bool, dest='wd', default = True, help='linearly interpolate between zero and the learning rate')
    parser.add_argument('--num_epochs', type=int, dest='num_epochs', default=201)
    parser.add_argument('--latent_dim', type=int, dest='latent_dim', default=118)
    parser.add_argument('--ratio', type=int, dest='ratio', default=5)
    parser.add_argument('--batch', type=int, dest='BATCH', default=200)
    parser.add_argument('--cp', type=int, default=0)
    arg = parser.parse_args()
    lr, weight_decay, num_epochs, latent_dim, ratio, BATCH = arg.lr, arg.wd, arg.num_epochs, arg.latent_dim, arg.ratio, arg.BATCH
    cp = arg.cp
    # Hyper Parameters
    betas = (0.5, 0.999)
    point = int(num_epochs // 10)
    check_epochs = [point*i for i in range(10)]
    check_epochs.append(num_epochs-1)


    download_path = '../../data'  # Set the path for the MNIST dataset
    num_gpus = 1 # Even if you do not have one
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
    trainset = torchvision.datasets.MNIST(root=download_path, train=True, download=True, transform=transform)
    testset = torchvision.datasets.MNIST(root=download_path, train=False, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH*num_gpus, shuffle=True, num_workers=64)
    testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH*num_gpus, shuffle=False, num_workers=64)

    G = Generator()
    C = Critic()
    if torch.cuda.is_available():
        G = G.cuda()
        C = C.cuda()

    if torch.cuda.device_count() > 1:
        G = torch.nn.DataParallel(G)
        C = torch.nn.DataParallel(C)

    if cp > 0:
        load_G = 'model_parameters/G' + str(cp) + '.pth'
        load_C = 'model_parameters/C' + str(cp) + '.pth'
        state_dict_G = torch.load(load_G)
        state_dict_C = torch.load(load_C)
        G.load_state_dict(state_dict_G["G_state_dict:"])
        C.load_state_dict(state_dict_C["C_state_dict:"])

    iter = 0
    test_iter = 0
    if not os.path.isdir('loss_summary'):
        os.makedirs('loss_summary')
    train_losses = pd.DataFrame(columns=['epoch','WD','GP','C_loss','G_loss'])
    test_losses = pd.DataFrame(columns=['epoch','WD','G_loss'])

    if not os.path.isdir('model_parameters'):
        os.makedirs('model_parameters')
    if not os.path.isdir('sample_images'):
        os.makedirs('sample_images')


    for epoch in range(num_epochs):
        if weight_decay:
            lr_ = lr * (num_epochs - epoch) / num_epochs
        else:
            lr_ = lr
        G_op = optim.Adam(G.parameters(), lr=lr_, betas=betas)
        C_op = optim.Adam(C.parameters(), lr=lr_, betas=betas)

        for _, data in enumerate(trainloader):
            if epoch > 40:
                """
                Change the ratio to improve Generator
                """
                ratio = 1

            for _ in range(ratio):
                images, labels = data
                y = one_hot(labels)
                if torch.cuda.is_available():
                    images, y = images.cuda(), y.cuda()
                C_op.zero_grad()
                samples = G(y, BATCH)
                d_samples = C(samples.detach(), y.detach())
                d_images = C(images, y)
                w_d = torch.mean(d_images) - torch.mean(d_samples) # Discriminator wants to maximize the WD
                gp = gradient_penalty(C, images, samples, y)
                c_loss = -w_d + gp
                c_loss.backward()
                C_op.step()

            G_op.zero_grad()
            samples_ = G(y, BATCH)
            d_samples = C(samples_, y)
            g_loss = -torch.mean(d_samples)
            g_loss.backward()
            G_op.step()
            iter += 1
            train_losses.loc[iter] = [epoch, w_d.item(), gp.item(), c_loss.item(), g_loss.item()]
            train_losses.to_csv('loss_summary/train_loss.csv')
            if iter % 150 == 0:
                idx = torch.randint(0, d_samples.shape[0], (16,))
                grid = torchvision.utils.make_grid(samples_.cpu()[idx])
                torchvision.utils.save_image(grid, 'sample_images/samples'+str(epoch)+'.png')
        train_group = train_losses.groupby(['epoch']).mean()
        train_group.to_csv('loss_summary/train_loss.csv')
        print("epoch={}, iteration={}, WD={}, D_loss={}, G_loss={}".format(epoch, iter, w_d.item(), c_loss.item(),
                                                                           g_loss.item()))

        if epoch in check_epochs:
            with torch.no_grad():
                for data in testloader:
                    images, labels = data
                    y = one_hot(labels)
                    images, y = images.cuda(), y.cuda()
                    G.eval()
                    C.eval()
                    outputs = G(y, BATCH)
                    d_samples_ = C(samples, y)
                    d_images_ = C(images, y)
                    w_d_ = torch.mean(d_images_) - torch.mean(d_samples_)
                    g_loss_ = -torch.mean(d_samples_)
                    test_losses.loc[test_iter] = [epoch, w_d_.item(), g_loss_.item()]
                    test_iter += 1
                    test_losses.to_csv('loss_summary/test_loss.csv')
        if epoch % 10 == 0:
            torch.save({'epoch:': epoch, 'G_state_dict:': G.state_dict()}, 'model_parameters/G' + str(epoch) + '.pth')
            torch.save({"epoch:": epoch, 'C_state_dict:': C.state_dict()}, 'model_parameters/C' + str(epoch) + '.pth')










