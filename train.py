import os
import torch
from torch import nn
from torch import optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.distributions import normal
from utils import Logger
import torchvision.transforms as transforms

from models import E1, E2, E3, Decoder, Disc, PatchDiscriminator, MsImageDis
from utils import save_imgs, save_model, load_model, save_stripped_imgs
from utils import CustomDataset

import argparse


def train(args):
    if not os.path.exists(args.out):
        os.makedirs(args.out)

    _iter = 0

    comp_transform = transforms.Compose([
        transforms.CenterCrop(args.crop),
        transforms.Resize(args.resize),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    domA_train = CustomDataset(os.path.join(args.root, 'trainA.txt'), transform=comp_transform)
    domB_train = CustomDataset(os.path.join(args.root, 'trainB.txt'), transform=comp_transform)

    A_label = torch.full((args.bs,), 1)
    B_label = torch.full((args.bs,), 0)

    e1 = E1(args.sep, args.resize // 64)
    e2 = E2(args.sep, args.resize // 64)
    e3 = E3(args.sep, args.resize // 64)
    decoder = Decoder(args.resize // 64)
    disc = Disc(args.sep, args.resize // 64)
    zero_encoding = torch.full((args.bs, args.sep * (args.resize // 64) * (
            args.resize // 64)), 0)
    if args.imgdisc > 0:
        domA_disc = MsImageDis(3)
        domB_disc = MsImageDis(3)

    l1 = nn.L1Loss()
    mse = nn.MSELoss()
    bce = nn.BCELoss()
    normaldist = normal.Normal(torch.tensor([0.0]), torch.tensor([1.0]))

    if torch.cuda.is_available():
        e1 = e1.cuda()
        e2 = e2.cuda()
        e3 = e3.cuda()
        decoder = decoder.cuda()
        disc = disc.cuda()
        zero_encoding = zero_encoding.cuda()
        if args.imgdisc > 0:
            domA_disc = domA_disc.cuda()
            domB_disc = domB_disc.cuda()

        A_label = A_label.cuda()
        B_label = B_label.cuda()

        l1 = l1.cuda()
        mse = mse.cuda()
        bce = bce.cuda()

    ae_params = list(e1.parameters()) + list(e2.parameters()) + list(
        e3.parameters()) + list(decoder.parameters())
    ae_optimizer = optim.Adam(ae_params, lr=args.lr, betas=(0.5, 0.999))


    disc_params = disc.parameters()
    disc_optimizer = optim.Adam(disc_params, lr=args.disclr, betas=(0.5, 0.999))

    if args.imgdisc > 0:
        imgdiscA_params = domA_disc.parameters()
        imgdiscA_optimizer = optim.Adam(imgdiscA_params, lr=args.disclr,
                                    betas=(0.5, 0.999))

        imgdiscB_params = domB_disc.parameters()
        imgdiscB_optimizer = optim.Adam(imgdiscB_params, lr=args.disclr,
                                    betas=(0.5, 0.999))

    if args.load != '':
        save_file = os.path.join(args.load, 'checkpoint')
        _iter = load_model(save_file, e1, e2, e3, decoder, ae_optimizer, disc,
                           disc_optimizer)

    e1 = e1.train()
    e2 = e2.train()
    e3 = e3.train()
    decoder = decoder.train()
    disc = disc.train()

    logger = Logger(args.out)

    print('Started training...')
    while True:
        domA_loader = torch.utils.data.DataLoader(domA_train, batch_size=args.bs,
                                                  shuffle=True, num_workers=6)
        domB_loader = torch.utils.data.DataLoader(domB_train, batch_size=args.bs,
                                                  shuffle=True, num_workers=6)
        if _iter >= args.iters:
            break

        for domA_img, domB_img in zip(domA_loader, domB_loader):
            if domA_img.size(0) != args.bs or domB_img.size(0) != args.bs:
                break

            domA_img = Variable(domA_img)
            domB_img = Variable(domB_img)

            if torch.cuda.is_available():
                domA_img = domA_img.cuda()
                domB_img = domB_img.cuda()

            domA_img = domA_img.view((-1, 3, args.resize, args.resize))
            domB_img = domB_img.view((-1, 3, args.resize, args.resize))

            ae_optimizer.zero_grad()

            A_common = e1(domA_img)
            A_separate_A = e2(domA_img)
            A_separate_B = e3(domA_img)
            A_encoding = torch.cat([A_common, A_separate_A, zero_encoding], dim=1)
            B_common = e1(domB_img)
            B_separate_A = e2(domB_img)
            B_separate_B = e3(domB_img)
            B_encoding = torch.cat([B_common, zero_encoding, B_separate_B], dim=1)

            A_decoding = decoder(A_encoding)
            B_decoding = decoder(B_encoding)

            A_common_decoding = decoder(torch.cat([A_common, zero_encoding,
                                                   B_separate_B], dim=1))
            B_common_decoding = decoder(torch.cat([B_common, zero_encoding,
                                                   B_separate_B], dim=1))
            A_common_separate_A = e2(A_common_decoding)
            A_common_separate_B = e3(A_common_decoding)
            B_common_separate_A = e2(B_common_decoding)
            B_common_separate_B = e3(B_common_decoding)

            A_reconstruction_loss = l1(A_decoding, domA_img)
            B_reconstruction_loss = l1(B_decoding, domB_img)
            A_separate_B_loss = mse(A_separate_B, zero_encoding)
            B_separate_A_loss = mse(B_separate_A, zero_encoding)
            A_common_separates_loss = mse(A_common_separate_A, zero_encoding) + \
                                      mse(A_common_separate_B, zero_encoding)
            B_common_separates_loss = mse(B_common_separate_A, zero_encoding) + \
                                      mse(B_common_separate_B, zero_encoding)

            logger.add_value('A_recon', A_reconstruction_loss)
            logger.add_value('B_recon', B_reconstruction_loss)
            logger.add_value('A_sep_B', A_separate_B_loss)
            logger.add_value('B_sep_A', B_separate_A_loss)
            logger.add_value('A_common_sep', A_common_separates_loss)
            logger.add_value('B_common_sep', B_common_separates_loss)

            loss = A_reconstruction_loss + B_reconstruction_loss + \
                   args.zeroweight * (A_separate_B_loss +
                                      B_separate_A_loss +
                                      A_common_separates_loss +
                                      B_common_separates_loss)

            if args.discweight > 0:
                preds_A = disc(A_common)
                preds_B = disc(B_common)
                distribution_adverserial_loss = args.discweight *\
                                                (bce(preds_A, B_label) + bce(preds_B, B_label))
                logger.add_value('distribution_adverserial', distribution_adverserial_loss)
                loss += distribution_adverserial_loss

            if args.reconweight > 0:
                normal_a = normaldist.sample(A_separate_A.size())\
                    .view(args.bs, args.sep * (args.resize // 64) * (args.resize // 64))
                normal_b = normaldist.sample(B_separate_B.size()) \
                    .view(args.bs, args.sep * (args.resize // 64) * (args.resize // 64))
                if torch.cuda.is_available():
                    normal_a = normal_a.cuda()
                    normal_b = normal_b.cuda()

                A_from_normal = decoder(torch.cat([A_common, normal_a,
                                              zero_encoding], dim=1))
                A_recon = e2(A_from_normal.view((-1, 3, args.resize, args.resize)))
                B_from_normal = decoder(torch.cat([B_common, zero_encoding,
                                                   normal_b], dim=1))
                B_recon = e3(B_from_normal.view((-1, 3, args.resize, args.resize)))

                content_recon_loss = mse(A_recon, normal_a) + mse(B_recon, normal_b)
                logger.add_value('content_recon', content_recon_loss)
                loss += args.reconweight * content_recon_loss

            if args.imgdisc > 0:
                AtoB = decoder(torch.cat([A_common, zero_encoding,
                                          B_separate_B], dim=1))
                BtoA = decoder(torch.cat([B_common, A_separate_A,
                                          zero_encoding], dim=1))

                image_adversarial_loss = domA_disc.calc_gen_loss(BtoA) + \
                                         domB_disc.calc_gen_loss(AtoB)
                logger.add_value('img_adversarial', image_adversarial_loss)
                loss += args.imgdisc * image_adversarial_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(ae_params, 5)
            ae_optimizer.step()

            if args.discweight > 0:
                disc_optimizer.zero_grad()

                A_common = e1(domA_img)
                B_common = e1(domB_img)

                disc_A = disc(A_common)
                disc_B = disc(B_common)

                loss = bce(disc_A, A_label) + bce(disc_B, B_label)
                logger.add_value('dist_disc', loss)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(disc_params, 5)
                disc_optimizer.step()

            if args.imgdisc > 0:
                A_common = e1(domA_img)
                B_common = e1(domB_img)
                A_separate_A = e2(domA_img)
                B_separate_B = e3(domB_img)
                AtoB = decoder(torch.cat([A_common, zero_encoding,
                                          B_separate_B], dim=1))
                BtoA = decoder(torch.cat([B_common, A_separate_A,
                                          zero_encoding], dim=1))

                loss_discA = domA_disc.calc_dis_loss(BtoA, domA_img)
                logger.add_value('img_disc_A', loss_discA)
                loss_discA.backward()
                torch.nn.utils.clip_grad_norm_(imgdiscA_params, 5)
                imgdiscA_optimizer.step()
                imgdiscA_optimizer.zero_grad()

                loss_discB = domB_disc.calc_dis_loss(AtoB, domB_img)
                logger.add_value('img_disc_B', loss_discB)
                loss_discB.backward()
                torch.nn.utils.clip_grad_norm_(imgdiscB_params, 5)
                imgdiscB_optimizer.step()
                imgdiscB_optimizer.zero_grad()


            if _iter % args.progress_iter == 0:
                print('Outfile: %s <<>> Iteration %d' % (args.out, _iter))

            if _iter % args.log_iter == 0:
                logger.log(_iter)

            logger.reset()

            if _iter % args.display_iter == 0:
                e1 = e1.eval()
                e2 = e2.eval()
                e3 = e3.eval()
                decoder = decoder.eval()

                save_imgs(args, e1, e2, e3, decoder, _iter, BtoA=True)
                save_imgs(args, e1, e2, e3, decoder, _iter, BtoA=False)
                save_stripped_imgs(args, e1, e2, e3, decoder, _iter, A=True)
                save_stripped_imgs(args, e1, e2, e3, decoder, _iter, A=False)

                e1 = e1.train()
                e2 = e2.train()
                e3 = e3.train()
                decoder = decoder.train()

            if _iter % args.save_iter == 0:
                save_file = os.path.join(args.out, 'checkpoint')
                save_model(save_file, e1, e2, e3, decoder, ae_optimizer, disc,
                           disc_optimizer, _iter)

            _iter += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default='')
    parser.add_argument('--out', default='out')
    parser.add_argument('--lr', type=float, default=0.0002)
    parser.add_argument('--bs', type=int, default=32)
    parser.add_argument('--iters', type=int, default=1250000)
    parser.add_argument('--resize', type=int, default=128)
    parser.add_argument('--crop', type=int, default=178)
    parser.add_argument('--sep', type=int, default=25)
    parser.add_argument('--discweight', type=float, default=0.001)
    parser.add_argument('--disclr', type=float, default=0.0002)
    parser.add_argument('--progress_iter', type=int, default=100)
    parser.add_argument('--display_iter', type=int, default=5000)
    parser.add_argument('--log_iter', type=int, default=100)
    parser.add_argument('--save_iter', type=int, default=10000)
    parser.add_argument('--load', default='')
    parser.add_argument('--num_display', type=int, default=12)
    parser.add_argument('--zeroweight', type=float, default=1.0)
    parser.add_argument('--reconweight', type=float, default=0.01)
    parser.add_argument('--imgdisc', type=float, default=0.001)

    args = parser.parse_args()

    train(args)
