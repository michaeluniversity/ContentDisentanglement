import os

import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from random import shuffle
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.utils as vutils

from PIL import Image


def save_imgs(args, e1, e2, e3, decoder, iters, BtoA=True, num_offsets=1):
    ''' saves images of translation B -> A or A -> B'''
    test_domA, test_domB = get_test_imgs(args)

    for k in range(num_offsets):
        exps = []
        for i in range(k * args.num_display,  (k + 1) * args.num_display):
            with torch.no_grad():
                if i == k * args.num_display:
                    filler = test_domB[i].unsqueeze(0).clone()
                    exps.append(filler.fill_(0))

                if BtoA:
                    exps.append(test_domB[i].unsqueeze(0))
                else:
                    exps.append(test_domA[i].unsqueeze(0))

        if BtoA:
            for i in range(k * args.num_display,  (k + 1) * args.num_display):
                exps.append(test_domA[i].unsqueeze(0))
                separate_A = e2(test_domA[i].unsqueeze(0))
                for j in range(k * args.num_display,  (k + 1) * args.num_display):
                    with torch.no_grad():
                        common_B = e1(test_domB[j].unsqueeze(0))
                        zero_encoding = torch.full((1, args.sep * (args.resize
                                                                   // 64) * ( args.resize // 64)), 0)
                        if torch.cuda.is_available():
                            zero_encoding = zero_encoding.cuda()

                        BA_encoding = torch.cat([common_B, separate_A, zero_encoding], dim=1)
                        BA_decoding = decoder(BA_encoding)
                        exps.append(BA_decoding)
        else:
            for i in range(k * args.num_display,  (k + 1) * args.num_display):
                exps.append(test_domB[i].unsqueeze(0))
                separate_B = e3(test_domB[i].unsqueeze(0))
                for j in range(k * args.num_display,  (k + 1) * args.num_display):
                    with torch.no_grad():
                        common_A = e1(test_domA[j].unsqueeze(0))
                        zero_encoding = torch.full((1, args.sep * (args.resize
                                                                   // 64) * ( args.resize // 64)), 0)
                        if torch.cuda.is_available():
                            zero_encoding = zero_encoding.cuda()

                        AB_encoding = torch.cat(
                            [common_A, zero_encoding, separate_B], dim=1)
                        AB_decoding = decoder(AB_encoding)
                        exps.append(AB_decoding)

        with torch.no_grad():
            exps = torch.cat(exps, 0)

        if BtoA:
            vutils.save_image(exps,
                              '%s/experiments_%06d_%d-BtoA.png' % (args.out,
                                                                   iters,
                                                                   k),
                              normalize=True, nrow=args.num_display + 1)
        else:
            vutils.save_image(exps,
                              '%s/experiments_%06d_%d-AtoB.png' % (args.out,
                                                                   iters,
                                                                   k),
                              normalize=True, nrow=args.num_display + 1)


def save_chosen_imgs(args, e1, e2, e3, decoder, iters, listA, listB, BtoA=True):
    ''' saves images of translation B -> A or A -> B'''
    test_domA, test_domB = get_test_imgs(args)

    exps = []
    for i in range(args.num_display):
        with torch.no_grad():
            if i == 0:
                filler = test_domB[i].unsqueeze(0).clone()
                exps.append(filler.fill_(0))

            if BtoA:
                exps.append(test_domB[listB[i]].unsqueeze(0))
            else:
                exps.append(test_domA[listA[i]].unsqueeze(0))

    if BtoA:
        for i in listA:
            exps.append(test_domA[i].unsqueeze(0))
            separate_A = e2(test_domA[i].unsqueeze(0))
            for j in listB:
                with torch.no_grad():
                    common_B = e1(test_domB[j].unsqueeze(0))
                    zero_encoding = torch.full((1, args.sep * (args.resize
                                                               // 64) * ( args.resize // 64)), 0)
                    if torch.cuda.is_available():
                        zero_encoding = zero_encoding.cuda()

                    BA_encoding = torch.cat([common_B, separate_A, zero_encoding], dim=1)
                    BA_decoding = decoder(BA_encoding)
                    exps.append(BA_decoding)
    else:
        for i in listB:
            exps.append(test_domB[i].unsqueeze(0))
            separate_B = e3(test_domB[i].unsqueeze(0))
            for j in listA:
                with torch.no_grad():
                    common_A = e1(test_domA[j].unsqueeze(0))
                    zero_encoding = torch.full((1, args.sep * (args.resize
                                                               // 64) * ( args.resize // 64)), 0)
                    if torch.cuda.is_available():
                        zero_encoding = zero_encoding.cuda()

                    AB_encoding = torch.cat(
                        [common_A, zero_encoding, separate_B], dim=1)
                    AB_decoding = decoder(AB_encoding)
                    exps.append(AB_decoding)

    with torch.no_grad():
        exps = torch.cat(exps, 0)

    if BtoA:
        vutils.save_image(exps,
                          '%s/experiments_%06d-BtoA.png' % (args.out,iters),
                          normalize=True, nrow=args.num_display + 1)
    else:
        vutils.save_image(exps,
                          '%s/experiments_%06d-AtoB.png' % (args.out,iters),
                          normalize=True, nrow=args.num_display + 1)


def save_stripped_imgs(args, e1, e2, e3, decoder, iters, A=True):
    test_domA, test_domB = get_test_imgs(args)
    exps = []
    zero_encoding = torch.full((1, args.sep * (args.resize // 64) * (
            args.resize // 64)), 0)
    if torch.cuda.is_available():
        zero_encoding = zero_encoding.cuda()

    for i in range(args.num_display):
        if A:
            image = test_domA[i]
        else:
            image = test_domB[i]
        exps.append(image.unsqueeze(0))
        common = e1(image.unsqueeze(0))
        content_zero_encoding = torch.full(common.size(), 0)
        if torch.cuda.is_available():
            content_zero_encoding = content_zero_encoding.cuda()
        separate_A = e2(image.unsqueeze(0))
        separate_B = e3(image.unsqueeze(0))
        exps.append(decoder(torch.cat([common, zero_encoding, zero_encoding], dim=1)))
        exps.append(decoder(torch.cat([content_zero_encoding, separate_A, zero_encoding], dim=1)))
        exps.append(decoder(torch.cat([content_zero_encoding, zero_encoding, separate_B], dim=1)))

    with torch.no_grad():
        exps = torch.cat(exps, 0)

    if A:
        vutils.save_image(exps,
                          '%s/experiments_%06d-Astripped.png' % (args.out, iters),
                          normalize=True, nrow=args.num_display)
    else:
        vutils.save_image(exps,
                          '%s/experiments_%06d-Bstripped.png' % (args.out, iters),
                          normalize=True, nrow=args.num_display)


def interpolate_fixed_common(args, e1, e2, e3, decoder, imgA1, imgA2, imgB1,
                          imgB2, content_img):
    test_domA, test_domB = get_test_imgs(args)
    exps = []
    common = e1(test_domB[content_img].unsqueeze(0))
    a1 = e2(test_domA[imgA1].unsqueeze(0))
    a2 = e2(test_domA[imgA2].unsqueeze(0))
    b1 = e3(test_domB[imgB1].unsqueeze(0))
    b2 = e3(test_domB[imgB2].unsqueeze(0))
    with torch.no_grad():
        filler = test_domB[0].unsqueeze(0).clone()
        exps.append(filler.fill_(0))
        exps.append(test_domA[imgA1].unsqueeze(0))
        for i in range(args.num_display - 2):
            exps.append(filler.fill_(0))
        exps.append(test_domA[imgA2].unsqueeze(0))

        for i in range(args.num_display):
            if i == 0:
                exps.append(test_domB[imgB1].unsqueeze(0))
            elif i == args.num_display - 1:
                exps.append(test_domB[imgB2].unsqueeze(0))
            else:
                exps.append(filler.fill_(0))

            for j in range(args.num_display):
                cur_sep_A = (float(j) / (args.num_display - 1)) * a2 + \
                            (1 - float(j) / (args.num_display - 1)) * a1
                cur_sep_B = (float(i) / (args.num_display - 1)) * b2 + \
                            (1 - float(i) / (args.num_display - 1)) * b1
                encoding = torch.cat([common, cur_sep_A, cur_sep_B], dim=1)
                decoding = decoder(encoding)
                exps.append(decoding)

    with torch.no_grad():
        exps = torch.cat(exps, 0)

    vutils.save_image(exps,
                      '%s/interpolation_fixed_C.png' % (args.out),
                      normalize=True, nrow=args.num_display + 1)


def interpolate_fixed_A(args, e1, e2, e3, decoder, imgC1, imgC2, imgB1,
                          imgB2, imgA):
    test_domA, test_domB = get_test_imgs(args)
    exps = []
    c1 = e1(test_domB[imgC1].unsqueeze(0))
    c2 = e1(test_domB[imgC2].unsqueeze(0))
    a = e2(test_domA[imgA].unsqueeze(0))
    b1 = e3(test_domB[imgB1].unsqueeze(0))
    b2 = e3(test_domB[imgB2].unsqueeze(0))
    with torch.no_grad():
        filler = test_domB[0].unsqueeze(0).clone()
        exps.append(filler.fill_(0))
        exps.append(test_domB[imgC1].unsqueeze(0))
        for i in range(args.num_display - 2):
            exps.append(filler.fill_(0))
        exps.append(test_domB[imgC2].unsqueeze(0))

        for i in range(args.num_display):
            if i == 0:
                exps.append(test_domB[imgB1].unsqueeze(0))
            elif i == args.num_display - 1:
                exps.append(test_domB[imgB2].unsqueeze(0))
            else:
                exps.append(filler.fill_(0))

            for j in range(args.num_display):
                cur_common = (float(j) / (args.num_display - 1)) * c2 + \
                            (1 - float(j) / (args.num_display - 1)) * c1
                cur_sep_B = (float(i) / (args.num_display - 1)) * b2 + \
                            (1 - float(i) / (args.num_display - 1)) * b1
                encoding = torch.cat([cur_common, a, cur_sep_B], dim=1)
                decoding = decoder(encoding)
                exps.append(decoding)

    with torch.no_grad():
        exps = torch.cat(exps, 0)

    vutils.save_image(exps,
                      '%s/interpolation_fixed_A.png' % (args.out),
                      normalize=True, nrow=args.num_display + 1)


def interpolate_fixed_B(args, e1, e2, e3, decoder, imgC1, imgC2, imgA1,
                          imgA2, imgB):
    test_domA, test_domB = get_test_imgs(args)
    exps = []
    c1 = e1(test_domB[imgC1].unsqueeze(0))
    c2 = e1(test_domB[imgC2].unsqueeze(0))
    a1 = e2(test_domA[imgA1].unsqueeze(0))
    a2 = e2(test_domA[imgA2].unsqueeze(0))
    b = e3(test_domB[imgB].unsqueeze(0))
    with torch.no_grad():
        filler = test_domB[0].unsqueeze(0).clone()
        exps.append(filler.fill_(0))
        exps.append(test_domB[imgC1].unsqueeze(0))
        for i in range(args.num_display - 2):
            exps.append(filler.fill_(0))
        exps.append(test_domB[imgC2].unsqueeze(0))

        for i in range(args.num_display):
            if i == 0:
                exps.append(test_domA[imgA1].unsqueeze(0))
            elif i == args.num_display - 1:
                exps.append(test_domA[imgA2].unsqueeze(0))
            else:
                exps.append(filler.fill_(0))

            for j in range(args.num_display):
                cur_common = (float(j) / (args.num_display - 1)) * c2 + \
                            (1 - float(j) / (args.num_display - 1)) * c1
                cur_sep_A = (float(i) / (args.num_display - 1)) * a2 + \
                            (1 - float(i) / (args.num_display - 1)) * a1
                encoding = torch.cat([cur_common, cur_sep_A, b], dim=1)
                decoding = decoder(encoding)
                exps.append(decoding)

    with torch.no_grad():
        exps = torch.cat(exps, 0)

    vutils.save_image(exps,
                      '%s/interpolation_fixed_B.png' % (args.out),
                      normalize=True, nrow=args.num_display + 1)


def output_images(args, e1, e2, e3, decoder):
    test_domA, test_domB = get_test_imgs(args)

    if not os.path.exists(args.out + "/AtoB") and args.out != "":
        os.mkdir(args.out + "/AtoB")
        os.mkdir(args.out + "/BtoA")

    for i in range(len(test_domA)):
        for j in range(len(test_domB)):
            a = test_domA[i].unsqueeze(0)
            b = test_domB[j].unsqueeze(0)
            a_common = e1(a)
            b_common = e1(b)
            a_sep = e2(a)
            b_sep = e3(b)
            zero = torch.full(a_sep.size(), 0).cuda()

            AtoB_enc = torch.cat([a_common, zero, b_sep], dim=1)
            BtoA_enc = torch.cat([b_common, a_sep, zero], dim=1)

            AtoB = decoder(AtoB_enc)
            BtoA = decoder(BtoA_enc)

            vutils.save_image([AtoB],
                              '%s/AtoB/%d_%d.png' % (args.out, i, j),
                              normalize=True, nrow=1)
            vutils.save_image([BtoA],
                              '%s/BtoA/%d_%d.png' % (args.out, i, j),
                              normalize=True, nrow=1)


def output_for_user_study(args, e1, e2, e3, decoder):
    test_domA, test_domB = get_test_imgs(args)

    if not os.path.exists(args.out + "/1") and args.out != "":
        os.mkdir(args.out + "/1")

    idx_a = [x for x in range(len(test_domA))]
    idx_b = [x for x in range(len(test_domB))]

    shuffle(idx_a)
    shuffle(idx_b)

    for i in range(20):
        a = test_domA[idx_a[i]].unsqueeze(0)
        b = test_domB[idx_b[i]].unsqueeze(0)
        a_c = e1(a)
        b_s = e3(b)
        zero = torch.full(b_s.size(), 0).cuda()
        enc = torch.cat([a_c, zero, b_s], dim=1)
        dec = decoder(enc)

        os.mkdir(args.out + ("/1/%d" % i))
        vutils.save_image([a],'%s/1/%d/a.png' % (args.out, i),
                          normalize=True, nrow=1)
        vutils.save_image([b],'%s/1/%d/b.png' % (args.out, i),
                          normalize=True, nrow=1)
        vutils.save_image([dec],'%s/1/%d/ab.png' % (args.out, i),
                          normalize=True, nrow=1)



def get_test_imgs(args):
    comp_transform = transforms.Compose([
        transforms.CenterCrop(args.crop),
        transforms.Resize(args.resize),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    domA_test = CustomDataset(os.path.join(args.root, 'testA.txt'), transform=comp_transform)
    domB_test = CustomDataset(os.path.join(args.root, 'testB.txt'), transform=comp_transform)

    domA_test_loader = torch.utils.data.DataLoader(domA_test, batch_size=64,
                                                   shuffle=False, num_workers=6)
    domB_test_loader = torch.utils.data.DataLoader(domB_test, batch_size=64,
                                                   shuffle=False, num_workers=6)

    for domA_img in domA_test_loader:
        domA_img = Variable(domA_img)
        if torch.cuda.is_available():
            domA_img = domA_img.cuda()
        domA_img = domA_img.view((-1, 3, args.resize, args.resize))
        domA_img = domA_img[:]
        break

    for domB_img in domB_test_loader:
        domB_img = Variable(domB_img)
        if torch.cuda.is_available():
            domB_img = domB_img.cuda()
        domB_img = domB_img.view((-1, 3, args.resize, args.resize))
        domB_img = domB_img[:]
        break

    return domA_img, domB_img


def save_model(out_file, e1, e3, e2, decoder, ae_opt, disc, disc_opt,
               imgdisc_A, imgA_opt, imgdisc_B, imgB_opt, iters, img_disc_arg):
    state = {
        'e1': e1.state_dict(),
        'e2': e2.state_dict(),
        'e3': e3.state_dict(),
        'decoder': decoder.state_dict(),
        'ae_opt': ae_opt.state_dict(),
        'disc': disc.state_dict(),
        'disc_opt': disc_opt.state_dict(),
        'iters': iters
    }
    if img_disc_arg > 0:
        state['imgdisc_A'] = imgdisc_A.state_dict()
        state['imgA_opt'] =  imgA_opt.state_dict()
        state['imgdisc_B'] = imgdisc_B.state_dict()
        state['imgA_opt'] = imgB_opt.state_dict()
    torch.save(state, out_file)
    return


def load_model(load_path, e1, e2, e3, decoder, ae_opt, disc, disc_opt,
               imgdisc_A, imgA_opt, imgdisc_B, imgB_opt, img_disc_arg):
    state = torch.load(load_path)
    e1.load_state_dict(state['e1'])
    e2.load_state_dict(state['e2'])
    e3.load_state_dict(state['e3'])
    decoder.load_state_dict(state['decoder'])
    ae_opt.load_state_dict(state['ae_opt'])
    disc.load_state_dict(state['disc'])
    disc_opt.load_state_dict(state['disc_opt'])
    if img_disc_arg > 0:
        imgdisc_A.load_state_dict(state['imgdisc_A'])
        imgA_opt.load_state_dict(state['imgA_opt'])
        imgdisc_B.load_state_dict(state['imgdisc_B'])
        imgB_opt.load_state_dict(state['imgA_opt'])
    return state['iters']


def load_model_for_eval(load_path, e1, e2, e3, decoder, ):
    state = torch.load(load_path)
    e1.load_state_dict(state['e1'])
    e2.load_state_dict(state['e2'])
    e3.load_state_dict(state['e3'])
    decoder.load_state_dict(state['decoder'])
    return state['iters']


IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def default_loader(path):
    return Image.open(path).convert('RGB')


class Logger():
    def __init__(self, path):
        self.full_path = '%s/log.txt' % path
        self.log_file = open(self.full_path, 'w+')
        self.log_file.close()
        self.map = {}

    def add_value(self, tag, value):
        self.map[tag] = value

    def log(self, iter):
        self.log_file = open(self.full_path, 'a')
        self.log_file.write('iter: %7d' % iter)
        for k,v in self.map.items():
            self.log_file.write('\t %s: %10.7f' % (k, v))
        self.log_file.write('\n')
        self.log_file.close()

    def reset(self):
        self.map = {}


class CustomDataset(data.Dataset):
    def __init__(self, path, transform=None, return_paths=False,
                 loader=default_loader):
        super(CustomDataset, self).__init__()

        with open(path) as f:
            imgs = [s.replace('\n', '') for s in f.readlines()]

        if len(imgs) == 0:
            raise (RuntimeError("Found 0 images in: " + path + "\n"
                                                               "Supported image extensions are: " +
                                ",".join(IMG_EXTENSIONS)))

        self.imgs = imgs
        self.transform = transform
        self.return_paths = return_paths
        self.loader = loader

    def __getitem__(self, index):
        path = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.return_paths:
            return img, path
        else:
            return img

    def __len__(self):
        return len(self.imgs)


if __name__ == '__main__':
    pass
