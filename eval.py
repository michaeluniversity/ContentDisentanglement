import argparse
import os
import torch
from models import E1, E2, E3, Decoder
from utils import save_imgs, load_model_for_eval, save_chosen_imgs, \
    interpolate_fixed_common, interpolate_fixed_A, interpolate_fixed_B


def eval(args):
    e1 = E1(args.sep, int((args.resize / 64)))
    e2 = E2(args.sep, int((args.resize / 64)))
    e3 = E3(args.sep, int((args.resize / 64)))
    decoder = Decoder(int((args.resize / 64)))
    zero_encoding = torch.full((args.bs, args.sep * (args.resize // 64) * (
            args.resize // 64)), 0)

    if torch.cuda.is_available():
        e1 = e1.cuda()
        e2 = e2.cuda()
        e3 = e3.cuda()
        decoder = decoder.cuda()
        zero_encoding = zero_encoding.cuda()

    if args.load != '':
        save_file = os.path.join(args.load, 'checkpoint')
        _iter = load_model_for_eval(save_file, e1, e3, e2, decoder)

    e1 = e1.eval()
    e2 = e2.eval()
    e3 = e3.eval()
    decoder = decoder.eval()

    if not os.path.exists(args.out) and args.out != "":
        os.mkdir(args.out)

    # save_imgs(args, e1, e2, e3, decoder, _iter, True, 12)
    # save_imgs(args, e1, e2, e3, decoder, _iter, False, 12)
    # save_chosen_imgs(args, e1, e2, e3, decoder, _iter, [0,4,6,13,27], [2,4,9,23,
    #                                                             25], True)
    # save_chosen_imgs(args, e1, e2, e3, decoder, _iter, [0,4,6,13,27], [2,4,9,23,
    #                                                             25], False)
    interpolate_fixed_common(args, e1, e2, e3, decoder, 35, 27, 4, 45, 34)
    interpolate_fixed_A(args, e1, e2, e3, decoder, 34, 8, 4, 45, 35)
    interpolate_fixed_B(args, e1, e2, e3, decoder, 34, 8, 35, 27, 45)


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default='')
    parser.add_argument('--load', default='')
    parser.add_argument('--out', default='')
    parser.add_argument('--resize', type=int, default=128)
    parser.add_argument('--crop', type=int, default=178)
    parser.add_argument('--sep', type=int, default=25)
    parser.add_argument('--bs', type=int, default=64)
    parser.add_argument('--num_display', type=int, default=20)

    args = parser.parse_args()

    eval(args)
