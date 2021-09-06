import argparse


parser = argparse.ArgumentParser()


# ================================ Dirs ================================
parser.add_argument('--proj_dir', default=r'D:\dummyproj', type=str)
parser.add_argument('--data_dir', default=r'D:\dummyproj\data', type=str)


# ============================ Learning Configs ============================
parser.add_argument('--batch_size', default=64, type=int)
parser.add_argument('--lr', default=3e-4, type=float)
parser.add_argument('--train_val_split_p', default=0.9, type=float)
parser.add_argument('--n_ephocs', default=100, type=int)
parser.add_argument('--pre_load_net_fn', default=None, type=str)

