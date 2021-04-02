from options import Options
from data_loader import DataLoader

if __name__ == '__main__':
    opt = Options().parse()
    data_loader = DataLoader(opt)
