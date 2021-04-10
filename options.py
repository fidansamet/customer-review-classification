import argparse


class Options():
    def initialize(self, parser):
        parser.add_argument('--dataroot', type=str, default='./datasets/all_sentiment_shuffled.txt',
                            help='path to dataset')
        parser.add_argument('--phase', type=str, default='train', help='train or test phase')
        parser.add_argument('--name', type=str, default='knn', help='name of the experiment')
        parser.add_argument('--category', type=str, default='sentiment', help='which category will classifier use')
        parser.add_argument('--feature', type=str, default='unigram', help='which feature will classifier use')
        parser.add_argument('--discard_punct', action='store_true', help='if specified, discard punctuations')
        parser.add_argument('--discard_sw', action='store_true', help='if specified, discard stop words')
        parser.add_argument('--tfidf', action='store_true',
                            help='if specified, use TF-IDF for sentiment classification (only for unigram)')

        self.parser = parser
        return parser

    def print_options(self, opt):
        options = ''
        options += '----------------- Options ---------------\n'
        for k, v in vars(opt).items():
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            options += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        options += '----------------- End -------------------'
        print(options)

    def parse(self):
        parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.opt = self.initialize(parser).parse_args()
        self.print_options(self.opt)
        return self.opt
