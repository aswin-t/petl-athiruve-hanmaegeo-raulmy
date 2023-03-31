import os
import re
import pandas as pd
import matplotlib.pyplot as plt


def analyze_results(df):

    for response in ['loss', 'val_loss', 'abas']:
        plt.figure(figsize=(8, 8))
        plt.suptitle(response)
        for cnt, feature in enumerate(['lr', 'wd', 'beta1', 'beta2']):
            plt.subplot(2, 2, cnt+1)
            plt.plot(df[feature], df[response], '.')
            plt.xlabel(feature)
            plt.ylabel(response)
        plt.show()


def optimization(folder, filename):
    lr_re = re.compile(r'.*learning_rate-(\d+\.*\d*|\d+e-*\d+)-')
    wd_re = re.compile(r'.*weight_decay-(\d+\.*\d*|\d+e-*\d+)-')
    beta1_re = re.compile(r'.*beta_1-(\d+\.*\d*|\d+e-*\d+)-')
    beta2_re = re.compile(r'.*beta_2-(\d+\.*\d*|\d+e-*\d+)-')
    history_re = re.compile(r'INFO - (\d+),(\d+\.*\d*),(\d+\.*\d*),(\d+\.*\d*),(\d+\.*\d*),(\d+\.*\d*),(\d+\.*\d*)\n')

    filepath = os.path.join(folder, filename)
    to_df = []
    iteration = train_loss = val_loss = -1
    with open(filepath, 'r') as infi:
        lines = infi.readlines()
        for line in lines:
            pat = history_re.search(line)
            if pat:
                iteration = int(pat.group(1))
                train_loss = float(pat.group(2))
                val_loss = float(pat.group(3))

            if 'Results:' in line:
                split = line.split(',')
                lr = float(lr_re.search(split[1]).group(1))
                try:
                    wd = float(wd_re.search(split[1]).group(1))
                except AttributeError:
                    print(split[1])
                beta1 = float(beta1_re.search(split[1]).group(1))
                beta2 = float(beta2_re.search(split[1]).group(1))
                scheduler = True if 'CosineDecayRestarts' in split[1] else False
                abas = float(split[7])

                to_df.append({'lr': lr, 'wd': wd, 'beta1': beta1, 'beta2': beta2, 'scheduler': scheduler, 'abas': abas,
                              'iteration': iteration, 'loss': train_loss, 'val_loss': val_loss})
                iteration = -1
                train_loss = -1
                val_loss = -1

    df = pd.DataFrame(to_df)
    analyze_results(df)


if __name__ == '__main__':
    fn = "betas-google_-_t5-base-lm-adapt-soft-benchmark.log"
    ff = '/home/aswin/Documents/MIDS/petl-athiruve-hanmaegeo-raulmy/checkpoints'
    optimization(ff, fn)
