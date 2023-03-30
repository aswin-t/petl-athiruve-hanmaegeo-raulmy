import os
import re
import pandas as pd


def optimization(folder, filename):
    lr_re = re.compile(r'.*learning_rate-(\d+\.*\d*|\d+e-*\d+)-')
    wd_re = re.compile(r'.*weight_decay-(\d+\.*\d*|\d+e-*\d+)-')
    filepath = os.path.join(folder, filename)
    to_df = []
    with open(filepath, 'r') as infi:
        lines = infi.readlines()
        for line in lines:
            if 'Results:' in line:
                split = line.split(',')
                lr = float(lr_re.search(split[1]).group(1))
                try:
                    wd = float(wd_re.search(split[1]).group(1))
                except AttributeError:
                    print(split[1])
                scheduler = True if 'CosineDecayRestarts' in split[1] else False
                abas = float(split[7])

                to_df.append({'lr': lr, 'wd': wd, 'scheduler': scheduler, 'abas': abas})

    df = pd.DataFrame(to_df)
    for col in df.columns:
        if col != 'abas':
            mean_ = df.groupby(col)['abas'].mean().sort_values(ascending=False)
            print(f'####')
            print(col)
            print(mean_)


if __name__ == '__main__':
    fn = "optimization_2-google_-_t5-base-lm-adapt-soft-benchmark.log"
    ff = '/home/aswin/Documents/MIDS/petl-athiruve-hanmaegeo-raulmy/checkpoints'
    optimization(ff, fn)
