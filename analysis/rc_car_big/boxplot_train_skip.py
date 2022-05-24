# %%
import matplotlib.pyplot as plt
import numpy as np
import rc_car_big

TRAIN_SKIP = [100, 50, 20, 10, 5, 1]
TEST_SKIP = 80
IM_SIZE = (45, 180)
PREPROC = ['None', 'ip.remove_sky', 'ip.remove_sky_and_histeq']


train_dict = dict()
for proc in PREPROC:
    _, ax = plt.subplots()
    train_dict[proc] = { 'labels': [], 'data': [], 'ax': ax }

class ErrBoxPlot:
    def __init__(self, analysis, preprocess, params):
        label = len(analysis.train_entries)
        train_dict[params['preprocess']]['labels'].append(label)

    def on_test(self, train_route, test_route, df, preprocess, params):
        # Cap at 90°
        err = np.minimum(df.heading_error, 90)
        train_dict[params['preprocess']]['data'].append(err)

    def on_finish(self, params):
        cur = train_dict[params['preprocess']]
        if len(cur['data']) == len(TRAIN_SKIP):
            ax = cur['ax']
            ax.boxplot(cur['data'])
            ax.set_ylim(bottom=0)
            ax.set_xlabel('Number of training images')
            ax.set_xticklabels(cur['labels'])
            ax.set_title(f'Preprocessing: {params["preprocess"]}')


paths = rc_car_big.get_paths()


rc_car_big.run_analysis(
    paths[0:1],
    [paths[5]],
    TRAIN_SKIP,
    [TEST_SKIP],
    [IM_SIZE],
    PREPROC,
    runner_classes=[ErrBoxPlot])
plt.show()
