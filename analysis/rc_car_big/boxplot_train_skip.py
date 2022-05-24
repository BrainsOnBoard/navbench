# %%
import matplotlib.pyplot as plt
import numpy as np
import rc_car_big

TRAIN_SKIP = [100, 50, 20, 10, 5, 1]
TEST_SKIP = 80
IM_SIZE = (45, 180)
PREPROC = 'None'


_, ax = plt.subplots()
data = []
labels = []

class ErrBoxPlot:
    def __init__(self, analysis, train_skip, preprocess):
        label = len(analysis.train_entries)
        labels.append(label)

    def on_test(self, train_route, test_route, df, preprocess, params):
        # Cap at 90°
        err = np.minimum(df.heading_error, 90)
        data.append(err)

    def on_finish(self, params):
        if len(data) == len(TRAIN_SKIP):
            ax.boxplot(data)
            ax.set_ylim(bottom=0)
            ax.set_xlabel('Number of training images')
            ax.set_xticklabels(labels)


paths = rc_car_big.get_paths()


rc_car_big.run_analysis(
    paths[0:1],
    [paths[5]],
    TRAIN_SKIP,
    [TEST_SKIP],
    [IM_SIZE],
    [PREPROC],
    runner_classes=[ErrBoxPlot])
plt.show()
