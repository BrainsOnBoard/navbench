#!/usr/bin/python
from argparse import ArgumentParser

import cv2
import matplotlib.pyplot as plt

import navbench as nb
from navbench import imgproc as ip


class IDFViewer(nb.Database):
    def __init__(self, database, goal=None, bound_size=None):
        nb.Database.__init__(self, database)
        self.figure, axes = plt.subplots(4)
        self.axes_image = None
        self.axes_diffim = None
        self.ax_goal, self.ax_imshow, _, self.ax_plot = axes
        self.ax_diff = axes[2]
        self.count = 0

        if goal is None:
            self.goal = goal = len(self) // 2
        else:
            self.goal = goal
        self.frame = goal
        print('Goal:', goal)

        if bound_size is None:
            bound_size = 100
        self.bounds = (max(0, goal - bound_size),
                       min(len(self) - 1, goal + bound_size))
        print('Bounds:', self.bounds)

        self.ax_imshow.set_xticks([])
        self.ax_imshow.set_yticks([])
        self.figure.canvas.mpl_connect(
            'button_press_event', self.mouse_clicked)

        resize = ip.resize(55, 180)
        entries = range(*self.bounds)
        self.images = self.read_images(entries, resize)
        self.snap = self.read_images(goal, resize)
        print(len(self.images), 'images loaded')

        idf = nb.route_idf(self.images, self.snap)
        ca = nb.calculate_ca(idf, medfilt_size=3)
        ca.plot(entries, filter_zeros=True, ax=self.ax_plot)

        self.ax_goal.set_xticks([])
        self.ax_goal.set_yticks([])
        self.ax_goal.imshow(self.snap, cmap='gray')

        self.ax_diff.set_xticks([])
        self.ax_diff.set_yticks([])

    def mouse_clicked(self, event):
        if event.inaxes == self.ax_plot and event.button == 1:
            self.frame = round(event.xdata)
            self.show_frame()

    def show_frame(self):
        image = self.images[int(self.frame - self.bounds[0])]
        diffim = cv2.subtract(self.snap, image)
        diffim = (diffim + 1) / 2

        if self.axes_image is None:
            self.axes_image = self.ax_imshow.imshow(image, cmap='gray')

            # **HACK**: If we show diffim here then axes_diffim ends up screwed
            # for some reason...
            self.axes_diffim = self.ax_diff.imshow(image, cmap='hot')
        else:
            self.axes_image.set_data(image)
            self.axes_diffim.set_data(diffim)
        self.ax_imshow.set_title('Frame %i/%i' % (self.frame + 1, len(self)))
        self.figure.canvas.draw()

    def run(self):
        self.show_frame()
        plt.show()


def main():
    parser = ArgumentParser(
        description='A tool for comparing image differences within an image database')
    parser.add_argument('--goal', nargs='?', type=int, help='goal frame')
    parser.add_argument('--bound-size', nargs='?', type=int,
                        help='number of frames either side of goal to test')
    parser.add_argument('database', help='database path')
    args = parser.parse_args()

    viewer = IDFViewer(args.database, args.goal, args.bound_size)
    viewer.run()


if __name__ == '__main__':
    main()
