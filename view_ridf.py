#!/usr/bin/python
from argparse import ArgumentParser

import cv2
import matplotlib.pyplot as plt

import navbench as nb
from navbench import improc as ip


class RIDFViewer(nb.Database):
    def __init__(self, database, goal=None, bound_size=None):
        nb.Database.__init__(self, database, fullpath=True)
        self.figure, axes = plt.subplots(4)
        self.axes_image = None
        self.axes_diffim = None
        self.ax_goal, self.ax_imshow, self.ax_ridf, self.ax_plot = axes
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

        resize = ip.resize(180, 55)
        entries = range(*self.bounds)
        self.images = self.read_images(entries, resize)
        self.snap = self.read_images(goal, resize)
        print(len(self.images), 'images loaded')

        errs = nb.route_ridf_errors(self.images, self.snap)
        ca_bounds, goal2 = nb.rca_bounds(errs)

        nb.plot_ca(entries, errs, ca_bounds, goal2, ax=self.ax_plot)

        self.ax_goal.set_xticks([])
        self.ax_goal.set_yticks([])
        self.ax_goal.imshow(self.snap, cmap='gray')

    def mouse_clicked(self, event):
        if event.inaxes == self.ax_plot and event.button == 1:
            self.frame = round(event.xdata)
            self.show_frame()

    def show_frame(self):
        image = self.images[int(self.frame - self.bounds[0])]
        ridf = nb.ridf(image, self.snap)

        if self.axes_image is None:
            self.axes_image = self.ax_imshow.imshow(image, cmap='gray')
        else:
            self.axes_image.set_data(image)
        self.ax_imshow.set_title('Frame %i/%i' % (self.frame + 1, len(self)))
        nb.plot_ridf(ridf, self.ax_ridf)
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

    viewer = RIDFViewer(args.database, args.goal, args.bound_size)
    viewer.run()


if __name__ == '__main__':
    main()
