#!/usr/bin/python
import sys

import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

import navbench as nb

SMALL_STEP = 1
BIG_STEP = 10  # frames


class DatabaseViewer(nb.Database):
    def __init__(self, path):
        nb.Database.__init__(self, path, fullpath=True)
        self.figure, self.axes = plt.subplots()
        self.figure.canvas.mpl_connect('key_press_event', self.key_pressed)
        self.axes_image = None

        self.axes.set_xticks([])
        self.axes.set_yticks([])

        axslider = self.figure.add_axes([.25, .1, .65, .03])
        self.fr_slider = Slider(axslider, 'Frame', 1,
                                len(self), valstep=SMALL_STEP)
        self.fr_slider.on_changed(lambda _: self.show_frame())

    def change_frame(self, change):
        newval = min(len(self), max(1, self.fr_slider.val + change))
        if newval != self.fr_slider.val:
            self.fr_slider.set_val(newval)
            self.show_frame()

    def key_pressed(self, event):
        if event.key == 'right':
            self.change_frame(SMALL_STEP)
        elif event.key == 'left':
            self.change_frame(-SMALL_STEP)
        elif event.key == 'up':
            self.change_frame(BIG_STEP)
        elif event.key == 'down':
            self.change_frame(-BIG_STEP)

    def show_frame(self):
        image = self.read_images(self.fr_slider.val - 1)

        if self.axes_image is None:
            self.axes_image = self.axes.imshow(image, cmap='gray')
        else:
            self.axes_image.set_data(image)
        self.axes.set_title('Frame %i/%i' % (self.fr_slider.val, len(self)))
        self.figure.canvas.draw()

    def run(self):
        self.show_frame()
        plt.show()


def usage():
    print(sys.argv[0], '[path to database]')
    sys.exit(1)


def main():
    if len(sys.argv) != 2:
        usage()

    try:
        viewer = DatabaseViewer(sys.argv[1])
    except FileNotFoundError:
        print(sys.argv[1], 'appears to not be a valid database')
        usage()
    viewer.run()


if __name__ == '__main__':
    main()
