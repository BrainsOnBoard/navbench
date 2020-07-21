#!/usr/bin/python
import sys

import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

import navbench as nb

BIG_INC = 10  # frames


def usage():
    print(sys.argv[0], '[path to database]')
    sys.exit(1)


def main():
    if len(sys.argv) != 2:
        usage()
    path = sys.argv[1]

    try:
        db = nb.Database(path, fullpath=True)
    except FileNotFoundError:
        print(sys.argv[1], 'appears to not be a valid database')
        usage()

    global axim
    axim = None
    def show_frame():
        global axim
        im = db.read_images(sframe.val - 1)

        # If we cache the returned AxisImage it is much faster to update this
        # rather than calling imshow() again
        if not axim:
            axim = ax.imshow(im, cmap='gray')
        else:
            axim.set_data(im)

        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title('Frame %i/%i' % (sframe.val, len(db)))
        fig.canvas.draw()

    def press(event):
        if event.key == 'right':
            if sframe.val < len(db):
                sframe.set_val(sframe.val + 1)
                show_frame()
        elif event.key == 'left':
            if sframe.val > 1:
                sframe.set_val(sframe.val - 1)
                show_frame()
        elif event.key == 'up':
            if sframe.val < len(db):
                sframe.set_val(min(sframe.val + BIG_INC, len(db)))
                show_frame()
        elif event.key == 'down':
            if sframe.val > 1:
                sframe.set_val(max(1, sframe.val - BIG_INC))
                show_frame()

    def update(_):
        show_frame()

    fig, ax = plt.subplots()
    fig.canvas.mpl_connect('key_press_event', press)

    axslider = fig.add_axes([.25, .1, .65, .03])
    sframe = Slider(axslider, 'Frame', 1, len(db), valstep=1)
    sframe.on_changed(update)

    show_frame()
    plt.show()


if __name__ == '__main__':
    main()
