#!/usr/bin/env python3
# Adapted from: https://github.com/BrainsOnBoard/bebop-demo/blob/master/src/plot/plot.py

import cv2
import matplotlib.pyplot as plt
import numpy as np
import sys
import time

# invoked when key pressed
def press(event):
    global wantsnap, kbkey, cmax, colprc, colprci, snapshot
    if event.key == ' ':
        global wantsnap
        wantsnap = True

    if event.key == "escape":
        kbkey = 0
    else:
        kbkey = 1 if len(event.key) != 1 else ord(event.key[0])
    if event.key == 'r':
        reset_fig()
    elif event.key == 'c':
        if colprci < len(colprc) - 1:
            colprci += 1
        else:
            colprci = 0
        if snapshot is not None:
            prc = colprc[colprci]
            cmax = 255 if prc == 0 else np.percentile(snapshot, 100 - prc) - \
                np.percentile(snapshot, prc)

# set the current snapshot to im
def set_snapshot(im):
    global snapshot, starttime, lastx, lasty, cmax
    starttime = time.time()
    snapshot = im
    ax_snap.imshow(snapshot, cmap='gray')
    ax_snap.axis('off')
    ax_snap.set_title('Stored view')
    ax_plot.cla()
    ax_plot.set_xlim([0, 30])
    ax_plot.grid()
    lastx = 0
    lasty = 0
    prc = colprc[colprci]
    cmax = 255 if prc == 0 else np.percentile(snapshot, 100 - prc) - \
        np.percentile(snapshot, prc)

# update the plot for a single frame
# we quit the program if user closes window or presses esc
def show(im):
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

    global lastx, lasty, wantsnap, kbkey
    if do_unwrap:
        # weirdly, we sometimes initially get images of 640x480 from the old
        # pixpro cameras, so we have to check!
        if im.shape[:2] == cam_res:
            im = cv2.remap(im, map_x, map_y, cv2.INTER_LINEAR)

    im_grey = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)

    # if user has requested a new snap, update
    if wantsnap:
        set_snapshot(im_grey)
        wantsnap = False

    # time is current time minus time when we took snap
    x = time.time() - starttime

    # return whatever key was pressed
    ret = kbkey
    kbkey = 1

    if not plt.fignum_exists(fignum):
        return 0

    # show current view
    ax_im.cla()
    ax_im.imshow(im)
    ax_im.axis('off')
    ax_im.set_title('Current view')

    # if snapshot is not set we don't plot anything else
    if snapshot is None:
        plt.pause(0.025) # render fig
        return ret

    if not plt.fignum_exists(fignum):
        return 0

    # get difference image
    diff = cv2.absdiff(im_grey, snapshot)

    # y is mean abs diff
    y = np.mean(diff, axis=(0, 1)) / 255

    # curtail colour axis if desired
    ax_diff.cla()
    prc = colprc[colprci]
    if prc != 0:
        diff = np.maximum(-cmax, np.minimum(cmax, diff))

    # show difference image
    ax_diff.imshow(diff)
    ax_diff.axis('off')
    ax_diff.set_title(
        'Image difference [%d/%d: %d]' % (100 - prc, prc, cmax))

    # plot mean abs diff over time
    ax_plot.plot([lastx, x], [lasty, y], 'r')

    # show 30s window
    if x > 30:
        ax_plot.set_xlim([x-20, x+10])

    plt.xlabel('Time (s)')
    plt.ylabel('Image difference')
    plt.title("%.2f fps" % (1 / (x - lastx)))

    # we need current x and y for next frame
    lastx = x
    lasty = y

    # render fig
    plt.pause(0.025)

    # return keyboard key pressed or zero
    return ret

# reset figure, deleting current snapshot
def reset_fig():
    global snapshot
    snapshot = None

    # reset axes
    ax_im.cla()
    ax_snap.cla()
    ax_diff.cla()
    ax_plot.cla()
    ax_snap.axis('off')
    ax_diff.axis('off')

# global variables
lastx = 0
lasty = 0
starttime = 0
wantsnap = False
kbkey = 1

# set up axes
ax_snap = plt.subplot2grid((2, 3), (0, 0))
ax_im = plt.subplot2grid((2, 3), (0, 1))
ax_diff = plt.subplot2grid((2, 3), (0, 2))
ax_plot = plt.subplot2grid((2, 3), (1, 0), colspan=3)
reset_fig()

# for curtailing colour axis
colprc = [0, 10, 25]
colprci = 0
cmax = 255

# open a new figure
fig = plt.gcf()
fignum = fig.number

# handle keypress events
fig.canvas.mpl_connect('key_press_event', press)

# disable default keypress handlers
keymaps = [key for key in plt.rcParams.keys() if key.startswith('keymap.')]
for k in keymaps:
    plt.rcParams[k] = []

# make full screen
figManager = plt.get_current_fig_manager()
figManager.full_screen_toggle()

# set plotting to interactive mode
plt.ion()

use_wifi = False
if len(sys.argv) > 1:
    if sys.argv[1] == "wifi":
        use_wifi = True
    else:
        cam_num = int(sys.argv[1])
else:
    cam_num = 0

is_pixpro = cam_num > 0  # dumb heuristic
do_unwrap = use_wifi or is_pixpro

map_x = []
map_y = []

if use_wifi:
    # Connect to PixPro over wifi (NB: only works with the old yellow ones!)
    cap = cv2.VideoCapture('http://172.16.0.254:9176')
else:
    print(f'Opening camera #{cam_num}')
    cap = cv2.VideoCapture(cam_num)
    assert cap.isOpened()

    if is_pixpro:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1440)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1440)

if do_unwrap:
    cam_res = (cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # values taken from pixpro_wifi.yaml in BoB robotics
    unwrap = {'centre': (0.5, 0.524414),
              'inner': 0.0244141,
              'outer': 0.5}

    outer_pixel = cam_res[1] * unwrap['outer']
    inner_pixel = cam_res[1] * unwrap['inner']
    centre_pixel = (unwrap['centre'][0] * cam_res[0], unwrap['centre'][1] * cam_res[1])

    unwrap_res = (90, 360)
    map_x = np.zeros(unwrap_res, dtype=np.float32)
    map_y = np.zeros(unwrap_res, dtype=np.float32)
    for i in range(unwrap_res[0]):
        for j in range(unwrap_res[1]):
            ifrac = i / unwrap_res[0]
            r = ifrac * (outer_pixel - inner_pixel) + inner_pixel
            th = 2 * np.pi * j / unwrap_res[1]

            map_x[i, j] = centre_pixel[0] - r * np.sin(th)
            map_y[i, j] = centre_pixel[1] + r * np.cos(th)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if show(frame) == 0:
        break

cap.release()
cv2.destroyAllWindows()
