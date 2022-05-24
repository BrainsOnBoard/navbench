import bob_robotics.navigation as bobnav
import gm_plotting
import matplotlib.pyplot as plt
import numpy as np

import rc_car_big


def distance2d(entry1, entry2):
    diff = np.array([entry1.x, entry1.y]) - np.array([entry2.x, entry2.y])
    if diff.ndim == 1:
        return np.hypot(*diff)
    else:
        return np.hypot(diff[0, :], diff[1, :])


class RouteQuiver:
    def __init__(self, analysis, preprocess, params):
        _, self.ax = plt.subplots()

        train_x, train_y = rc_car_big.to_merc(analysis.train_route)
        self.ax.plot(train_x, train_y, label='Training route')

    def on_test(self, train_route, test_route, df, preprocess, params):
        label = test_route.name.replace('unwrapped_', '')
        x, y = rc_car_big.to_merc(test_route)
        lines = self.ax.plot(x, y, '--', label=label)
        colour = lines[0].get_color()
        self.ax.plot(x[0], y[0], 'o', color=colour)

        bobnav.anglequiver(
            self.ax, x[df.index], y[df.index],
            # scale=1e6, scale_units='xy',
            df.heading, color=colour, zorder=lines[0].zorder + 1,
            alpha=0.8, invert_y=True)

    def on_finish(self, params):
        gm_plotting.APIClient().add_satellite_image_background(self.ax)

        self.ax.legend(bbox_to_anchor=(1, 1))

class DistanceVsError:
    def __init__(self, analysis, preprocess, params):
        _, self.ax = plt.subplots()

    def on_test(self, train_route, test_route, df, preprocess, params):
        label = test_route.name.replace('unwrapped_', '')

        # Cap at 90°
        err = np.minimum(df.heading_error, 90)
        dist_along_train = train_route.distance[df.nearest_train_idx]

        self.ax.scatter(dist_along_train, err, label=label, alpha=0.5, marker='.')
        self.ax.set_xlabel("Distance along training route (m)")
        self.ax.set_ylabel("Heading error (°)")

    def on_finish(self, params):
        self.ax.legend(bbox_to_anchor=(1, 1))
        self.ax.set_xlim(left=0)
        self.ax.set_ylim(bottom=0)


# class PlotRunner:
#     def __init__(self, analysis, train_skip, preprocess):
#         _, self.route_ax = plt.subplots()
#         _, self.dist_err_ax = plt.subplots()
#         # _, ax2 = plt.subplots()
#         _, self.scatter_ax = plt.subplots()

#         train_x, train_y = rc_car_big.to_merc(analysis.train_route)
#         self.route_ax.plot(train_x, train_y, label='Training route')

#     def on_test(self, train_route, test_route, df, preprocess, params):
#         label = test_route.name.replace('unwrapped_', '')
#         x, y = rc_car_big.to_merc(test_route)
#         lines = self.route_ax.plot(x, y, '--', label=label)
#         colour = lines[0].get_color()
#         self.route_ax.plot(x[0], y[0], 'o', color=colour)

#         nb.anglequiver(
#             self.route_ax, x[df.index], y[df.index],
#             # scale=1e6, scale_units='xy',
#             df.heading, color=colour, zorder=lines[0].zorder + 1,
#             alpha=0.8, invert_y=True)

#         # Cap at 90°
#         err = np.minimum(df.heading_error, 90)
#         dist_along_train = train_route.distance[df.nearest_train_idx]

#         self.dist_err_ax.scatter(dist_along_train, err, label=label, alpha=0.5, marker='.')
#         self.dist_err_ax.set_xlabel("Distance along training route (m)")
#         self.dist_err_ax.set_ylabel("Heading error (°)")

#         # ax2.plot(dist_along_train, medfilt(err, kernel_size=5), label=label)
#         # ax2.set_xlabel("Distance along training route (m)")
#         # ax2.set_ylabel("Heading error (°)")

#         dist_to_train = distance2d(df, train_route.loc[df.nearest_train_idx])
#         self.scatter_ax.scatter(dist_to_train, err, label=label, alpha=0.5, marker='.')
#         self.scatter_ax.set_xlabel("Distance to training route (m)")
#         self.scatter_ax.set_ylabel("Heading error (°)")

#     def on_finish(self, params):
#         gm_plotting.APIClient().add_satellite_image_background(self.route_ax)

#         self.route_ax.legend(bbox_to_anchor=(1, 1))

#         self.dist_err_ax.legend(bbox_to_anchor=(1, 1))
#         self.dist_err_ax.set_xlim(left=0)
#         self.dist_err_ax.set_ylim(bottom=0)

#         # ax2.legend(bbox_to_anchor=(1, 1))
#         # ax2.set_xlim(left=0)
#         # ax2.set_ylim(bottom=0)

#         self.scatter_ax.legend(bbox_to_anchor=(1, 1))
#         self.scatter_ax.set_xlim(0, 20)
#         self.scatter_ax.set_ylim(bottom=0)

#         plt.show()
