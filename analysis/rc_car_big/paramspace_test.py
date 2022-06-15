import rc_car_big
from paramspace import ParamDim
import bob_robotics.navigation as bobnav
import matplotlib.pyplot as plt
import gm_plotting

# TRAIN_SKIP = [50]
# TEST_SKIP = [80]
# IM_SIZE = [(45, 180)]
# # IM_SIZE = [(360, 1440)] , (180, 720), (90, 360), (45, 180), (22, 90)]
# PREPROC = ['ip.histeq']
# # PREPROC = ['None', 'ip.histeq']

class RouteQuiver:
    ax = dict()

    def __call__(self, df, train_route, test_route, **kwargs): #, test_route, df, preprocess, params):
        if not train_route.name in self.ax:
            _, self.ax[train_route.name] = plt.subplots()
        ax = self.ax[train_route.name]

        label = test_route.name.replace('unwrapped_', '')
        x, y = rc_car_big.to_merc(test_route)
        lines = ax.plot(x, y, '--', label=label)
        colour = lines[0].get_color()
        ax.plot(x[0], y[0], 'o', color=colour)

        bobnav.anglequiver(
            ax, x[df.index], y[df.index],
            # scale=1e6, scale_units='xy',
            df.heading, color=colour, zorder=lines[0].zorder + 1,
            alpha=0.8, invert_y=True)

    def on_finish(self):
        for ax in self.ax.values():
            gm_plotting.APIClient().add_satellite_image_background(ax)
            ax.legend(bbox_to_anchor=(1, 1))
        plt.show()

routeQuiver = RouteQuiver()
paths = rc_car_big.get_paths()
pspace = dict(train_route_path=paths[0],
              test_route_path=paths[0],
              train_skip=50,
              test_skip=80,
              test_skip_offset=0,
              im_size=ParamDim(default=(45, 180), values=[(45, 180)]),  #, values=[(360, 1440), (180, 720), (90, 360), (45, 180), (22, 90)]),
            #   preprocess_str=ParamDim(default='None', values=['None',
            #   'ip.histeq']))
              preprocess_str='ip.histeq')

rc_car_big.run_analysis_params(pspace, to_run=(routeQuiver,))
routeQuiver.on_finish()
print('EXITING')
