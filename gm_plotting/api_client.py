import json
import os
import urllib

import appdirs  # For finding cache dir
import googlemaps
import numpy as np
import PIL

from . import coords


class APIClient:
    cache_path = appdirs.user_cache_dir('gm_plotting')

    def __init__(self, key=None, **kwargs):
        # If we don't have the module, then we can't make a client
        if not googlemaps:
            return

        # Try to get it from environment variable
        if not key:
            key = os.environ['GOOGLE_MAPS_API_KEY']

        # Client for connecting to Google's API
        self.client = googlemaps.Client(key=key, **kwargs)

    def address_to_gps(self, address):
        filename = f'address_{urllib.parse.quote_plus(address)}.json'
        filepath = os.path.join(self.cache_path, filename)
        if os.path.exists(filepath):
            with open(filepath, 'r') as file:
                details = json.load(file)
        else:
            details = self.client.geocode(address)
            with open(filepath, 'w') as file:
                json.dump(details, file)

        return tuple(details[0]['geometry']['location'].values())

    def get_satellite_image(self, gps_coords, zoom=15):
        filename = f'image_zoom{zoom}_coords{gps_coords[0]:.8f}_{gps_coords[1]:.8f}.png'
        filepath = os.path.join(self.cache_path, filename)
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        if not os.path.exists(filepath):
            with open(filepath, 'wb') as file:
                for data in self.client.static_map(
                        size=(640, 640),
                        center=gps_coords, zoom=zoom, maptype='satellite'):
                    file.write(data)

        return PIL.Image.open(filepath)

    def add_satellite_image_background(self, ax):
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        mlat = np.mean(ylim)
        mlon = np.mean(xlim)

        mframe = 1.25 * 2.0 ** -np.arange(22)
        def contains_lims(mframe):
            return xlim[0] >= mlon - mframe and xlim[1] <= mlon + mframe and ylim[0] >= mlat - mframe and ylim[1] <= mlat + mframe

        # Get highest zoom level which includes whole plot
        zoom = len(mframe) - next(i for i, mf in enumerate(reversed(mframe), 1) if contains_lims(mf))
        mframe = mframe[zoom]

        # Get satellite image for these coordinates
        centre_gps = coords.merc_to_gps(mlat, mlon)
        img = self.get_satellite_image(centre_gps, zoom=zoom)

        # Calculate boundaries for background image
        extent = (mlon - mframe, mlon + mframe, mlat + mframe, mlat - mframe)
        ax.imshow(img, extent=extent, zorder=-np.inf)

        # Put back axis limits as they were. Note that y direction is reversed.
        ax.set_xlim(xlim)
        ax.set_ylim(ylim[::-1])
