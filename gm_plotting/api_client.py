import json
import os
import urllib
from warnings import warn

import appdirs  # For finding cache dir
import googlemaps
import PIL

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
        filename = f'image_zoom{zoom}_coords{gps_coords[0]}_{gps_coords[1]}.png'
        filepath = os.path.join(self.cache_path, filename)
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        if not os.path.exists(filepath):
            with open(filepath, 'wb') as file:
                for data in self.client.static_map(
                        size=(640, 640),
                        center=gps_coords, zoom=zoom, maptype='satellite'):
                    file.write(data)

        return PIL.Image.open(filepath)
