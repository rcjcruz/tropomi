from collections import namedtuple

# Variables
# Define lat/lon for cities of interest
Point = namedtuple('Point', 'lon lat')
cities_coords = {'Toronto_coords': Point(-79.347, 43.651070),
                 'Hamilton_coords': Point(-79.8711, 43.2557),
                 'Ottawa_coords': Point(-75.6972, 45.4215),
                 'Egbert_coords': Point(-79.7609, 44.2337),
                 'Montreal_coords': Point(-73.5673, 45.5017),
                 'Vancouver_coords': Point(-123.1207, 49.2827),
                 'New York_coords': Point(-74.0060, 40.7128),
                 'Chicago_coords': Point(-87.6298, 41.8781),
                 'Boston_coords': Point(-71.0589, 42.3601),
                 'Los Angeles_coords': Point(-118.2437, 34.0522),
                 'Washington, DC_coords': Point(-77.0369, 38.9072)}


cities = {'toronto': Point(-79.3832, 43.6532),
          'hamilton': Point(-79.8711, 43.2557),
          'montreal': Point(-73.5673, 45.5017),
          'new_york': Point(-74.0060, 40.7128),
          'vancouver': Point(-123.1207, 49.2827),
          'los_angeles': Point(-118.2437, 34.0522)}

wind_type = {'A': [0, 2], 'B': [2, 4], 'C': [4, 6], 'D': [6, 8],
             'E': [8, 10], 'F': [10, 12], 'G': [12, 15], 'H': [15, 20], 'I': [20, 30]}

nomen_dict = {'may_1819': 'MAY89', 'may_20': 'MAY20',
                'march_19': 'MAR19', 'march_20': 'MAR20',
                'april_19': 'APR19', 'april_20': 'APR20',
                'june_19': 'JUN19', 'june_20': 'JUN20',
                'pre-vid': 'PRE19', 'covid':  'COV19',
                'test': 'TESTS',
                'rotated': 'ROT', 'cartesian': 'CAR'}


def get_plot_limits(city, extent=1, res=0.05):
    cities = {'toronto': Point(-79.3832, 43.6532),
              'montreal': Point(-73.5673, 45.5017),
              'hamilton': Point(-79.8711, 43.2557),
              'new_york': Point(-74.0060, 40.7128),
              'vancouver': Point(-123.1207, 49.2827),
              'los_angeles': Point(-118.2437, 34.0522)}
    try:
        city_coords = cities[city]
    except KeyError:
        print('Not a valid city. Valid cities include %s' %
              list(cities.keys()))
    else:
        plot_limits = (city_coords.lon-extent,  # llcrnlon
                       city_coords.lon+extent+res,  # urcrnlon
                       city_coords.lat-extent,  # llcrnlat
                       city_coords.lat+extent+res)  # urcrnlat
        return plot_limits
