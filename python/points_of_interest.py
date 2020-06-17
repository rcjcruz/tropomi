from collections import namedtuple

# Variables
# Define lat/lon for cities of interest
Point = namedtuple('Point', 'lon lat')
cities_coords = {'Toronto_coords': Point(-79.347, 43.651070),
                 'Ottawa_coords': Point(-75.6972, 45.4215),
                #  'Montreal_coords': Point(-73.5673, 45.5017),
                #  'New York_coords': Point(-74.0060, 40.7128),
                #  'Chicago_coords': Point(-87.6298, 41.8781),
                #  'Boston_coords': Point(-71.0589, 42.3601),
                #  'Washington, DC_coords': Point(-77.0369, 38.9072)
                 }

cities = {'toronto': Point(-79.3832, 43.6532),
            'montreal': Point(-73.5673, 45.5017),
            'new_york': Point(-74.0060, 40.7128),
            'vancouver': Point(-123.1207, 49.2827),
            'los_angeles': Point(-118.2437, 34.0522)}

# Define plot extent centred around Toronto
extent = 1
# must provide city 
# plot_limits = (city_coords.lon-extent,
#                 city_coords.lon+extent,
#                 city_coords.lat-extent,
#                 city_coords.lat+extent)