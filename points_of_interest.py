from collections import namedtuple

# Variables
# Define lat/lon for cities of interest
Point = namedtuple('Point', 'lon lat')
cities_coords = {'Toronto_coords': Point(-79.3832, 43.6532),
                 'Ottawa_coords': Point(-75.6972, 45.5215),
                 'Montreal_coords': Point(-73.5673, 45.5017),
                 'New York_coords': Point(-74.0060, 40.7128),
                 'Chicago_coords': Point(-87.6298, 41.8781),
                 'Boston_coords': Point(-71.0589, 42.3601),
                 'Washington, DC_coords': Point(-77.0369, 38.9072)
                 }

# Define plot extent centred around Toronto
extent_size = 10
plot_limits = (cities_coords['Toronto_coords'].lon - extent_size,
               cities_coords['Toronto_coords'].lon + extent_size,
               cities_coords['Toronto_coords'].lat - extent_size,
               cities_coords['Toronto_coords'].lat + extent_size)
