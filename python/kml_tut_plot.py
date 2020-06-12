import simplekml
kml = simplekml.Kml()
ground = kml.newgroundoverlay(name='Toronto')
ground.icon.href = 'https://pasteboard.co/JcHg7yW.png'
ground.latlonbox.north = 44.65107
ground.latlonbox.south = 42.65107
ground.latlonbox.east =  -78.347
ground.latlonbox.west =  -80.347
ground.latlonbox.rotation = -14
kml.save("../kml/GroundOverlay.kml")