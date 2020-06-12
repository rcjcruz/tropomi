import simplekml
import points_of_interest as poi 

kml = simplekml.Kml()
fol = kml.newfolder(name='Ferb', description='Image of Ferb')
ground = fol.newgroundoverlay(name='GroundOverlay')
ground.icon.href = 'https://i.pinimg.com/originals/3b/a5/7e/3ba57eaaaa9c99fc92aad3fc6a3620dc.png'
ground.gxlatlonquad.coords = [(18.410524,-33.903972),(18.411429,-33.904171),
                              (18.411757,-33.902944),(18.410850,-33.902767)]
# or
#ground.latlonbox.north = -33.902828
#ground.latlonbox.south = -33.904104
#ground.latlonbox.east =  18.410684
#ground.latlonbox.west =  18.411633
#ground.latlonbox.rotation = -14
kml.save("GroundOverlay.kml")