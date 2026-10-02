OFFLINE ROUTING DATA
====================

The emergency engine works without road-routing data by calculating straight-line distance and a compass bearing to the nearest cached hospital.

For true offline road navigation, place a routing graph here as:
    data/routing/route.graphml

The graph must contain node latitude/longitude attributes. A graph can be generated from an OpenStreetMap regional extract using a routing tool such as OSMnx while internet is available, then copied to the device for offline use.

Do not describe straight-line bearing as turn-by-turn road navigation.
