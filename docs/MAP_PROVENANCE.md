# Native ArcGIS map provenance

## Export method

These figures were newly exported with the installed, licensed ArcGIS Pro 3.6 `arcpy.mp` layout engine on November 7, 2026. Existing ArcGIS cartography supplied the Light Gray basemap template. New layouts and a temporary geodatabase were saved separately under the Desktop portfolio's arcgis-exports-private folder. Original APRX projects, shapefiles and research inputs were not saved over.

## Chicago prototype

Source: Chicago_gun_violence/chicago_hex_0_2mi_landmarks_wgs84. The original shapefile's CRS is read by ArcGIS. A fresh count check found 347 polygons and a sum of 616 in lm_count. Four explicit classes show 0, 1 to 5, 6 to 20 and 21 to 79 landmarks. This upstream preparation artifact is not the final 11,448-cell model grid. The final table contains 320 Chicago rows; the prototype and final-model IDs must not be treated as interchangeable.

## Boston source facilities

Source: boston_boundary_wkt.csv and the project's Schools.geojson, University.geojson and Hospitals.geojson. The Boston boundary has longitude/latitude bounds and is used in WGS84; source point geometries are spatially filtered against it. Counts were checked against the earlier source-map record: 270 schools, 25 universities and 26 hospitals. The municipal boundary includes water. The image displays source features, not trained-model counts, scores or causal effects.

## Review and exclusions

Legend overflow is checked before export. Figures are visually inspected, retain basemap attribution, and do not display real tweet data. Loose boundary CSVs elsewhere in the old 35-city preparation folder contain mixed coordinate magnitudes and incomplete CRS declarations. They were not combined under a guessed WGS84 label. This observation concerns those loose inputs; it does not establish that the trained model or every historical map used them incorrectly.
