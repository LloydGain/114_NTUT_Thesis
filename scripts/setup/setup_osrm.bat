@echo off

set MAP=taiwan-latest

set DATA_DIR=%~dp0..\..\data\osrm

if not exist "%DATA_DIR%" (
    mkdir "%DATA_DIR%"
)

if not exist "%DATA_DIR%\%MAP%.osm.pbf" (
    powershell Invoke-WebRequest -Uri http://download.geofabrik.de/asia/%MAP%.osm.pbf -OutFile "%DATA_DIR%\%MAP%.osm.pbf"
)

if not exist "%DATA_DIR%\%MAP%.osrm" (
    echo Building OSRM graph...
    docker run --rm -v "%DATA_DIR%:/data" osrm/osrm-backend osrm-extract -p /opt/car.lua /data/%MAP%.osm.pbf
    docker run --rm -v "%DATA_DIR%:/data" osrm/osrm-backend osrm-partition /data/%MAP%.osrm
    docker run --rm -v "%DATA_DIR%:/data" osrm/osrm-backend osrm-customize /data/%MAP%.osrm
) else (
    echo OSRM graph already exists. Skipping build.
)

echo Starting OSRM server...
docker run --rm -p 5000:5000 -v "%DATA_DIR%:/data" osrm/osrm-backend osrm-routed --algorithm mld /data/%MAP%.osrm