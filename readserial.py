#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 17 19:34:26 2019

@author: phakawat
"""

import serial
import re
import pynmea2

GAS_PORT = "/dev/ttyUSB0"
GPS_PORT = "/dev/ttyACM1"

class ReadSerial:
    def __init__(self):
        self.gas = serial.Serial(GAS_PORT,baudrate=9600,timeout=2.5)
#        self.gps = serial.Serial(GPS_PORT,baudrate=9600,timeout=2.5)
        
    def stripGasData(self,gas_data):
        gas_data = re.sub(".+:","",gas_data)
        gas_data = gas_data.replace("ppm","")
        return gas_data
    
    def readGas(self):
        gas_data = self.gas.readline()
        gas_data = gas_data.decode("utf8")
        gas_data = gas_data.split()
        all_gas = []
        try:
            for gas in gas_data:
                gas = self.stripGasData(gas)
                all_gas.append(float(gas))
        except Exception as err:
            print(err)
            all_gas = [0,0,0]
        return all_gas
    
    def readGPS(self):
        gps_data = self.gps.readline()
        gps_data = gps_data.decode("utf8")
        if gps_data[0:6]=="$GNGLL":
            coordinate = pynmea2.parse(gps_data)
            coorlat = coordinate.latitude
            coorlon = coordinate.longitude
            return [coorlat,coorlon]
                
if __name__ == "__main__":
    read = ReadSerial()
    while True:
        print(read.readGas())
