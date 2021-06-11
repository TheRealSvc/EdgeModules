#!/bin/bash

source /virtenv/bin/activate
python3 -u main.py 


{
    "Devices": [
        {
            "PathOnHost": "/dev/snd",
            "PathInContainer": "/dev/snd",
            "CgroupPermissions": "rwm"
        }
    ]
}