# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for
# full license information.

import time
import os
import sys
import asyncio
from six.moves import input
import threading
from azure.iot.device.aio import IoTHubModuleClient
import paho.mqtt.client as mqtt #import the client1
from sqlalchemy import create_engine, select,Column, Integer, String, Date
from sqlalchemy.orm import declarative_base, sessionmaker
import datetime
try:
    from zoneinfo import ZoneInfo
except ImportError:
    from backports.zoneinfo import ZoneInfo

Base = declarative_base()

# sqlAlchemy table model 
class Maintables(Base):
    __tablename__ = 'Maintables'
    id=Column(Integer, primary_key=True, autoincrement=True)
    device=Column('device',String(64))
    topic = Column('topic',String(255))
    timestamp = Column('timestamp',Date)
    message = Column('message', String(1000))
    createdAt = Column('createdAt',Date)
    updatedAt = Column('updatedAt',Date)


async def taskAzure(): 
    try:
        if not sys.version >= "3.5.3":
            raise Exception( "Azure: The sample requires python 3.5.3+. Current version of Python: %s" % sys.version )
        print ( "Azure: IoT Hub Client for Python" )

        # The client object is used to interact with your Azure IoT hub.
        module_client = IoTHubModuleClient.create_from_connection_string("HostName=testhub321.azure-devices.net;DeviceId=maindevice;SharedAccessKey=1Hl5MNfsC9JW+eBKdthoDx0aI5yIB7/f06Aj4JGuhJc=", websockets=True)

        # connect the client.
        await module_client.connect()

        # define behavior for receiving an input message on input1
        async def input1_listener(module_client):
            while True:
                input_message = await module_client.on_message_received("input1")  # blocking call
                print("Azure: the data in the message received on input1 was ")
                print(input_message.data)
                print("Azure: custom properties are")
                print(input_message.custom_properties)
                print("Azure: forwarding mesage to output1")
                await module_client.send_message_to_output(input_message, "output1")

        # define behavior for halting the application
        def stdin_listener():
            while True:
                try:
                    selection = input("Azure: Press Q to quit\n")
                    if selection == "Q" or selection == "q":
                        print("Quitting...")
                        break
                except:
                    time.sleep(20)

        # Schedule task for C2D Listener
        listeners = asyncio.gather(input1_listener(module_client))

        print ( "Azure: The sample is now waiting for messages. ")

        # Run the stdin listener in the event loop
        loop = asyncio.get_event_loop()
        user_finished = loop.run_in_executor(None, stdin_listener)

        # Wait for user to indicate they are done listening for messages
        await user_finished

        # Cancel listening
        listeners.cancel()

        # Finally, disconnect
        await module_client.disconnect()

    except Exception as e:
        print ( "Azure: Unexpected error in azure iot %s " % e )
        raise


async def main():      

    def on_message(client, userdata, message):
        engine = create_engine('postgresql://postgres:a1345@192.168.0.246:5432/svcsolutions')
        print("before sqlalchemy session creation")
        Session = sessionmaker(bind=engine)
        print("in between session creation")
        session = Session()
        print("after session creation")
        print("MQTT message that was received ", str(message.payload.decode("utf-8")))
        print("MQTT message topic=", message.topic)
        print("MQTT message qos=", message.qos)
        print("MQTT message retain flag=", message.retain)
        # Watch out below: topic and device can be confused !!! MQTT-topic = $device and Database-topic = "wakeword" 
        newRow = Maintables( device= message.topic, topic= "wakeword", timestamp=str(datetime.datetime.now()), message=str(message.payload.decode("utf-8")), createdAt=str(datetime.datetime.now(ZoneInfo("Europe/Amsterdam"))), updatedAt=str(datetime.datetime.now()))
        session.add(newRow)
        print('Postgres: before  commit')
        session.commit()
        print('Postgres: after commit')
        session.close()    
   
    def on_connect(client,userdata, flags, rc):
        print("in mqtt on_connect callback fun")
        client.subscribe("#")  

    broker_address="192.168.0.246" # parametrize that !!! e.g. read from host config.toml
    print("MQTT: creating new instance")
    client = mqtt.Client("P1") #create new instance
    client.on_message=on_message #attach function to callback
    client.on_connect=on_connect
    print("MQTT: connecting to broker")
    client.connect(broker_address, 1884) #connect to broker
    print("MQTT: starting loop")
    client.loop_forever()


#async def main():
#    tasks = asyncio.run(taskAzure()) # , taskMQTT()) 
#    await asyncio.sleep(10)

if __name__ == "__main__":
    asyncio.run(main())# , taskMQTT()) 