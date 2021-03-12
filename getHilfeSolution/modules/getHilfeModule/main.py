# Copyright (c) Stephan von Cölln Solutions
# This Edge-Device Module receives a message from the 
# enddevice and stores some info in a database  
import time
import os
import sys
import asyncio
from six.moves import input
import threading
import psycopg2
from datetime import datetime
from azure.iot.device.aio import IoTHubModuleClient

async def main():
    conn_str = os.getenv("IOTHUB_DEVICE_CONNECTION_STRING")
    # path to the root ca cert used on your iot edge device (must copy the pem file to this downstream device)
    # example:   /home/azureuser/edge_certs/azure-iot-test-only.root.ca.cert.pem
    ca_cert = os.getenv("IOTEDGE_ROOT_CA_CERT_PATH")


    # The client object is used to interact with your Azure IoT hub.
    try:
        if not sys.version >= "3.5.3":
            raise Exception( "The sample requires python 3.5.3+. Current version of Python: %s" % sys.version )    
        print ( "IoT Hub Client for Python" )
        #conn = psycopg2.connect(
        #host="192.168.0.246", 
        #database="svcSolutionsDB",
        #user="stephan",
        #password="a1345"
        #)
        #print(conn)
        #cur = conn.cursor() 
        #insertDate = datetime.now() # the datetime to be inserted
        # The client object is used to interact with your Azure IoT hub.
        module_client = IoTHubModuleClient.create_from_connection_string('HostName=testhub321.azure-devices.net;DeviceId=maindevice;SharedAccessKey=H+27aF7QAmHVpJyx0mV+LW1x5CkmTCk7pltL/a+Pjfw=') 
        print("hier 1")
        # connect the client.
        await module_client.connect()
        print("hier 2")
        # define behavior for receiving an input message on input1
        async def input1_listener(module_client):
            while True:
                print("hier 3")
                input_message = await module_client.receive_message_on_input("input1")  # blocking call
                print("hier 4")
                message_content = input_message.data 
                #message_time = input_message.custom_properties["event_time"]
                message_id = input_message.message_id 
                #cur.execute("INSERT INTO public.\"eventTable\" (event_id,event_message, event_date) VALUES (%s,%s,%s)", (message_id, message_content, message_time))  
                #conn.commit()
                print("the data in the message received on input1 was")
                print(input_message.data)
                print("custom properties are")
                print(input_message.custom_properties)
                print("forwarding message to output1")
                await module_client.send_message_to_output(input_message, "output1")            
                print("hier 5 - message sent")

        # define behavior for halting the application
        def stdin_listener():
            while True:
                try:
                    selection = input("Press Q to quit\n")
                    if selection == "Q" or selection == "q":
                        print("Quitting...")
                        break
                except:
                    time.sleep(60)
    	
       
        # Schedule task for C2D Listener
        listeners = asyncio.gather(input1_listener(module_client))
        print ( "The sample is now waiting for messages.")

         # Run the stdin listener in the event loop
        loop = asyncio.get_running_loop()
        user_finished = loop.run_in_executor(None, stdin_listener)

        # Wait for user to indicate they are done listening for messages
        await user_finished

        # Cancel listening
        listeners.cancel()

        # Finally, disconnect
        await module_client.disconnect()

        # Finally, disconnect
        #await module_client.disconnect()
        #conn.close()    
    
    except Exception as e:
        print ( "Unexpected error %s " % e )
        raise

if __name__ == "__main__":
    asyncio.run(main())