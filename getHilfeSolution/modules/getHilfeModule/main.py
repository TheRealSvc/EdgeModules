# Copyright (c) Stephan von Cölln Solutions
# This Edge-Device Module receices a message from the 
# enddevice and stores some info in a database  
import time
import os
import sys
import asyncio
from six.moves import input
import threading
from azure.iot.device.aio import IoTHubModuleClient
import psycopg2
from datetime import datetime



async def main():
    try:
        if not sys.version >= "3.5.3":
            raise Exception( "The sample requires python 3.5.3+. Current version of Python: %s" % sys.version )    
        print ( "IoT Hub Client for Python" )
        conn = psycopg2.connect(
            host="localhost",
            database="svcSolutionsDB",
            user="stephan",
            password="a1345"
        )
        cur = conn.cursor() 
        insertDate = datetime.now() # the datetime to be inserted 

        # The client object is used to interact with your Azure IoT hub.
        module_client = IoTHubModuleClient.create_from_edge_environment()

        # connect the client.
        await module_client.connect()

        # define behavior for receiving an input message on input1
        async def input1_listener(module_client):
            while True:
                input_message = await module_client.receive_message_on_input("input1")  # blocking call
                message_content = input_message.data 
                message_time = input_message.custom_properties["event_time"]
                message_id = input_message.message_id 

                cur.execute("INSERT INTO public.\"eventTable\" (event_id,event_message, event_date) VALUES (%s,%s,%s)", (message_id, message_content, message_time))  
                conn.commit()
                conn.close()

                print("the data in the message received on input1 was ")
                print(input_message.data)
                print("custom properties are")
                print(input_message.custom_properties)
                print("forwarding mesage to output1")
                await module_client.send_message_to_output(input_message, "output1")

        # define behavior for halting the application
        def stdin_listener():
            while True:
                try:
                    selection = input("Press Q to quit\n")
                    if selection == "Q" or selection == "q":
                        print("Quitting...")
                        break
                except:
                    time.sleep(10)

        # Schedule task for C2D Listener
        listeners = asyncio.gather(input1_listener(module_client))

        print ( "The sample is now waiting for messages. ")

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
        print ( "Unexpected error %s " % e )
        raise

if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())
    loop.close()

    # If using Python 3.7 or above, you can use following code instead:
    # asyncio.run(main())