## create file with sample code
import torch
import pyaudio
import numpy as np
from azure.iot.device.aio import IoTHubModuleClient
from azure.iot.device import Message
import re
import random
from datetime import datetime
import asyncio
from utils import read_audiostream
import paho.mqtt.client as mqtt #import the client1
import time 
import paho.mqtt.publish as publish #import the client1
import socket


broker_address="192.168.0.246" 


async def startWakewordService():
    # The client object is used to interact with your Azure IoT Edge device.
    module_client = IoTHubModuleClient.create_from_connection_string("HostName=testhub321.azure-devices.net;DeviceId=sensoredge;SharedAccessKey=YId4cq/nZP1CYLgIx0CK3gakA/fnlSftNpJl+/i5TNA=", websockets=True)
    #module_client = IoTHubModuleClient.create_from_edge_environment() # only for testing the connections_string was used
    #await module_client.connect()

    # voice detection using onnx
    onnx = True
    # otherwise torch
    if (onnx):
        import onnx
        import onnxruntime
        onnx_model = onnx.load('./savedModels/model_de.onnx')
        onnx.checker.check_model(onnx_model)
        ort_session = onnxruntime.InferenceSession('./savedModels/model_de.onnx')
        decoder = torch.load('./savedModels/decoder_de.pth')
    else:
        device = torch.device('cpu')  # gpu also works, but our models are fast enough for CPU
        model = torch.jit.load('./savedModels/model_de.zip')
        decoder = torch.load('./savedModels/decoder_de.pth')
    #########################################################################
    CHUNK = 96000  #40960 # number of data points to read at a time
    RATE =  16000  #44100 # time resolution of the recording device (Hz)
    WAKEWORDS = ['hilfe', 'hülfe','hilf','pfleger', 'notfall', 'sonne', 'test']

    p = pyaudio.PyAudio() # start the PyAudio class
    stream = p.open(format=pyaudio.paInt16, channels=1, rate=RATE, input=True, frames_per_buffer=CHUNK)  # uses default input device

    try:
        while 1: #do it a few times just to see
            data = torch.from_numpy(np.fromstring(stream.read(CHUNK),dtype=np.int16)).float()
            data = torch.reshape(data, (1,CHUNK))
            data = read_audiostream(data, RATE, 16000) # maybe not required ? 16000 Hz required as input !

            # noinspection PyPackageRequirements
            if (not onnx):
                output = model(data)
            else:
                output = data

            for example in output:
                if(not onnx):
                    chunkoutput = decoder(example.cpu())
                else:
                    onnx_input = example.detach().cpu().numpy().reshape(1,-1)
                    ort_inputs = {'input': onnx_input}
                    ort_outs = ort_session.run(None, ort_inputs)
                    chunkoutput = decoder(torch.Tensor(ort_outs[0])[0])

                for wakeword in WAKEWORDS:
                    if re.search(wakeword, chunkoutput):
                        print('Wakeword erkannt: ' + wakeword + ' ... tue etwas !')
                        # Connect the module client.
                        print("Sending message...")
                        msg = Message("Someone called " + wakeword) # thats the payload
                        msg.message_id = str(random.randint(0,1000000))
                        msg.module = 'sendwakeword'
                        msg.custom_properties["event_time"] = str(datetime.now())
                        msg.user_id="sensoredge" ;
                        await module_client.send_message_to_output(msg, "output1")
                        print("Message successfully sent via standard routes")
                        print("now sending device to device message via mqtt routes")
                        publish.single(str(socket.gethostname()), "Someone called " + wakeword + " via MQTT", hostname=broker_address,port=1884) # this is the MQTT breakout

    except KeyboardInterrupt:
        print("Process interupted by keyboard interaction")
        msg = Message("Process interupted by keyboard interaction. Module should restart automatically") # thats the payload
        msg.message_id = str(random.randint(0,1000000))
        msg.custom_properties["event_time"] = str(datetime.now())
        await module_client.send_message_to_output(msg, "output1")
        p.terminate()
        await module_client.disconnect()

if __name__ == "__main__":
    asyncio.run(startWakewordService())