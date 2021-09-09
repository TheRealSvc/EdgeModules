import paho.mqtt.client as mqtt #import the client1
import time 
import paho.mqtt.publish as publish #import the client1

broker_address="192.168.0.246" 

i=0 
while i<10:
        publish.single("test","hallo from the pi from within sendmqttmessage module"+str(i), hostname=broker_address,port=1884)
        i+=1
        time.sleep(10)