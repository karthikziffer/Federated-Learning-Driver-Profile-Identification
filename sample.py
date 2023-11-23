
import paho.mqtt.client as mqtt
import time

def on_message(client, userdata, message):
    print("message received " ,str(message.payload.decode("utf-8")))
    print("message topic=",message.topic)
    print("message qos=",message.qos)
    print("message retain flag=",message.retain)



client =mqtt.Client(client_id="client2", clean_session=True, userdata=None, transport="tcp")

client.on_message=on_message

rc = client.connect(host='localhost', keepalive=1000)

print(rc)

while True:
  pr = client.subscribe("/data")
  client.loop_forever()