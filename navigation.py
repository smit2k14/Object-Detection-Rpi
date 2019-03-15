'''
import requests
response = requests.get(f'https://maps.googleapis.com/maps/api/directions/json?origin=Disneyland&destination=Universal+Studios+Hollywood&travelMode=WALKING&key=AIzaSyCM45T52QtMXz-ouXC7Nqc674fwR2J3rzA')
resp_json_payload = response.json()

distance = resp_json_payload['routes'][0]['legs'][0]['distance']['text']
duration = resp_json_payload['routes'][0]['legs'][0]['distance']['duration']

steps = resp_json_payload['routes'][0]['legs'][0]['end_address']

end_addr = resp_json_payload['routes'][0]['legs'][0]['start_address']

print(distance)
'''

print("\n\n\n\n\n\n\n\n\n\n\n\n\n\n")

start_address = "Football Ground, DA-IICT, Gandhinagar"
end_address = "Reliance Chowkdi, Sargasan, Gandhinagar"

counter = 0
path = ["Head Straight", "Turn Left", "Cross Roads Ahead", "Turn Right", "Head Straight", "You have reached your destination"]

for i in range(len(path)):
	x = input()
	print(path[i])

x = input()