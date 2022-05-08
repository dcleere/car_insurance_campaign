# script to test main.py - produces prediction based on new data passed to the model

import requests

BASE = 'http//127.0.0.1:5000/predictions'

data =
     {"data":{
       "Age": 25,
       "Job": "admin.",
       "Marital": "single",
       "Education": "secondary",
       "Default": 0,
       "Balance": 1,
       "HHInsurance": 1,
       "CarLoan": 1,
       "Communication": "NA",
       "LastContactDay": 12,
       "LastContactMonth": "may",
       "NoOfContacts": 12,
       "DaysPassed": -1,
       "PrevAttempts": 0,
       "Outcome": "NA",
       "CallStart": "17:17:42",
       "CallEnd": "17:18:06"
     }}

response = request.post(BASE, data = data)
print(response.json())