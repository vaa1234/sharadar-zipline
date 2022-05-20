import requests

def notify_telegram(msg, recipient):

    data = {"to": recipient, "message": msg, "silent": False}
    headers = {'X-Auth-Token': 'JH9f84ghrg', "content-type":"application/json", "User-Agent":"curl/7.68.0"}
    requests.post("http://localhost:8080/msg", json=data, headers=headers)