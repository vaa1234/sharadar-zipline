import requests

def notify_telegram(msg):

    data = {"message": msg, "silent": False}
    headers = {"content-type":"application/json", "User-Agent":"curl/7.68.0"}
    status = requests.post("http://localhost:8080/msg", json=data, headers=headers)

    return status.ok