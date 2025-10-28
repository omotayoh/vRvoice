import json
import socket

def send_activate_vset(host: str, port: int, group: str, name: str, token: str = None, timeout: float = 1.0):
    msg = {
        "action": "activate_pair",  # FIXED: use activate_pair
        "group": group,
        "name": name,
    }
    if token:
        msg["token"] = token

    data = (json.dumps(msg) + "\n").encode("utf-8")
    with socket.create_connection((host, port), timeout=timeout) as s:
        s.sendall(data)
        s.shutdown(socket.SHUT_WR)
        s.settimeout(timeout)
        ack = s.recv(4096)

    try:
        resp = json.loads(ack.decode("utf-8").strip())
    except Exception:
        resp = {"ok": False, "error": "invalid_ack"}
    return resp