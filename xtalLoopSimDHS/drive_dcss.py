"""Drive the goniometer through dcss without BluIce.

Connects to dcss's GUI port as a minimal client, takes master, and sends the
three commands the DHS must answer: a spindle move, a stage move and a shuttered
oscillation. Prints what dcss broadcasts back and the camera server's pose
before and after each command, so one run shows the whole chain
dcss -> xtalLoopSimDHS -> camera server.

    .venv/bin/python drive_dcss.py [--dcss localhost:14243] [--camera http://localhost:8081]

Wire detail this depends on: dcss reads the login line as a 200-byte
space-padded protocol-1 frame and speaks protocol 2 (12-char text length,
13-char binary length, a space, the text) for everything after it. Taking
master steals it from any BluIce that is connected.
"""
import argparse
import getpass
import json
import socket
import time
import urllib.request

KEEP = ("gonio_phi", "sample_x", "video_trigger", "stog_log error")
NOISE = ("configure", "system_message", "user_message", "base_units")


def send(sock: socket.socket, text: str) -> None:
    body = text.encode()
    sock.sendall(f"{len(body):12d}{0:13d} ".encode() + body)


def recv(sock: socket.socket, timeout: float) -> str | None:
    sock.settimeout(timeout)
    try:
        hdr = b""
        while len(hdr) < 26:
            chunk = sock.recv(26 - len(hdr))
            if not chunk:
                return None
            hdr += chunk
        n_text, n_bin = int(hdr[:12]), int(hdr[12:25])
        data = b""
        while len(data) < n_text + n_bin:
            chunk = sock.recv(n_text + n_bin - len(data))
            if not chunk:
                return None
            data += chunk
        return data[:n_text].rstrip(b"\x00\n ").decode(errors="replace")
    except socket.timeout:
        return None


def pose(camera: str) -> str:
    doc = json.loads(urllib.request.urlopen(camera + "/status", timeout=5).read())
    p = doc["positions"]
    return f"rotx={p['rotx']:.3f} tx={p['tx']:.3f} moving={doc['moving']}"


def listen(sock: socket.socket, seconds: float) -> None:
    end = time.time() + seconds
    while time.time() < end:
        msg = recv(sock, 0.5)
        if msg and any(k in msg for k in KEEP) and not any(k in msg for k in NOISE):
            print("   dcss ->", msg[:120])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dcss", default="localhost:14243", help="dcss GUI port")
    ap.add_argument("--camera", default="http://localhost:8081", help="camera server")
    ap.add_argument("--user", default=getpass.getuser(),
                    help="login name sent in gtos_client_is_gui (default: the current user)")
    args = ap.parse_args()
    host, port = args.dcss.rsplit(":", 1)

    sock = socket.create_connection((host, int(port)))
    login = f"gtos_client_is_gui {args.user} DRIVE-DCSS-SESSION localhost :0"
    sock.sendall(login.encode().ljust(200, b" "))
    print("dcss:", recv(sock, 5))
    send(sock, "gtos_become_master force")
    listen(sock, 3.0)

    for cmd, wait in (("gtos_start_motor_move gonio_phi 90", 4),
                      ("gtos_start_motor_move sample_x 0.2", 3),
                      ("gtos_start_oscillation gonio_phi video_trigger 30 2", 5)):
        print(f"\n>>> {cmd}   [camera before: {pose(args.camera)}]")
        send(sock, cmd)
        listen(sock, wait)
        print(f"    camera after: {pose(args.camera)}")
    sock.close()


if __name__ == "__main__":
    main()
