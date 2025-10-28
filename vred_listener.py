
import socketserver
import threading
import json
import queue
import traceback

# Qt timer on VRED main thread (PySide6 in newer VRED; PySide2 in older)
try:
    from PySide6.QtCore import QTimer  # VRED 2024+
except Exception:
    from PySide2.QtCore import QTimer  # VRED ≤2023

# API v1 helper: activates a Variant Set by NAME
# Docs: vrVariants.selectVariantSet(name) activates a variant set.
from vrVariants import selectVariantSet

HOST = "0.0.0.0"      # listen on all interfaces; use "127.0.0.1" if ASR runs on same machine
PORT = 8888           # must match the ASR client
SHARED_SECRET = None  # e.g., "mysecret"; set both client and server; None to disable

# If you sometimes send (group, name) and the actual VSet name includes group,
# flip this to True and adjust _derive_vset_name() accordingly.
USE_GROUP_IN_VSET_NAME = False

# ==========================
# Globals
# ==========================
_server = None
_server_thread = None
_dispatch_timer = None
_command_queue = queue.Queue()
_last_activation = None  # last vset name (to de-duplicate rapid repeats)


def _enqueue_action(action: dict):
    """Put a validated action dict on the queue for main-thread execution."""
    _command_queue.put(action)


def _derive_vset_name(group: str, name: str) -> str:
    """
    If the client sends both group and name (activate_pair),
    derive the VSet name string to feed to selectVariantSet().
    """
    if USE_GROUP_IN_VSET_NAME and group:
     
        return f"{group}: {name}"
    return name


def _dispatch_actions():
    """Runs in the VRED main thread via QTimer. Executes queued actions safely."""
    global _last_activation
    try:
        while True:
            action = _command_queue.get_nowait()
            act = action.get("action")

            if act == "activate_vset":
                vset_name = (action.get("vset_name") or "").strip()
                if not vset_name:
                    print("[vred] Missing vset_name.")
                    continue

                # De-dupe rapid repeats
                if _last_activation == vset_name:
                    continue
                _last_activation = vset_name

                print(f"[vred] Activating Variant Set: '{vset_name}'")
                try:
                    selectVariantSet(vset_name)
                except Exception as e:
                    print(f"[vred] Error activating VSet '{vset_name}': {e}")

            elif act == "activate_pair":
                group = (action.get("group") or "").strip()
                name = (action.get("name") or "").strip()
                if not name:
                    print("[vred] Missing 'name' in activate_pair.")
                    continue

                vset_name = _derive_vset_name(group, name)

                # De-dupe rapid repeats
                if _last_activation == vset_name:
                    continue
                _last_activation = vset_name

                print(f"[vred] Pair → group='{group}', name='{name}' → VSet='{vset_name}'")
                try:
                    selectVariantSet(vset_name)
                except Exception as e:
                    print(f"[vred] Error activating VSet '{vset_name}': {e}")

            elif act == "ping":
                # NOP; ACK already returned on the socket thread
                pass

            else:
                print(f"[vred] Unknown action: {act}")

    except queue.Empty:
        # nothing to do this tick
        pass


class VREDHandler(socketserver.StreamRequestHandler):
    """
    Handles one TCP connection. We read NDJSON lines, validate, enqueue,
    and send back a small JSON ACK.
    """

    def handle(self):
        peer = self.client_address
        print(f"[vred] Connection from {peer}")
        try:
            for raw_line in self.rfile:
                try:
                    line = raw_line.decode("utf-8").strip()
                    if not line:
                        continue
                    msg = json.loads(line)
                except Exception:
                    self._send({"ok": False, "error": "invalid_json"})
                    continue

                # Optional simple auth
                if SHARED_SECRET is not None:
                    if msg.get("token") != SHARED_SECRET:
                        self._send({"ok": False, "error": "unauthorized"})
                        continue

                action = msg.get("action")

                if action == "activate_vset":
                    vset_name = (msg.get("vset_name") or "").strip()
                    if not vset_name:
                        self._send({"ok": False, "error": "missing_vset_name"})
                        continue
                    _enqueue_action({"action": "activate_vset", "vset_name": vset_name})
                    self._send({"ok": True})

                elif action == "activate_pair":
                    group = (msg.get("group") or "").strip()
                    name = (msg.get("name") or "").strip()
                    if not name:
                        self._send({"ok": False, "error": "missing_name"})
                        continue
                    _enqueue_action({"action": "activate_pair", "group": group, "name": name})
                    self._send({"ok": True})

                elif action == "ping":
                    self._send({"ok": True, "pong": True})

                else:
                    self._send({"ok": False, "error": "unknown_action"})

        except Exception as e:
            print(f"[vred] Handler error: {e}")
            traceback.print_exc()

        print(f"[vred] Disconnected {peer}")

    def _send(self, obj: dict):
        data = (json.dumps(obj) + "\n").encode("utf-8")
        self.wfile.write(data)
        self.wfile.flush()


class ThreadingTCPServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True  # kill handler threads with main program


def start_vred_listener(host: str = HOST, port: int = PORT):
    """Start the socket listener and the main-thread dispatch timer."""
    global _server, _server_thread, _dispatch_timer

    if _server is not None:
        print("[vred] Listener already running.")
        return
    _server = ThreadingTCPServer((host, port), VREDHandler)
    _server_thread = threading.Thread(target=_server.serve_forever, name="VREDListener", daemon=True)
    _server_thread.start()
    print(f"[vred] Listening on {host}:{port}")

    # Start a QTimer to poll and dispatch actions on the VRED main thread
    _dispatch_timer = QTimer()
    _dispatch_timer.setInterval(50)  # ms
    _dispatch_timer.timeout.connect(_dispatch_actions)
    _dispatch_timer.start()
    print("[vred] Dispatch timer started (50 ms)")


def stop_vred_listener():
    """Stop the socket listener and the dispatch timer."""
    global _server, _server_thread, _dispatch_timer
    try:
        if _dispatch_timer is not None:
            _dispatch_timer.stop()
            _dispatch_timer = None
        if _server is not None:
            _server.shutdown()
            _server.server_close()
            _server = None
        if _server_thread is not None:
            _server_thread.join(timeout=1.0)
            _server_thread = None
        print("[vred] Listener stopped.")
    except Exception as e:
        print(f"[vred] Error stopping listener: {e}")


# Auto-start when the script is run inside VRED
if __name__ == "__main__":
    start_vred_listener()