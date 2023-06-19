import zmq
import json
from typing import List


def segment_weight(n1, n2, data):
    assert "length" in data, f"Expected the 'length' of segment to be known. ({n1}, {n2})"
    return float(data['length']) + float(f"0.{n1}{n2}")


class Client:
    def __init__(self, port: int = 5560, watermark: int = 8192):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.DEALER)

        self.socket.set(zmq.SNDHWM, watermark)
        self.socket.set(zmq.RCVHWM, watermark)

        self.address = f"tcp://127.0.0.1:{port}"
        self.socket.bind(self.address)

        self.poller = zmq.Poller()

    def compute(self, array: List[bytes]):
        print('Starting client request to compute')
        self.poller.register(self.socket, zmq.POLLIN | zmq.POLLOUT)

        msg_send = 0
        msg_received = 0
        msg_count = len(array)
        results = {}

        # Switch messages between sockets
        while msg_received < msg_count:
            socks = dict(self.poller.poll())

            if (socks.get(self.socket) & zmq.POLLIN) == zmq.POLLIN:
                message_id, message = self.socket.recv_multipart()

                # Decode
                message_id = int(message_id.decode())
                message = json.loads(message.decode())

                # Append
                results[message_id] = message
                msg_received += 1

            if (socks.get(self.socket) & zmq.POLLOUT) == zmq.POLLOUT:
                # Send until all messages are sent
                self.socket.send_multipart([
                    str(msg_send).encode(),
                    array[msg_send]
                ])
                msg_send += 1
                if msg_send == msg_count:
                    self.poller.modify(self.socket, zmq.POLLIN)
            print(msg_received)

        print('Ending')
        # Sort by id
        return [value for (_, value) in sorted(results.items(), key=lambda item: item[0])]
