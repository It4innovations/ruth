import dataclasses
import logging
import time

import zmq
import json
from typing import Any, List


logger = logging.getLogger(__name__)


@dataclasses.dataclass
class Message:
    kind: str
    data: Any


def segment_weight(n1, n2, data):
    assert "length" in data, f"Expected the 'length' of segment to be known. ({n1}, {n2})"
    return float(data['length']) + float(f"0.{n1}{n2}")


class Client:
    def __init__(self, port: int = 5560, broadcast_port: int = 5561, watermark: int = 8192):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.DEALER)

        self.socket.set(zmq.SNDHWM, watermark)
        self.socket.set(zmq.RCVHWM, watermark)

        self.address = f"tcp://*:{port}"
        self.socket.bind(self.address)

        # Create broadcast socket
        self.broadcast_socket = self.context.socket(zmq.PUB)
        self.broadcast_address = f"tcp://*:{broadcast_port}"
        self.broadcast_socket.bind(self.broadcast_address)

        self.poller = zmq.Poller()

        # Give subscribers a chance to join
        time.sleep(1)

    def compute(self, messages: List[Message], timeout_s=10) -> List[Any]:
        self.poller.register(self.socket, zmq.POLLIN | zmq.POLLOUT)

        msg_send = 0
        msg_received = 0
        msg_count = len(messages)
        results = {}

        logger.debug(f"Sending {len(messages)} message(s) to workers")

        total_size = 0

        # Switch messages between sockets
        while msg_received < msg_count:
            result = self.poller.poll(timeout=timeout_s * 1000)
            if not result:
                raise Exception(f"ZeroMQ communication has timed out after {timeout_s} second(s)")
            socks = dict(result)

            if (socks.get(self.socket) & zmq.POLLIN) == zmq.POLLIN:
                message_id, status, payload = self.socket.recv_multipart()

                # Decode
                message_id = int(message_id.decode())

                if status != b"ok":
                    raise Exception(f"Invalid response to {messages[message_id].kind}: {status}")

                payload = json.loads(payload.decode())

                # Append
                results[message_id] = payload
                msg_received += 1

            if (socks.get(self.socket) & zmq.POLLOUT) == zmq.POLLOUT:
                message = messages[msg_send]
                payload = json.dumps(message.data).encode()
                total_size += len(payload)

                # Send until all messages are sent
                self.socket.send_multipart([
                    str(msg_send).encode(),
                    message.kind.encode(),
                    payload
                ])
                msg_send += 1
                if msg_send == msg_count:
                    self.poller.modify(self.socket, zmq.POLLIN)

        logger.debug(f"Sent {total_size} bytes")
        # Sort by id
        return [value for (_, value) in sorted(results.items(), key=lambda item: item[0])]

    def broadcast(self, message: Message):
        """
        Broadcasts given message to all currently connected workers.
        """
        func_call = f"traffic-sim:{message.kind}"

        payload = json.dumps(message.data).encode()
        logger.debug(f"Broadcasting {func_call}, {len(payload)} bytes")
        self.broadcast_socket.send_multipart([
            str(func_call).encode(),
            payload
        ])
