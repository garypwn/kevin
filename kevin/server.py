import ipaddress
import logging
import os

import flask
from flask import Flask, Response

from kevin.environment.snake_engine import SnakeEngine, SnakeGameState


class FlaskServer(SnakeEngine):

    server: Flask
    name: str
    running = False

    def __init__(self, name: str):
        self.name = name

    def __bool__(self): return self.running

    def run(self):
        server = self.server

        @server.get("/")
        def on_info():
            pass

        @server.post("/start")
        def on_start():
            pass

        @server.post("/move")
        def on_move():
            pass

        @server.post("/end")
        def on_end():
            pass

        @server.after_request
        def set_headers(response: Response):
            response.headers.set("server", "Kevin")
            return response

        server.run(host="0.0.0.0", port=os.environ.get("PORT", "8000"))

    def board(self) -> SnakeGameState:
        pass

    def submit_move(self, snake_id):
        pass

