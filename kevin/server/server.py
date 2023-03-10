import ipaddress
import logging
import os
import random

import flask
from flask import Flask, Response, request, Request


class FlaskServer:
    # todo make this work with models

    flask: Flask
    name: str
    running = False

    #  Key value pairs of "game id" -> {game, turn, board, you}
    games: dict = {}

    def __init__(self, name: str):
        self.name = name
        self.flask = Flask(name)

    def __bool__(self): return self.running

    def save_game(self, game: Request):
        game_id = game["game"]["id"]
        self.games[game_id] = game

    def run(self):
        #  Register methods with the flask server
        server = self.flask

        @server.get("/")
        def on_info():
            return {
                "apiversion": "1",
                "author": "Gary",
                "color": "#EE2222",
                "head": "default",
                "tail": "default"
            }

        @server.post("/start")
        def on_start():
            self.save_game(request.get_json())
            return "ok"

        @server.post("/move")
        def on_move():
            self.save_game(request.get_json())
            return {"move": random.choice(["up", "down", "left", "right"])}

        @server.post("/end")
        def on_end():
            print(self.games)
            return "ok"

        @server.after_request
        def set_headers(response: Response):
            response.headers.set("server", "Kevin")
            return response

        server.run(host="0.0.0.0", port=os.environ.get("PORT", "8000"))
        self.running = True


fs = FlaskServer("Kevin")
fs.run()
