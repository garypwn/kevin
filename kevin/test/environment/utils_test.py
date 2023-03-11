from kevin.src.environment.utils import observation_from_dict as utils


def observation_from_empty_test():
    game = {
        "game": None,
        "turn": 0,
        "board": {
            "height": 11,
            "width": 11,
            "food": None,
            "hazards": None,
            "snakes": None,
        },
        "you": {
            "id": "snake_1"
        }
    }


print(utils.observation_from_dict({
    "game": None,
    "turn": 0,
    "board": {
        "height": 11,
        "width": 11,
        "food": None,
        "hazards": None,
        "snakes": None,
    },
    "you": {
        "id": "snake_1"
    }
}
    , 4))
