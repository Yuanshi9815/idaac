from procgen.default_context import default_context_options

# Experiment on environment: Leaper
def get_default_context_options_leaper():
    return {
        **default_context_options["leaper"],**{
            # "world_dim": 13,
            # "max_road": 3,
            # "max_log": 3,
            # "max_extra_space": 2,
        }
    }

def context_decoder_leaper(context_id):
    road_num = context_id % 4
    log_num = (context_id // 4) % 4
    explicit_context = {
        "max_road": road_num,
        "min_road": road_num,
        "max_log": log_num,
        "min_log": log_num,
    }
    return {**get_default_context_options_leaper(),**explicit_context}

def context_encoder_leaper(explict_context):
    road_num = explict_context["num_road_lanes"]
    log_num = explict_context["num_log_lanes"]
    context_id = road_num + 4 * log_num
    return context_id


# Experiment on environment: Heist
def get_default_context_options_heist():
    return {
        **default_context_options["heist"],**{
        }
    }

def context_decoder_heist(context_id):
    maze_dim = (context_id % 3) * 2 + 5
    num_keys = context_id // 3
    explicit_context = {
        "max_maze_dim": maze_dim,
        "min_maze_dim": maze_dim,
        "min_keys": num_keys,
        "max_keys": num_keys
    }
    return {**get_default_context_options_heist(),**explicit_context}

def context_encoder_heist(explict_context):
    maze_dim = explict_context["maze_dim"]
    num_keys = explict_context["num_keys"]
    context_id = (maze_dim - 5) // 2 + 3 * num_keys 
    return context_id

settings = {
    "leaper": {
        "context_encoder": context_encoder_leaper,
        "context_decoder": context_decoder_leaper,
        "context_shape": (4, 4),
        "context_options": get_default_context_options_leaper(),
    },
    "heist": {
        "context_encoder": context_encoder_heist,
        "context_decoder": context_decoder_heist,
        "context_shape": (3, 4),
        "context_options": get_default_context_options_heist(),
    },
}