import os
import json
from procgen.default_context import default_context_options


def get_all_episodic_context(env_name):
    # 读取这个py文件的绝对路径
    path = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(path, 'distribution', '{}.json'.format(env_name))
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data

def get_context_options(env_name, episodic_context):
    result = default_context_options[env_name]
    if env_name == 'bossfight':
        result = {
            **result,
            "max_rounds_num": episodic_context['num_rounds'],
            "min_rounds_num": episodic_context['num_rounds'],
            "max_round_health": episodic_context['round_health'],
            "min_round_health": episodic_context['round_health'],
            "max_barriers_num": episodic_context['num_barriers'],
            "min_barriers_num": episodic_context['num_barriers'],
        }
    elif env_name == 'climber':
        result = {
            **result,
            "max_platforms": episodic_context['num_platforms'],
            "min_platforms": episodic_context['num_platforms'],
        }
    elif env_name == 'coinrun':
        # difficulty, num_sections
        result = {
            **result,
            "max_difficulty": episodic_context['difficulty'],
            "min_difficulty": episodic_context['difficulty'],
            "max_section_num": episodic_context['num_sections'],
            "min_section_num": episodic_context['num_sections'],
        }
    elif env_name == 'dodgeball':
        result = {
            **result,
            "base_num_enemies": episodic_context['num_enemies'],
            "max_extra_enemies": 0,
            "allow_left_exit": episodic_context['exit_wall_id'] == 0,
            "allow_right_exit": episodic_context['exit_wall_id'] == 1,
            "allow_top_exit": episodic_context['exit_wall_id'] == 2,
            "allow_bottom_exit": episodic_context['exit_wall_id'] == 3,
        }
    elif env_name == 'fruitbot':
        result = {
            **result,
            "max_num_fruits": episodic_context['num_good'],
            "min_num_fruits": episodic_context['num_good'],
            "min_foods": episodic_context['num_bad'],
            "max_foods": episodic_context['num_bad'],
        }
    elif env_name == 'heist':
        result = {
            **result,
            "max_maze_dim": episodic_context['maze_dim'],
            "min_maze_dim": episodic_context['maze_dim'],
            "max_keys": episodic_context['num_keys'],
            "min_keys": episodic_context['num_keys'],
        }
    elif env_name == 'leaper':
        result = {
            "max_log": episodic_context['num_log_lanes'],
            "min_log": episodic_context['num_log_lanes'],
            "max_road": episodic_context['num_road_lanes'],
            "min_road": episodic_context['num_road_lanes'],
        }
    elif env_name == 'maze':
        result = {
            **result,
            "max_maze_dim": episodic_context['maze_dim'],
            "min_maze_dim": episodic_context['maze_dim'],
        }
    elif env_name == 'ninja':
        result = {
            **result,
            "max_difficulty": episodic_context['difficulty'],
            "min_difficulty": episodic_context['difficulty'],
            "max_num_sections": episodic_context['num_sections'],
            "min_num_sections": episodic_context['num_sections'],
        }



    return result



if __name__ == '__main__':
    env_name = 'coinrun'
    print(get_context_options(env_name, list(get_all_episodic_context(env_name).values())[0]['context']))