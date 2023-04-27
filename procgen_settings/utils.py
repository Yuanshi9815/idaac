import os
import json
import random
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
            **result,
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

def get_context_setting(env_name, context_setting_id):
    path = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(path, 'mask.json'.format(env_name))
    with open(json_path) as f:
        data = json.load(f)
    return data[env_name][context_setting_id]

distribution_cache = {}
flip_distribution_cache = {}

def sample_a_conext(env_name, context_setting_id, flip=False):
    if env_name not in distribution_cache:
        distribution_cache[env_name] = {}
        flip_distribution_cache[env_name] = {}
    if context_setting_id not in distribution_cache[env_name]:
        distribution_cache[env_name][context_setting_id] = {}
        flip_distribution_cache[env_name][context_setting_id] = {}
        masked_contexts = get_context_setting(env_name, context_setting_id)['masked_contexts']
        all_prob = 0
        flip_all_prob = 0
        for context_id, context in get_all_episodic_context(env_name).items():
            if context_id in masked_contexts:
                flip_all_prob += context['prob']
            else:
                all_prob += context['prob']
        for context_id, context in get_all_episodic_context(env_name).items():
            if context_id in masked_contexts:
                flip_distribution_cache[env_name][context_setting_id][context_id] = context['prob'] / flip_all_prob
            else:
                distribution_cache[env_name][context_setting_id][context_id] = context['prob'] / all_prob
    random_num = random.random()
    random_id = None
    target_distribution = flip_distribution_cache[env_name][context_setting_id] if flip else distribution_cache[env_name][context_setting_id]
    for context_id, prob in target_distribution.items():
        if random_num < prob:
            random_id = context_id
            break
        random_num -= prob
    if random_id is None:
        random_id = list(target_distribution.keys())[-1]
    return get_context_options(env_name, get_all_episodic_context(env_name)[random_id]['context'])

if __name__ == '__main__':
    env_name = 'leaper'
    print(get_context_options(env_name, list(get_all_episodic_context(env_name).values())[0]['context']))
    print(get_context_setting(env_name, "1"))
    print(sample_a_conext(env_name, "3"))