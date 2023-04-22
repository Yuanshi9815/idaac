import json
import numpy as np

def get_empty_episode_info():
    return {
        "episode_return": [],
        "episode_length": [],
    }

training_info_example = {
    "value_loss": np.array([]),
    "policy_loss": np.array([]),
    "dist_entropy": np.array([]),
    "total_loss": np.array([]),
}

class ContextMonitor(object):
    def __init__(self, target_env_ratio, context_space, log_path):
        self.target_env_ratio = target_env_ratio
        self.context_space = context_space
        self.log_path = log_path
        self.context_str_to_id = {}
        self.context_id_to_str = {}
        # update when episode ends
        self.contextual_episode_info = [] # all envs
        self.episode_info_t_env = get_empty_episode_info() # target en
        self.episode_info_c_env = get_empty_episode_info() # contextual env
        self.episode_info_a_env = get_empty_episode_info() # all envs

        # update after each algorithm step (update)
        self.taining_info_t_env = {}
        self.taining_info_c_env = {}
        self.taining_info_a_env = {}

        self.loss_info = None

    def add_context(self, context_str):
        if context_str not in self.context_str_to_id:
            context_id = len(self.context_str_to_id)
            self.context_str_to_id[context_str] = context_id
            self.context_id_to_str[context_id] = context_str
            self.contextual_episode_info.append(get_empty_episode_info())
            with open(self.log_path + '/context_id_map.json', 'w') as f:
                f.write(json.dumps({
                    'context_str_to_id': self.context_str_to_id,
                    'context_id_to_str': self.context_id_to_str,
                }, ensure_ascii=False, indent=4))
        return self.context_str_to_id[context_str]
    
    def extent(self, contexts):
        idxs = []
        for context in contexts:
            idx = self.add_context(json.dumps(context))
            idxs.append(idx)
        return idxs
    
    def before_algo_step(self):
        for _ in range(len(self.contextual_episode_info)):
            self.contextual_episode_info = get_empty_episode_info()
        self.episode_info_a_env = get_empty_episode_info()
        self.episode_info_t_env = get_empty_episode_info()
        self.episode_info_c_env = get_empty_episode_info()

    def before_env_step(self):
        pass

    def add_episode_info(self, context_id, from_target_env,episode_info):
        self.contextual_episode_info[context_id]["episode_return"].append(episode_info["episode_return"])
        self.contextual_episode_info[context_id]["episode_length"].append(episode_info["episode_length"])
        aim_episode_info = self.episode_info_t_env if from_target_env else self.episode_info_c_env
        aim_episode_info["episode_return"].append(episode_info["episode_return"])
        aim_episode_info["episode_length"].append(episode_info["episode_length"])
        self.episode_info_a_env["episode_return"].append(episode_info["episode_return"])
        self.episode_info_a_env["episode_length"].append(episode_info["episode_length"])

    def add_step_info(self, loss_info):
        self.loss_info = loss_info

    def after_algo_step(self):
        new_loss_info = {
            key: [float(v) for v in value.tolist()] for key, value in self.loss_info.items()
        }
        with open(self.log_path + '/raw_log.json', 'a') as f:
            f.write(json.dumps({
                'loss_info': new_loss_info,
                'contextual_episode_info': self.contextual_episode_info,
            }, ensure_ascii=False))
            f.write('\n\n')
            