from procgen.default_context import default_context_options
import random

def sample_bigfish_conext(hardmode=False):
    result = default_context_options['bigfish']
    if hardmode:
        result['start_r'] = .5
    else:
        start_rs = [0.5,0.75,0.65,0.73,0.82,0.91,1.0,1.1,1.2,1.31,1.42,1.53]
        result['start_r'] = random.choice(start_rs)
    return result

if __name__ == '__main__':
    print(sample_bigfish_conext())
    print(sample_bigfish_conext(hardmode=True))