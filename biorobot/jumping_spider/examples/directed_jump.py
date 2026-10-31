"""Random actions in the directed jump: ``python -m biorobot.jumping_spider.examples.directed_jump --backend mjc``."""

import numpy as np

from biorobot.jumping_spider.examples.common import parse_args, run

if __name__ == "__main__":
    run(
        "jumping_spider/directed_jump",
        parse_args(__doc__),
        target_azimuth_range=(-np.pi / 4, np.pi / 4),
        camera_ids=(0, 1),
    )
