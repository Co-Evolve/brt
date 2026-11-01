"""Random actions in the platform jump: ``python -m biorobot.jumping_spider.examples.platform_jump --backend mjc``.

The spider has a dragline anchored behind the start platform in this example; its brake is the last action.
"""

from biorobot.jumping_spider.examples.common import parse_args, run
from biorobot.jumping_spider.specification import (
    default_jumping_spider_morphology_specification,
)

if __name__ == "__main__":
    run(
        "jumping_spider/platform_jump",
        parse_args(__doc__),
        morphology_specification=default_jumping_spider_morphology_specification(
            dragline=True
        ),
        camera_ids=(1, 2),
    )
