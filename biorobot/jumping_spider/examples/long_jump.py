"""Random actions in the long jump: ``python -m biorobot.jumping_spider.examples.long_jump --backend mjc``."""

from biorobot.jumping_spider.examples.common import parse_args, run

if __name__ == "__main__":
    run("jumping_spider/long_jump", parse_args(__doc__), color_contacts=True)
