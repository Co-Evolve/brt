from biorobot.jumping_spider.mjcf.arena.platform_jump import PlatformJumpArenaConfiguration, MJCFPlatformJumpArena
from biorobot.jumping_spider.mjcf.morphology.morphology import MJCFJumpingSpiderMorphology
from biorobot.jumping_spider.mjcf.morphology.specification.default import default_jumping_spider_specification

if __name__ == '__main__':
    # arena_configuration = LongJumpArenaConfiguration(track_size=(10, 3))
    # jump_arena = MJCFLongJumpArena(configuration=arena_configuration)

    arena_configuration = PlatformJumpArenaConfiguration()
    jump_arena = MJCFPlatformJumpArena(configuration=arena_configuration)

    morphology_specification = default_jumping_spider_specification()
    morphology = MJCFJumpingSpiderMorphology(specification=morphology_specification)

    jump_arena.attach(
        other=morphology, free_joint=True, position=[0, 0, 1]
    )

    jump_arena.export_to_xml_with_assets("./mjcf")
