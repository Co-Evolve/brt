from moojoco.mjcf.component import MJCFSubComponent


class MJCFTarget(MJCFSubComponent):
    def _build(self):
        self._target = self.mjcf_body.add(
            "site",
            name="target_site",
            type="sphere",
            size=[0.2],
            rgba=[1.0, 0.0, 0.0, 1.0],
        )
