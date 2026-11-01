/**
 * Seahorse tail demo controller: open-loop muscle contraction set from the page's sliders.
 *
 * The default `seahorse/tail` model has a pair of force-controlled ventral HM beams that span the whole tail (one
 * dextral, one sinistral) and position-controlled MVM beams between every two neighbouring segments. As in
 * `biorobot/seahorse/examples/common.py` (`contraction`), a contraction level in [0, 1] maps every actuator from the
 * upper end of its ctrlrange (relaxed) to the lower end (fully contracted): a larger pulling force for force-controlled
 * beams, a shorter target length for position-controlled ones.
 *
 * Gravity is on (the `seahorse/tail` default), so the hanging tail straightens again when its
 * muscles relax. The HM beams curl the tail more the harder they pull (at the default 0.4 of their maximum force, by
 * ~90 degrees in total); the MVM beams, which pull in a fraction of their rest length, curl it fully from ~0.4.
 *
 * The page writes the levels into `settings`; with `settings.cycle`, every level is multiplied by a smooth
 * contract/relax cycle, (1 - cos(2 pi t / period)) / 2. Actuators are matched by name, so other architectures (more HM
 * beams, no MVM beams, ...) work too: every HM beam on a coronal side follows that side's slider.
 */
export const settings = {
  hmDextral: 0.4, // contraction level of the HM beams with "dextral" in their name
  hmSinistral: 0.4, // ... and of those with "sinistral"
  mvm: 0, // contraction level of all MVM beams
  cycle: true, // modulate all levels with a contract/relax cycle
  period: 4, // s, period of the cycle
};

const HM_PATTERN = /\/hm_beam_.*(dextral|sinistral)/;
const MVM_PATTERN = /\/mvm_beam_/;

let groups = null;

/** For every actuator: which setting drives it (or null) and its relaxed/contracted controls. */
export function init(model, data, actuatorNames) {
  const range = model.actuator_ctrlrange;
  groups = actuatorNames.map((name, i) => {
    const hm = HM_PATTERN.exec(name);
    const key = hm ? (hm[1] === "dextral" ? "hmDextral" : "hmSinistral") : MVM_PATTERN.test(name) ? "mvm" : null;
    return { key, relaxed: range[2 * i + 1], contracted: range[2 * i] };
  });
}

export function control(t, model, data, actuatorNames) {
  if (!groups) init(model, data, actuatorNames);
  const modulation = settings.cycle ? (1 - Math.cos((2 * Math.PI * t) / settings.period)) / 2 : 1;
  return Float64Array.from(groups, ({ key, relaxed, contracted }) => {
    const level = key ? modulation * settings[key] : 0;
    return relaxed + level * (contracted - relaxed);
  });
}
