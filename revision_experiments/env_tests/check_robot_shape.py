"""Control-flow checks of robot_shape against the recording mock: which
PyBullet shapes are created, and that nothing else about the robot changes."""
import importlib.util
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / 'mock'))
import pybullet as p

spec = importlib.util.spec_from_file_location(
    'env_new', str(Path(__file__).resolve().parents[2] / 'envs' / 'kabaddi_env.py'))
E = importlib.util.module_from_spec(spec)
spec.loader.exec_module(E)

passed = failed = 0


def check(name, cond, detail=''):
    global passed, failed
    if cond:
        passed += 1
        print(f'  pass  {name}')
    else:
        failed += 1
        print(f'  FAIL  {name}   {detail}')


def shapes(**kw):
    """(collision shapes, visual shapes, robot bodies) logged at reset."""
    env = E.RoboKabaddiEnv(n_defenders=3, **kw)
    p.LOG.clear()
    o, _ = env.reset(seed=0)
    col = [dict(c[2]) | {'type': c[1][0]} for c in p.LOG if c[0] == 'colshape']
    vis = [dict(c[2]) | {'type': c[1][0]} for c in p.LOG if c[0] == 'visshape']
    bodies = [dict(c[2]) for c in p.LOG if c[0] == 'createMultiBody'
              and dict(c[2]).get('baseMass', 0) > 0]
    env.close()
    return col, vis, bodies, o


env = E.RoboKabaddiEnv(n_defenders=3)
check('robot shape defaults to cylinder', env.robot_shape == 'cylinder')
env.close()
try:
    E.RoboKabaddiEnv(n_defenders=3, robot_shape='sphere')
    check('unknown shape rejected', False)
except AssertionError:
    check('unknown shape rejected', True)

col, vis, bodies, o_cyl = shapes()
robots_c = [c for c in col if c['type'] == p.GEOM_CYLINDER]
check('default: four cylinder robots, radius 0.085, height 0.315',
      len(robots_c) == 4 and all(c['radius'] == 0.085 and c['height'] == 0.315
                                 for c in robots_c), robots_c)

col, vis, bodies_b, o_box = shapes(robot_shape='box')
robots_b = [c for c in col if c['type'] == p.GEOM_BOX
            and c['halfExtents'][0] == 0.085]
check('box: four 0.17 x 0.17 x 0.315 m box robots',
      len(robots_b) == 4 and all(tuple(c['halfExtents']) == (0.085, 0.085, 0.1575)
                                 for c in robots_b), robots_b)
check('box: no cylinder collision shapes left',
      not any(c['type'] == p.GEOM_CYLINDER for c in col))
check('box: visual shapes match the collision shapes',
      sum(1 for v in vis if v['type'] == p.GEOM_BOX
          and tuple(v['halfExtents']) == (0.085, 0.085, 0.1575)) == 4)
check('box: same masses and spawn positions as the cylinder',
      [(b['baseMass'], b['basePosition']) for b in bodies_b]
      == [(b['baseMass'], b['basePosition']) for b in bodies])
check('box: observation identical at reset (same width and values)',
      (o_box == o_cyl).all() and o_box.shape == (11 + 2 * 3,))
for loco in ('planar', 'legacy'):
    env = E.RoboKabaddiEnv(n_defenders=3, robot_shape='box', locomotion=loco,
                           defender_type='physical', start_layout='back_row')
    env.reset(seed=1)
    for _ in range(20):
        env.step(E.np.array([0.3, 1.0, 0.0], dtype=E.np.float32))
    env.close()
    check(f'box runs under {loco} locomotion with physical defenders', True)

print(f'\n{passed} passed, {failed} failed')
