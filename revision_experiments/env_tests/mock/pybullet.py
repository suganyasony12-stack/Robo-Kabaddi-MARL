"""Recording stand-in for PyBullet. Deterministic, frictionless integrator —
it checks control flow and API usage, NOT physics."""
import math
GUI, DIRECT, GEOM_BOX, GEOM_CYLINDER = 1, 2, 3, 4
LINK_FRAME, WORLD_FRAME = 1, 2
LOG = []
_b = {}
_n = [0]
_dt = [1.0 / 240.0]

def _r(x):
    if isinstance(x, float): return round(x, 12)
    if isinstance(x, (list, tuple)): return tuple(_r(v) for v in x)
    if isinstance(x, dict): return tuple(sorted((k, _r(v)) for k, v in x.items()))
    return x

def _log(name, a, k): LOG.append((name, _r(a), _r(k)))

def connect(*a, **k): _log('connect', a, k); _b.clear(); _n[0] = 0; _shape[0] = 100; return 0
def disconnect(*a, **k): _log('disconnect', a, k)
def resetDebugVisualizerCamera(*a, **k): _log('camera', a, k)
def setAdditionalSearchPath(*a, **k): _log('searchpath', a, k)
def resetSimulation(*a, **k): _log('resetSimulation', a, k); _b.clear(); _n[0] = 0; _shape[0] = 100
def setGravity(*a, **k): _log('setGravity', a, k)
def setTimeStep(dt, **k): _log('setTimeStep', (dt,), k); _dt[0] = dt
def _new(pos, mass):
    i = _n[0]; _n[0] += 1
    _b[i] = dict(pos=list(pos), yaw=0.0, lin=[0.0]*3, ang=[0.0]*3, mass=mass,
                 force=[0.0]*3)
    return i
def loadURDF(*a, **k): _log('loadURDF', a, k); return _new([0, 0, 0], 0)
_shape = [100]
def createCollisionShape(*a, **k): _log('colshape', a, k); _shape[0] += 1; return _shape[0]
def createVisualShape(*a, **k): _log('visshape', a, k); _shape[0] += 1; return _shape[0]
def createMultiBody(**k):
    _log('createMultiBody', (), k)
    return _new(k.get('basePosition', [0, 0, 0]), k.get('baseMass', 0))
def changeDynamics(*a, **k): _log('changeDynamics', a, k)
def resetBaseVelocity(bid, linearVelocity=None, angularVelocity=None, **k):
    _log('resetBaseVelocity', (bid, linearVelocity, angularVelocity), k)
    if linearVelocity is not None: _b[bid]['lin'] = list(linearVelocity)
    if angularVelocity is not None: _b[bid]['ang'] = list(angularVelocity)
def stepSimulation(*a, **k):
    _log('stepSimulation', a, k)
    for b in _b.values():
        if b['mass'] > 0:
            for i in range(3):
                b['lin'][i] += b['force'][i] / b['mass'] * _dt[0]
                b['pos'][i] += b['lin'][i] * _dt[0]
            b['force'] = [0.0] * 3      # forces last one step, as in PyBullet
            b['yaw'] += b['ang'][2] * _dt[0]
def getQuaternionFromEuler(e): y = e[2]; return (0.0, 0.0, math.sin(y/2), math.cos(y/2))
def getEulerFromQuaternion(q): return (0.0, 0.0, 2*math.atan2(q[2], q[3]))
def getBasePositionAndOrientation(bid, **k):
    _log('getPose', (bid,), k)
    b = _b[bid]; return tuple(b['pos']), getQuaternionFromEuler([0, 0, b['yaw']])
def resetBasePositionAndOrientation(bid, pos, orn, **k):
    _log('resetPose', (bid, pos, orn), k)
    _b[bid]['pos'] = list(pos); _b[bid]['yaw'] = getEulerFromQuaternion(orn)[2]
def getBaseVelocity(bid, **k):
    _log('getVel', (bid,), k); return tuple(_b[bid]['lin']), tuple(_b[bid]['ang'])
def getContactPoints(*a, **k): _log('contacts', a, k); return []
def applyExternalForce(bid, link, force, pos, flags, **k):
    _log('applyExternalForce', (bid, link, force, pos, flags), k)
    assert flags == WORLD_FRAME and link == -1
    _b[bid]['force'] = [_b[bid]['force'][i] + force[i] for i in range(3)]
COV_ENABLE_GUI = 1
_dbg = [0]
def configureDebugVisualizer(*a, **k): _log('configureDebugVisualizer', a, k)
def addUserDebugText(text, pos, **k):
    _log('addUserDebugText', (text, pos), k)
    if 'replaceItemUniqueId' in k: return k['replaceItemUniqueId']
    _dbg[0] += 1; return _dbg[0]
