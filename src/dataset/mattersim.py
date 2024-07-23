import sys
sys.path.append('./data/build')
import MatterSim
import math


class ACTION:
    STOP    = (0,  0,  0)
    RIGHT   = (0,  1,  0)  # turn right 30°
    LEFT    = (0, -1,  0)  # turn left 30°
    UP      = (0,  0,  1)  # look up 30°
    DOWN    = (0,  0, -1)  # look down 30°
    MOVE    = lambda i: (i, 0, 0)  # move to i-th navigable viewpoint


class MatterSimEnv:
    '''
    A wrapper of MatterSim environments
    using discretized viewpoints
    '''
    def __init__(self, render=False, img_size=(640, 480), vfov=60, discrete_view=True):
        self.image_w = img_size[0]
        self.image_h = img_size[1]
        self.vfov = vfov

        self.sim = MatterSim.Simulator()
        self.sim.setRenderingEnabled(render)
        self.sim.setDiscretizedViewingAngles(discrete_view)
        self.sim.setBatchSize(1)
        self.sim.setCameraResolution(self.image_w, self.image_h)
        self.sim.setCameraVFOV(math.radians(self.vfov))
        self.sim.initialize()

    def new_episode(self, scan_id, viewpoint_id, heading, elevation=0):
        self.sim.newEpisode([scan_id], [viewpoint_id], [heading], [elevation])

    def get_state(self):
        """
        Get feature and state.

        Returns:
            state
        """
        state = self.sim.getState()[0]
        return state

    def step(self, action):
        """
        Args:
            actions: (index, heading, elevation)
        """
        indices = [int(action[0])]
        headings = [float(action[1])]
        elevations = [float(action[2])]
        self.sim.makeAction(indices, headings, elevations)


class MatterSimEnvBatch:
    def __init__(self, batch_size, render=False, img_size=(640, 480), vfov=60):
        self.image_w = img_size[0]
        self.image_h = img_size[1]
        self.vfov = vfov

        self.batch_size = batch_size
        self.sim = MatterSim.Simulator()
        self.sim.setRenderingEnabled(render)
        self.sim.setBatchSize(self.batch_size)
        if render:
            self.sim.setPreloadingEnabled(True)
            self.sim.setCacheSize(2 * batch_size)
        self.sim.setDiscretizedViewingAngles(True)
        self.sim.setCameraResolution(self.image_w, self.image_h)
        self.sim.setCameraVFOV(math.radians(self.vfov))
        self.sim.initialize()

    def new_episodes(self, scan_ids, viewpoint_ids, headings, elevations=None):
        """
        elevations will be zeros if None provided.
        """
        if elevations is None:
            elevations = [0] * self.batch_size
        self.sim.newEpisode(scan_ids, viewpoint_ids, headings, elevations)

    def get_states(self):
        """
        Get a list of features and states.

        Returns:
            features, states
            features is None if feature_path is None.
        """
        states = self.sim.getState()
        return states

    def step(self, actions):
        """
        Args:
            actions: an sequence(iterable) of (index, heading, elevation)
        """
        indices = []
        headings = []
        elevations = []
        for index, heading, elevation in actions:
            indices.append(int(index))
            headings.append(float(heading))
            elevations.append(float(elevation))
        self.sim.makeAction(indices, headings, elevations)
