import math
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm

import sys
sys.path.append('src')
from dataset import MatterSimEnv
import utils


class Foo:
    def __init__(self):
        self.env = MatterSimEnv(render=False)

    def get_candidates(self, scan_id, viewpoint_id):
        """
        Args:
            scan_id: str
            viewpoint_id: str
        Returns:
            DataFrame
        """
        candidates = dict()
        for i in range(36):
            if i == 0:
                self.env.new_episode(scan_id, viewpoint_id, 0, math.radians(-30))
            elif i % 12 == 0:
                self.env.step((0, 1, 1))  # turn right and look up
            else:
                self.env.step((0, 1, 0))  # turn right
            
            # get state
            state = self.env.get_state()

            # add candidate
            for j, viewpoint in enumerate(state.navigableLocations[1:]):
                viewpoint_id  = viewpoint.viewpointId
                rel_heading   = viewpoint.rel_heading
                rel_elevation = viewpoint.rel_elevation

                # viewpoint distance to the center of current view
                angle_distance = np.linalg.norm([rel_heading, rel_elevation])

                if viewpoint_id not in candidates or angle_distance < candidates[viewpoint_id]['angle_distance']:
                    candidate = {
                        'viewpoint_id': viewpoint_id,
                        'rgb': None,
                        'heading': rel_heading + state.heading,
                        'elevation': rel_elevation + state.elevation,
                        'view_index': i,
                        'navigable_index': j + 1,
                        'angle_distance': angle_distance,
                        'distance': viewpoint.rel_distance,
                        'xyz': np.array([viewpoint.x, viewpoint.y, viewpoint.z]),
                    }
                    candidates[viewpoint_id] = candidate
        candidates = pd.DataFrame(candidates.values())
        return candidates

    def all_candidates(self):
        with open('./connectivity/scans.txt') as f:
            scans = f.readlines()

        candidate_buffer = dict()
        for i, scan_id in enumerate(scans):
            scan_id = scan_id.strip()
            print(i+1, scan_id)

            graph = utils.load_graph(scan_id)
            for viewpoint_id in tqdm(graph.nodes):
                long_id = '%s_%s' % (scan_id, viewpoint_id)
                candidate_buffer[long_id] = self.get_candidates(scan_id, viewpoint_id)

        with open('./data/candidate_buffer_v2.pkl', 'wb') as f:
            pickle.dump(candidate_buffer, f)


if __name__ == '__main__':
    Foo().all_candidates()
