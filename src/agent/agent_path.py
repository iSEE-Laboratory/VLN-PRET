import torch
import numpy as np
import pandas as pd
import os.path as osp
import torch.nn as nn
import networkx as nx
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.distributions import Categorical

import utils
from config import device
from dataset import connectivity


class AgentPath(nn.Module):
    def __init__(self, config, model):
        super().__init__()
        self.model = model
        self.STOP_embedding = nn.Parameter(torch.zeros(1, 1, self.model.hidden_dim))
        # NOTE: I find that I forgot to save these parameters, so the loaded model is slightly different

        self.max_step = config.max_step
        self.loss_weight = config.loss_weight
        self.mask_visited = config.mask_visited
        self.use_panorama = config.use_panorama
        self.use_directed = config.use_directed

    def reset(self, batch):
        self.groundtruth_paths = batch['path']
        self.text_features, self.text_padding_mask = self.model.forward_text(batch['instruction'].tolist())
        if (not self.use_panorama) or self.use_directed:
            self.graph = BatchGraph_trajectory(batch)
        else:
            self.graph = BatchGraph_undirected_trajectory(batch)

        # used during inference.
        if not self.training:
            self.step = 0
            self.batch_path = None
            self.stopped = np.zeros(len(batch), dtype=np.bool_)

    def take_action(self, obs):
        """
        make low-level action according to make high-level action.
        """
        assert not self.training

        if self.batch_path is None:
            self.step += 1
            probs, unvisited_or_current = self._take_action(obs)
            batch_path = self._get_path(obs, probs.argmax(dim=1).tolist(), unvisited_or_current)
            self.batch_path = pd.DataFrame(batch_path)
            self.local_step = 1

        # get local action to implement global action
        local_actions = np.zeros(len(obs), dtype=np.int64)  # defulat action is STOP
        if self.batch_path.shape[1] > 1:
            for i in range(obs.shape[0]):
                if self.stopped[i]:
                    continue
                next_viewpoint_id = self.batch_path.at[i, self.local_step]
                local_actions[i] = self._get_local_selection(obs.at[i, 'candidates'], next_viewpoint_id)

            # temporary stop due to batch data.
            stop_at_this_step = self.batch_path.iloc[:, 1].isna().to_numpy()
            self.stopped |= stop_at_this_step
            temporary_stop = (local_actions == 0) & ~self.stopped
            local_actions[temporary_stop] = -1

        # global action is ended
        self.local_step += 1
        if self.local_step >= self.batch_path.shape[1]:
            self.batch_path = None

        return local_actions

    def _get_path(self, obs, target_ids, unvisited_or_current):
        source_batch = obs['viewpoint_id'].tolist()
        target_batch = []

        for i, target_id in enumerate(target_ids):
            target = unvisited_or_current[i][target_id]
            target_batch.append(target)
        batch_path = self.graph.get_path(source_batch, target_batch)
        return batch_path

    def _get_global_selection(self, unvisited_or_current, next_viewpoint_id):
        if next_viewpoint_id is None:  # STOP
            action = 0
        else:
            action = unvisited_or_current.index(next_viewpoint_id)
        return action

    def _get_local_selection(self, candidates, next_viewpoint_id):
        if next_viewpoint_id is None:
            action = 0  # STOP
        else:
            action = candidates['viewpoint_id'].tolist().index(next_viewpoint_id)
            action += 1  # + 1 for action STOP
        return action

    # @profile
    def _take_action(self, obs):
        # panoramic feature
        panoramic_features = obs['panoramic_feature'].tolist()
        panoramic_features = np.stack(panoramic_features, axis=0)  # (B, 36, C)
        panoramic_features = torch.from_numpy(panoramic_features).to(device)

        # candidate feature
        candidate_features = obs['candidate_feature'].tolist()
        candidate_lengths = [len(candidates) for candidates in candidate_features]
        for i, c in enumerate(candidate_features):
            candidate_features[i] = torch.from_numpy(c).to(device)
        candidate_features = pad_sequence(candidate_features, batch_first=True)
        candidate_padding_mask = utils.length_to_mask(candidate_lengths, device=device)

        # update edge features
        edge_tokens = self.model.forward_OPE(candidate_features, panoramic_features, candidate_padding_mask)
        self.graph.update_graph(obs, edge_tokens)

        # add STOP action embedding
        path_features, path_padding_mask, local_features, local_padding_mask = self.graph.get_path_features(obs)
        stop_embedding = self.STOP_embedding.expand(len(obs), 1, -1)
        local_features = torch.cat([stop_embedding, local_features], dim=1)
        local_padding_mask = F.pad(local_padding_mask, (1, 0), value=False)

        # forward path score
        path_tokens, local_tokens = self.model.forward_MAM(
            self.text_features, self.text_padding_mask,
            path_features, path_padding_mask,
            local_features, local_padding_mask)
        self.graph.update_path_feature(obs, local_tokens)

        # select path
        unvisited_or_current, path_feature, padding_mask = self.graph.get_unvisited_viewpoints(obs, self.mask_visited)
        path_score = self.model.forward_CCM(path_feature, padding_mask)
        path_score = path_score - path_score.max(dim=1, keepdim=True).values
        probs = torch.softmax(path_score, dim=1)

        return probs, unvisited_or_current

    def forward(self, batch, env):
        loss_log = pd.Series(dtype=np.float32)

        # Teacher Forcing
        loss_TF = 0.0
        if self.loss_weight > 0:
            loss_TF = self._TF_rollout(batch, env)
            loss_log['IL'] = loss_TF.item()

        # Student Forcing
        loss_SF = 0.0
        if self.loss_weight < 1:
            loss_SF = self._SF_rollout(batch, env)
            loss_log['SF'] = loss_SF.item()

        # total loss
        loss = self.loss_weight * loss_TF + (1 - self.loss_weight) * loss_SF
        return loss, loss_log

    def _TF_rollout(self, batch, env):
        self.reset(batch)
        obs = env.reset()

        loss = 0.0
        stopped = torch.zeros(len(batch), 1, dtype=torch.bool, device=device)
        step = torch.zeros(len(batch), 1, dtype=torch.float32, device=device)
        for _ in range(self.max_step):
            probs, unvisited_or_current = self._take_action(obs)

            # get ground truth global action to compute loss
            global_teacher_actions = []
            for i in range(len(batch)):
                action = self._get_global_selection(unvisited_or_current[i], obs.at[i, 'next_viewpoint_id'])
                global_teacher_actions.append(action)
            global_teacher_actions = torch.tensor(global_teacher_actions, device=device)

            not_stopped = ~stopped
            step += not_stopped.float()

            loss = loss + utils.cross_entropy_loss(probs, global_teacher_actions, reduction='none') * not_stopped

            # get local teacher action to step
            local_actions = obs['teacher_selection'].to_numpy(dtype=np.int64)
            local_actions_gpu = torch.from_numpy(local_actions).to(device).unsqueeze(1)

            stopped = stopped | (local_actions_gpu == 0)
            if stopped.all():
                break

            obs = env.step_viewpoint(obs, local_actions)
        return (loss / step).mean()

    def _get_teacher_actions(self, obs, unvisited_or_current):
        teacher_actions = []

        goal = [path[-1] for path in self.groundtruth_paths]
        for i in range(len(obs)):
            # get closest node to goal
            scan_id = obs.at[i, 'scan_id']
            viewpoint_id = obs.at[i, 'viewpoint_id']
            if viewpoint_id == goal[i]:  # if reach the goal, stop
                next_viewpoint_id = viewpoint_id
            elif self.mask_visited and goal[i] in self.graph.visited_batch[i]:
                # loss will not be computed if miss the target when mask visited nodes.
                next_viewpoint_id = None
            else:
                min_distance_to_target = 1e6

                target_candidates = set(self.groundtruth_paths[i]) & set(unvisited_or_current[i])
                if viewpoint_id in target_candidates:
                    target_candidates.remove(viewpoint_id)
                if len(target_candidates) > 0:
                    # we encorage the model to go back to gt path if makes mistake.
                    for node in target_candidates:
                        distance_to_target = connectivity.get_distance(scan_id, node, goal[i])
                        if distance_to_target < min_distance_to_target:
                            next_viewpoint_id = node
                            min_distance_to_target = distance_to_target
                else:
                    # TODO: for RxR, the shortest path may hinder agents learn to align path and instruction
                    shortest_path_to_goal = connectivity.get_path(scan_id, viewpoint_id, goal[i])
                    for node in set(shortest_path_to_goal) & set(unvisited_or_current[i]):
                        distance_to_target = connectivity.get_distance(scan_id, node, goal[i])
                        if distance_to_target < min_distance_to_target:
                            next_viewpoint_id = node
                            min_distance_to_target = distance_to_target

                assert next_viewpoint_id is not None

            # get target
            action = self._get_global_selection(unvisited_or_current[i], next_viewpoint_id) if next_viewpoint_id is not None else -1
            teacher_actions.append(action)
        target = torch.tensor(teacher_actions, device=device)
        return target

    def _SF_rollout(self, batch, env):
        self.reset(batch)
        obs = env.reset()

        # rollout
        loss = 0.0
        stopped = np.zeros(len(batch), dtype=np.bool_)
        step = torch.zeros(len(batch), 1, dtype=torch.float32, device=device)
        for _ in range(self.max_step):
            probs, unvisited_or_current = self._take_action(obs)
            actions = Categorical(probs=probs).sample()
            actions[stopped] = 0

            # step of each batch 
            not_stopped = torch.from_numpy(~stopped).to(device).unsqueeze(1)
            step += not_stopped.float()

            # get teacher action
            target = self._get_teacher_actions(obs, unvisited_or_current)
            actions[target == -1] = 0
            loss = loss + utils.cross_entropy_loss(probs, target, target < 0, reduction='none') * not_stopped

            # step to target viewpoint to the global action
            batch_path = self._get_path(obs, actions.tolist(), unvisited_or_current)
            obs_ = env.step_target(obs, batch_path)
            stop_at_this_step = np.array([len(path) == 1 for path in batch_path])

            stopped = stopped | stop_at_this_step
            if stopped.all():
                break

            obs = obs_
        return (loss / step).mean()

    def save(self, log_dir):
        # NOTE: you may need to save self.STOP_embedding
        for name, model in self.named_children():
            path = osp.join(log_dir, f'{name}.pt')
            torch.save(model.state_dict(), path)

    def load(self, log_dir, strict=True):
        # NOTE: you may need to load self.STOP_embedding
        load_any = False
        for name, model in self.named_children():
            path = osp.join(log_dir, f'{name}.pt')
            if osp.exists(path):
                checkpoint = torch.load(path, map_location='cpu')
                model.load_state_dict(checkpoint, strict=strict)
                print(f'{name}.pt loaded')
                load_any = True
        assert load_any, 'Does not load any model!'




class BatchGraph_trajectory:
    """
    This trajectory strategy should be improved.
    """
    def __init__(self, batch):
        batch_size = len(batch)
        self.hidden_dim = None

        self.to_be_updated = None
        self.trajectory_batch = [[None] for _ in range(batch_size)]

        self.start_point = [path[0] for path in batch['path'].tolist()]
        self.graph_batch = [nx.DiGraph() for _ in range(batch_size)]
        self.visited_batch = [set() for _ in range(batch_size)]

    def update_graph(self, obs, candidate_features,):
        """
        Construct graph and edge feature.
        """
        self.hidden_dim = candidate_features.shape[2]
        self.to_be_updated = [set() for _ in range(len(obs))]

        for i, candidates in enumerate(obs['candidates']):
            v1 = obs.at[i, 'viewpoint_id']
            self.visited_batch[i].add(v1)

            # update trajectory
            last_node = self.trajectory_batch[i][-1]
            if last_node is None:  # just start navigation
                self.trajectory_batch[i].append(v1)
            elif last_node == v1:  # stopped
                continue
            else:
                path = nx.shortest_path(self.graph_batch[i], last_node, v1)
                for j, node in enumerate(path):
                    # path: [last_node, ..., v1]
                    # trajectory: [..., last_node]
                    if node == self.trajectory_batch[i][-1]:
                        next_node = path[j+1]
                        if next_node != self.trajectory_batch[i][-2]:
                            continue
                        else:  # go back
                            self.trajectory_batch[i].pop()
                    else:
                        self.trajectory_batch[i].append(node)

            # add new nodes and edges
            for j in range(candidates.shape[0]):
                v2 = candidates.at[j, 'viewpoint_id']
                distance = candidates.at[j, 'distance']
                self.graph_batch[i].add_edge(v1, v2, distance=distance, feature=candidate_features[i, j])

                if v2 not in self.visited_batch[i]:
                    self.to_be_updated[i].add(v2)

    def get_path_features(self, obs):
        """
        Returns:
            path_features: (B, path_length, 768)
            path_padding_mask: (B, path_length)
        """
        path_features = []
        local_features = []
        for i, graph in enumerate(self.graph_batch):
            current_viewpoint_id = obs.at[i, 'viewpoint_id']
            trajectory = self.trajectory_batch[i][1:]  # remove None
            path_feature = [graph.edges[v, trajectory[i+1]]['feature'] for i, v in enumerate(trajectory[:-1])]
            path_feature = torch.stack(path_feature) \
                                if len(path_feature) > 0 \
                                else torch.zeros(0, self.hidden_dim, device=device)
            path_features.append(path_feature)

            local_feature = [graph.edges[current_viewpoint_id, v]['feature'] for v in self.to_be_updated[i]]
            local_feature = torch.stack(local_feature) \
                                if len(local_feature) > 0 \
                                else torch.zeros(0, self.hidden_dim, device=device)
            local_features.append(local_feature)
        path_lengths = [f.shape[0] for f in path_features]
        path_padding_mask = utils.length_to_mask(path_lengths, device=device)
        path_features = pad_sequence(path_features, batch_first=True)

        local_lengths = [f.shape[0] for f in local_features]
        local_padding_mask = utils.length_to_mask(local_lengths, device=device)
        local_features = pad_sequence(local_features, batch_first=True)
        return path_features, path_padding_mask, local_features, local_padding_mask

    def update_path_feature(self, obs, path_feature):
        for i, graph in enumerate(self.graph_batch):
            v1 = obs.at[i, 'viewpoint_id']
            graph.add_node(v1, path_feature=path_feature[i, 0])  # STOP embedding

            for j, v2 in enumerate(self.to_be_updated[i]):
                graph.add_node(v2, path_feature=path_feature[i, j+1])

    def get_unvisited_viewpoints(self, obs, mask_visited=True):
        """
        get unvisited or current viewpoints score.
        Returns:
            unvisited_batch: list[list[str]]
            score_batch: (B, N)
            padding_mask: (B, N)
        """
        score_batch = []
        unvisited_batch = []
        for i, graph in enumerate(self.graph_batch):
            current_viewpoint_id = obs.at[i, 'viewpoint_id']
            unvisited = [v for v in graph.nodes if v not in self.visited_batch[i]] \
                        if mask_visited \
                        else [v for v in graph.nodes if v != current_viewpoint_id]
            unvisited_or_current = [current_viewpoint_id] + unvisited
            unvisited_batch.append(unvisited_or_current)

            score = [graph.nodes[v]['path_feature'] for v in unvisited_or_current]
            score = torch.stack(score)  # (N,)
            score_batch.append(score)
        lengths = [s.shape[0] for s in score_batch]
        padding_mask = utils.length_to_mask(lengths, device=device)
        score_batch = pad_sequence(score_batch, batch_first=True)  # (B, N)
        return unvisited_batch, score_batch, padding_mask

    def get_path(self, source_batch, target_batch):
        """
        Args:
            source_batch: list[str]
            target_batch: list[str]
        Returns:
            list[list[str]]
        """
        path_batch = []
        for i, graph in enumerate(self.graph_batch):
            source = source_batch[i]
            target = target_batch[i]
            path = nx.shortest_path(graph, source, target)
            path_batch.append(path)
        return path_batch




class BatchGraph_undirected_trajectory(BatchGraph_trajectory):
    """
    undirected graph, feature is stored on node.
    """
    def __init__(self, batch):
        batch_size = len(batch)
        self.hidden_dim = None

        self.to_be_updated = None
        self.last_nodes = [None for _ in range(batch_size)]
        self.distance_batch = [dict() for _ in range(batch_size)]

        self.trajectory_batch = [[None] for _ in range(batch_size)]

        self.start_point = [path[0] for path in batch['path'].tolist()]
        self.graph_batch = [nx.DiGraph() for _ in range(batch_size)]
        self.visited_batch = [set() for _ in range(batch_size)]

    def update_graph(self, obs, panoramic_features):
        self.hidden_dim = panoramic_features.shape[2]
        self.to_be_updated = [set() for _ in range(len(obs))]

        for i, candidates in enumerate(obs['candidates']):
            v1 = obs.at[i, 'viewpoint_id']
            if v1 == self.last_nodes[i]:  # stopped
                continue

            self.last_nodes[i] = v1
            self.visited_batch[i].add(v1)
            self.graph_batch[i].add_node(v1, feature=panoramic_features[i].mean(dim=0))

            # update fidelity trajectory
            remove_detours = True
            if remove_detours:
                last_node = self.trajectory_batch[i][-1]
                if last_node is None:  # start navigation
                    self.trajectory_batch[i].append(v1)
                elif last_node == v1:  # stopped
                    continue
                else:
                    path = nx.shortest_path(self.graph_batch[i], last_node, v1)
                    for j, node in enumerate(path):
                        # path: [last_node, ..., v1]
                        # trajectory: [..., last_node]
                        if node == self.trajectory_batch[i][-1]:
                            next_node = path[j+1]
                            if next_node != self.trajectory_batch[i][-2]:
                                continue
                            else:  # go back
                                self.trajectory_batch[i].pop()
                        else:
                            self.trajectory_batch[i].append(node)

            for j in range(candidates.shape[0]):
                v2 = candidates.at[j, 'viewpoint_id']
                distance = candidates.at[j, 'distance']
                view_index = candidates.at[j, 'view_index']
                self.graph_batch[i].add_edge(v1, v2, distance=distance)
                if 'feature' not in  self.graph_batch[i].nodes[v2]:
                    # first see the node
                    self.graph_batch[i].add_node(v2, feature=[panoramic_features[i, view_index]])
                elif isinstance(self.graph_batch[i].nodes[v2]['feature'], list):
                    # partially observed
                    self.graph_batch[i].nodes[v2]['feature'].append(panoramic_features[i, view_index])

                if v2 not in self.visited_batch[i]:
                    self.to_be_updated[i].add(v2)

    def get_path_features(self, obs):
        """
        Returns:
            path_features: (B, path_length, 768)
            path_padding_mask: (B, path_length)
        """
        path_features = []
        local_features = []
        for i, graph in enumerate(self.graph_batch):
            trajectory = self.trajectory_batch[i][1:]  # remove None
            path_feature = [graph.nodes[v]['feature'] for v in trajectory]
            path_feature = torch.stack(path_feature) \
                                if len(path_feature) > 0 \
                                else torch.zeros(0, self.hidden_dim, device=device)
            path_features.append(path_feature)

            local_feature = [sum(graph.nodes[v]['feature']) / len(graph.nodes[v]['feature']) \
                             for v in self.to_be_updated[i]]
            local_feature = torch.stack(local_feature) \
                                if len(local_feature) > 0 \
                                else torch.zeros(0, self.hidden_dim, device=device)
            local_features.append(local_feature)
        path_lengths = [f.shape[0] for f in path_features]
        path_padding_mask = utils.length_to_mask(path_lengths, device=device)
        path_features = pad_sequence(path_features, batch_first=True)

        local_lengths = [f.shape[0] for f in local_features]
        local_padding_mask = utils.length_to_mask(local_lengths, device=device)
        local_features = pad_sequence(local_features, batch_first=True)
        return path_features, path_padding_mask, local_features, local_padding_mask

