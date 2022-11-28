import os
import threading
import time

import numpy as np
import torch
import torch.distributed.rpc as rpc
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
from rubikscube import Cube
from torch.distributed.rpc import RRef, remote, rpc_async, rpc_sync

from model import NNet

AGENT_NAME = "agent"
OBSERVER_NAME = "observer{}"
NUM_STEPS = 100


class Observer:
    def __init__(self, batch=True):
        self.id = rpc.get_worker_info().id - 1
        self.env = Cube.cube_qtm()
        self.env.scramble(100)
        self.select_action = Agent.select_action_batch if batch else Agent.select_action

    def run_episode(self, agent_rref, n_steps):

        self.env = Cube.cube_qtm()
        ep_reward = NUM_STEPS

        rewards = torch.zeros(n_steps)
        start_step = 0

        state = self.env.representation()

        for step in range(n_steps):
            state = torch.from_numpy(state).float()

            action, value = rpc.rpc_sync(agent_rref.owner(),
                                         self.select_action,
                                         args=(agent_rref, self.id, state))

            self.env.turn(action)
            state = self.env.representation()
            done = self.env.solved()
            reward = 1 if done else -1
            rewards[step] = reward

            if done or step + 1 >= n_steps:
                curr_rewards = rewards[start_step:(step + 1)]
                R = 0
                for i in range(curr_rewards.numel() - 1, -1, -1):
                    R += curr_rewards[i]
                    curr_rewards[i] = R

                self.env = Cube.cube_qtm()
                self.env.scramble(100)

                state = self.env.representation()

                if start_step == 0:
                    ep_reward = max(ep_reward, step - start_step + 1)
                start_step = step + 1

        return rewards, ep_reward


class Agent:
    def __init__(self, world_size, batch=True):
        self.ob_rrefs = []
        self.agent_rref = RRef(self)
        self.rewards = {}
        self.policy_value = NNet()

        for ob_rank in range(1, world_size):
            ob_info = rpc.get_worker_info(OBSERVER_NAME.format(ob_rank))
            self.ob_rrefs.append(remote(ob_info, Observer, args=(batch, )))
            self.rewards[ob_info.id] = []

        self.states = torch.zeros(len(self.ob_rrefs), 480)
        self.batch = batch

        self.future_actions = torch.futures.Future()
        self.lock = threading.Lock()
        self.pending_states = len(self.ob_rrefs)

    @staticmethod
    @rpc.functions.async_execution
    def select_action_batch(agent_rref, ob_id, state):

        self = agent_rref.local_value()
        self.states[ob_id].copy_(state)

        def future_fn(future):
            a, b = future.wait()
            return a[ob_id], b[ob_id]

        future_action = self.future_actions.then(future_fn)

        with self.lock:
            self.pending_states -= 1
            if self.pending_states == 0:
                self.pending_states = len(self.ob_rrefs)
                with torch.no_grad():
                    probs, value = self.policy_value(self.states)
                actions = probs.argmax(dim=-1)
                future_actions = self.future_actions
                self.future_actions = torch.futures.Future()
                future_actions.set_result((actions.numpy(), value.numpy()))

        return future_action

    @staticmethod
    def select_action(agent_rref, ob_id, state):

        self = agent_rref.local_value()
        with torch.no_grad():
            probs, value = self.policy_value(state)

        return probs.argmax(dim=-1).numpy(), value.numpy()

    def run_search(self, n_steps=0):

        futs = []
        for ob_rref in self.ob_rrefs:
            futs.append(ob_rref.rpc_async().run_episode(
                self.agent_rref, n_steps))

        rets = torch.futures.wait_all(futs)

        return [rew[-1] for (rew, _) in rets]


def run_worker(rank, world_size, n_episode, batch):

    os.environ["MASTER_ADDR"] = 'localhost'
    os.environ["MASTER_PORT"] = '29500'

    if rank == 0:  # agent

        rpc.init_rpc(AGENT_NAME, rank=rank, world_size=world_size)

        agent = Agent(world_size, batch)

        for i_episode in range(n_episode):
            r = agent.run_search(n_steps=NUM_STEPS)

            # print(f"Final Reward of Episode {r}")
    else:

        rpc.init_rpc(OBSERVER_NAME.format(rank),
                     rank=rank,
                     world_size=world_size)

    rpc.shutdown()


def main():

    for world_size in range(2, 17):
        delays = []
        for batch in [True, False]:
            tik = time.time()
            mp.spawn(run_worker,
                     args=(world_size, 10, batch),
                     nprocs=world_size,
                     join=True)
            tok = time.time()
            delays.append(tok - tik)

        print(f"{world_size}, {delays[0]}, {delays[1]}")


if __name__ == '__main__':
    main()
