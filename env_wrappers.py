"""Environment utilities for Pong with manual frame repetition."""
from __future__ import annotations

from typing import List, Tuple, Optional

import gymnasium as gym
import numpy as np

try:
    import ale_py
except ImportError as exc:  # pragma: no cover - helpful error when ALE missing
    raise ImportError(
        "ale-py is required for Atari environments. Install with 'pip install ale-py'."
    ) from exc


_ALE_REGISTERED = False


def _ensure_ale_registered() -> None:
    global _ALE_REGISTERED
    if not _ALE_REGISTERED:
        if hasattr(gym, "register_envs"):
            gym.register_envs(ale_py)
        else:  # pragma: no cover
            raise RuntimeError("This version of gymnasium does not expose register_envs; update gymnasium.")
        _ALE_REGISTERED = True


class _Float32ObsWrapper(gym.ObservationWrapper):
    """Convert uint8 observations into float32 in [0, 1]."""

    def __init__(self, env: gym.Env):
        super().__init__(env)
        assert len(self.observation_space.shape) in (2, 3)
        self._obs_shape = self.observation_space.shape

    def observation(self, observation: np.ndarray) -> np.ndarray:
        obs = observation.astype(np.float32) / 255.0
        if obs.ndim == 2:
            obs = np.expand_dims(obs, axis=-1)
        return obs


def make_pong_env(seed: int, record_dir: Optional[str] = None) -> gym.Env:
    """Create a Pong environment with frameskip=1 and grayscale 84x84 observations."""

    _ensure_ale_registered()
    make_kwargs = dict(
        frameskip=1,
        repeat_action_probability=0.25,
        obs_type="rgb",
        full_action_space=False,
    )
    if record_dir is not None:
        make_kwargs["render_mode"] = "rgb_array"
    env = gym.make("ALE/Pong-v5", **make_kwargs)
    env = gym.wrappers.AtariPreprocessing(
        env,
        noop_max=1,
        frame_skip=1,
        screen_size=84,
        grayscale_obs=True,
        grayscale_newaxis=True,
        scale_obs=False,
    )
    env = _Float32ObsWrapper(env)
    if record_dir is not None:
        env = gym.wrappers.RecordVideo(
            env,
            video_folder=record_dir,
            episode_trigger=lambda episode_id: True,
            name_prefix="pong_episode",
        )
    env.reset(seed=seed)
    env.action_space.seed(seed)
    return env


def repeat_action_four(env: gym.Env, action: int) -> Tuple[List[np.ndarray], float, bool, dict]:
    """Repeat ``action`` for four frames and return the collected data."""

    frames: List[np.ndarray] = []
    total_reward = 0.0
    done = False
    info: dict = {}
    frame_shape = env.observation_space.shape
    zero_frame = np.zeros(frame_shape, dtype=np.float32)

    for _ in range(4):
        obs, reward, terminated, truncated, info = env.step(action)
        frames.append(obs.astype(np.float32))
        total_reward += float(reward)
        done = bool(terminated or truncated)
        if done:
            break

    if len(frames) < 4:
        frames.extend([zero_frame.copy() for _ in range(4 - len(frames))])

    return frames, total_reward, done, info
