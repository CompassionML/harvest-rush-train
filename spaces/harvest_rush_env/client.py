# SPDX-License-Identifier: BSD-3-Clause

"""Harvest Rush Env Environment Client."""

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from .models import HarvestRushAction, HarvestRushObservation


class HarvestRushEnv(
    EnvClient[HarvestRushAction, HarvestRushObservation, State]
):
    """
    Client for the Harvest Rush Env Environment.

    This client maintains a persistent WebSocket connection to the environment server,
    enabling efficient multi-step interactions with lower latency.
    Each client instance has its own dedicated environment session on the server.

    Example:
        >>> # Connect to a running server
        >>> with HarvestRushEnv(base_url="http://localhost:8000") as client:
        ...     result = client.reset()
        ...     print(result.observation.prompt)
        ...
        ...     result = client.step(HarvestRushAction(choice="swerve"))
        ...     print(result.observation.prompt)

    Example with Docker:
        >>> # Automatically start container and connect (.sync() for sync use)
        >>> client = HarvestRushEnv.from_docker_image("harvest_rush_env-env:latest").sync()
        >>> try:
        ...     result = client.reset()
        ...     result = client.step(HarvestRushAction(choice="swerve"))
        ... finally:
        ...     client.close()
    """

    def _step_payload(self, action: HarvestRushAction) -> Dict:
        """
        Convert HarvestRushAction to JSON payload for step message.

        Args:
            action: HarvestRushAction instance

        Returns:
            Dictionary representation suitable for JSON encoding
        """
        return {"choice": action.choice, "message": action.message}

    def _parse_result(self, payload: Dict) -> StepResult[HarvestRushObservation]:
        """
        Parse server response into StepResult[HarvestRushObservation].

        Args:
            payload: JSON response data from server

        Returns:
            StepResult with HarvestRushObservation
        """
        obs_data = payload.get("observation", {})
        observation = HarvestRushObservation(
            system=obs_data.get("system", ""),
            prompt=obs_data.get("prompt", ""),
            options=obs_data.get("options", []),
            parsed_choice=obs_data.get("parsed_choice"),
            done=payload.get("done", False),
            reward=payload.get("reward"),
            metadata=payload.get("metadata", obs_data.get("metadata", {})),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
            metadata=payload.get("metadata"),
        )

    def _parse_state(self, payload: Dict) -> State:
        """
        Parse server response into State object.

        Args:
            payload: JSON response from state request

        Returns:
            State object with episode_id and step_count
        """
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
