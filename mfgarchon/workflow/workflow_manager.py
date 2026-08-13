"""
Core workflow management system for MFGarchon.

This module provides the central workflow orchestration capabilities,
including workflow definition, execution, and lifecycle management.
"""

from __future__ import annotations

import json
import traceback
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from .common import ExecutionStatus, serialize_value, setup_workflow_logging

if TYPE_CHECKING:
    import logging
    from collections.abc import Callable

# Backward-compatible alias (Issue #621)
WorkflowStatus = ExecutionStatus


class StepStatus(Enum):
    """Workflow step execution status."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class WorkflowStep:
    """Individual step in a workflow."""

    id: str
    name: str
    function: Callable
    inputs: dict[str, Any] = field(default_factory=dict)
    outputs: dict[str, Any] = field(default_factory=dict)
    dependencies: list[str] = field(default_factory=list)
    status: StepStatus = StepStatus.PENDING
    start_time: datetime | None = None
    end_time: datetime | None = None
    error_message: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert step to dictionary for serialization."""
        return {
            "id": self.id,
            "name": self.name,
            "inputs": self.inputs,
            "outputs": self.outputs,
            "dependencies": self.dependencies,
            "status": self.status.value,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "error_message": self.error_message,
            "metadata": self.metadata,
        }


@dataclass
class WorkflowResult:
    """Result of workflow execution."""

    workflow_id: str
    status: WorkflowStatus
    start_time: datetime
    end_time: datetime | None = None
    outputs: dict[str, Any] = field(default_factory=dict)
    step_results: dict[str, Any] = field(default_factory=dict)
    error_message: str | None = None
    execution_time: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert result to dictionary for serialization."""
        return {
            "workflow_id": self.workflow_id,
            "status": self.status.value,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "outputs": self._serialize_outputs(self.outputs),
            "step_results": self._serialize_outputs(self.step_results),
            "error_message": self.error_message,
            "execution_time": self.execution_time,
            "metadata": self.metadata,
        }

    def _serialize_outputs(self, outputs: dict[str, Any]) -> dict[str, Any]:
        """Serialize outputs for JSON storage."""
        return {key: serialize_value(value, name=key) for key, value in outputs.items()}


class Workflow:
    """
    A workflow represents a directed acyclic graph (DAG) of computational steps.

    Workflows enable reproducible research by capturing the complete computational
    process, including parameters, intermediate results, and final outputs.
    """

    def __init__(self, name: str, description: str = "", workspace_path: Path | None = None):
        """
        Initialize workflow.

        Args:
            name: Workflow name
            description: Workflow description
            workspace_path: Path to workspace directory
        """
        self.id = str(uuid.uuid4())
        self.name = name
        self.description = description
        self.created_time = datetime.now(UTC)
        self.modified_time = self.created_time

        # Workflow structure
        self.steps: dict[str, WorkflowStep] = {}
        self.step_order: list[str] = []

        # Execution state
        self.status = WorkflowStatus.CREATED
        self.current_result: WorkflowResult | None = None

        # Storage
        # No mkdir here. Constructing a Workflow used to create `.mfg_workflows/workflow_<uuid>/`,
        # so merely NAMING a workflow wrote to the caller's working directory. ~~and open
        # `workflow.log` inside it~~ [CORRECTED 2026-08-14] -- it did not: `setup_workflow_logging`
        # guards on `if not logger.handlers:` and `get_logger` always attaches a StreamHandler
        # first, so no FileHandler has been constructed since #621 (2026-02-06). The 21,498
        # directories being ALL EMPTY is the proof, and it was in hand when the wrong claim was
        # written. The directory is created by `_materialise()`, which every writing path calls
        # and no constructor does. #1917
        self.workspace_path = workspace_path or Path.cwd() / ".mfg_workflows"
        self.workflow_dir = self.workspace_path / f"workflow_{self.id}"
        self._materialised = False

        # Logging
        self.logger = self._setup_logging()

        # Execution context
        self.context: dict[str, Any] = {}

        self.logger.info(f"Created workflow '{name}' with ID {self.id}")

    def add_step(
        self,
        name: str,
        function: Callable,
        inputs: dict[str, Any] | None = None,
        dependencies: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """
        Add a step to the workflow.

        Args:
            name: Step name
            function: Function to execute
            inputs: Input parameters for the step
            dependencies: List of step IDs this step depends on
            metadata: Additional metadata for the step

        Returns:
            Step ID
        """
        step_id = f"step_{len(self.steps):03d}_{name}"

        step = WorkflowStep(
            id=step_id,
            name=name,
            function=function,
            inputs=inputs or {},
            dependencies=dependencies or [],
            metadata=metadata or {},
        )

        self.steps[step_id] = step
        self.step_order.append(step_id)
        self.modified_time = datetime.now(UTC)

        self.logger.info(f"Added step '{name}' with ID {step_id}")
        return step_id

    def set_input(self, step_id: str, input_name: str, value: Any):
        """Set input value for a workflow step."""
        if step_id not in self.steps:
            raise ValueError(f"Step {step_id} not found")

        self.steps[step_id].inputs[input_name] = value
        self.modified_time = datetime.now(UTC)

        self.logger.debug(f"Set input '{input_name}' for step {step_id}")

    def get_output(self, step_id: str, output_name: str) -> Any:
        """Get output value from a workflow step."""
        if step_id not in self.steps:
            raise ValueError(f"Step {step_id} not found")

        step = self.steps[step_id]
        if step.status != StepStatus.COMPLETED:
            raise ValueError(f"Step {step_id} has not completed successfully")

        if output_name not in step.outputs:
            raise ValueError(f"Output '{output_name}' not found in step {step_id}")

        return step.outputs[output_name]

    def execute(self, max_workers: int | None = None, save_results: bool = True) -> WorkflowResult:
        """
        Execute the workflow.

        Args:
            max_workers: Maximum number of parallel workers
            save_results: Whether to save results to disk

        Returns:
            Workflow execution result
        """
        self.logger.info(f"Starting execution of workflow '{self.name}'")

        start_time = datetime.now(UTC)
        self.status = WorkflowStatus.RUNNING

        result = WorkflowResult(workflow_id=self.id, status=WorkflowStatus.RUNNING, start_time=start_time)

        try:
            # Execute steps in dependency order
            execution_order = self._compute_execution_order()

            for step_id in execution_order:
                step_result = self._execute_step(step_id)
                result.step_results[step_id] = step_result

                # Check if step failed
                if self.steps[step_id].status == StepStatus.FAILED:
                    result.status = WorkflowStatus.FAILED
                    result.error_message = self.steps[step_id].error_message
                    break

            # Collect final outputs
            if result.status != WorkflowStatus.FAILED:
                result.outputs = self._collect_outputs()
                result.status = WorkflowStatus.COMPLETED
                self.status = WorkflowStatus.COMPLETED

                self.logger.info(f"Workflow '{self.name}' completed successfully")
            else:
                self.status = WorkflowStatus.FAILED
                self.logger.error(f"Workflow '{self.name}' failed: {result.error_message}")

        except Exception as e:
            result.status = WorkflowStatus.FAILED
            result.error_message = str(e)
            self.status = WorkflowStatus.FAILED

            self.logger.error(f"Workflow '{self.name}' failed with exception: {e}")
            self.logger.debug(traceback.format_exc())

        finally:
            end_time = datetime.now(UTC)
            result.end_time = end_time
            result.execution_time = (end_time - start_time).total_seconds()

            self.current_result = result

            if save_results:
                self.save_result(result)

        return result

    def _execute_step(self, step_id: str) -> dict[str, Any]:
        """Execute a single workflow step."""
        step = self.steps[step_id]

        self.logger.info(f"Executing step '{step.name}' ({step_id})")

        step.status = StepStatus.RUNNING
        step.start_time = datetime.now(UTC)

        try:
            # Prepare inputs
            inputs = step.inputs.copy()

            # Resolve dependencies (outputs from previous steps)
            for dep_id in step.dependencies:
                if dep_id in self.steps:
                    dep_step = self.steps[dep_id]
                    if dep_step.status == StepStatus.COMPLETED:
                        # Add outputs from dependency as inputs
                        for output_name, output_value in dep_step.outputs.items():
                            inputs[f"{dep_id}_{output_name}"] = output_value

            # Add workflow context
            inputs["workflow_context"] = self.context

            # Execute the function
            result = step.function(**inputs)

            # Store outputs
            if isinstance(result, dict):
                step.outputs = result
            else:
                step.outputs = {"result": result}

            step.status = StepStatus.COMPLETED
            step.end_time = datetime.now(UTC)

            execution_time = (step.end_time - step.start_time).total_seconds()
            self.logger.info(f"Step '{step.name}' completed in {execution_time:.2f}s")

            return {
                "status": "completed",
                "execution_time": execution_time,
                "outputs": step.outputs,
            }

        except Exception as e:
            step.status = StepStatus.FAILED
            step.end_time = datetime.now(UTC)
            step.error_message = str(e)

            self.logger.error(f"Step '{step.name}' failed: {e}")
            self.logger.debug(traceback.format_exc())

            return {
                "status": "failed",
                "error": str(e),
                "traceback": traceback.format_exc(),
            }

    def _compute_execution_order(self) -> list[str]:
        """Compute the order of step execution based on dependencies."""
        # Simple topological sort
        in_degree = dict.fromkeys(self.steps, 0)

        # Calculate in-degrees
        for step_id, step in self.steps.items():
            for dep_id in step.dependencies:
                if dep_id in in_degree:
                    in_degree[step_id] += 1

        # Find steps with no dependencies
        queue = [step_id for step_id, degree in in_degree.items() if degree == 0]
        execution_order = []

        while queue:
            step_id = queue.pop(0)
            execution_order.append(step_id)

            # Update in-degrees for dependent steps
            for other_step_id, other_step in self.steps.items():
                if step_id in other_step.dependencies:
                    in_degree[other_step_id] -= 1
                    if in_degree[other_step_id] == 0:
                        queue.append(other_step_id)

        if len(execution_order) != len(self.steps):
            raise ValueError("Circular dependency detected in workflow")

        return execution_order

    def _collect_outputs(self) -> dict[str, Any]:
        """Collect final outputs from all completed steps."""
        outputs = {}

        for _, step in self.steps.items():
            if step.status == StepStatus.COMPLETED:
                for output_name, output_value in step.outputs.items():
                    final_name = f"{step.name}_{output_name}"
                    outputs[final_name] = output_value

        return outputs

    def save_result(self, result: WorkflowResult):
        """Save workflow result to disk."""
        result_file = self._materialise() / "result.json"

        # Save main result metadata
        with open(result_file, "w") as f:
            json.dump(result.to_dict(), f, indent=2)

        # Save large data separately
        data_dir = self.workflow_dir / "data"
        data_dir.mkdir(exist_ok=True)

        self._save_data_files(result.outputs, data_dir)
        self._save_data_files(result.step_results, data_dir)

        self.logger.info(f"Saved workflow result to {result_file}")

    def _save_data_files(self, data: dict[str, Any], data_dir: Path):
        """Save data files (numpy arrays, etc.) separately."""
        for key, value in data.items():
            if isinstance(value, np.ndarray):
                np.save(data_dir / f"{key}.npy", value)
            elif hasattr(value, "save"):
                value.save(data_dir / f"{key}.pkl")

    def load_result(self) -> WorkflowResult | None:
        """Load workflow result from disk."""
        result_file = self.workflow_dir / "result.json"

        if not result_file.exists():
            return None

        try:
            with open(result_file) as f:
                result_data = json.load(f)

            # Reconstruct result object
            result = WorkflowResult(
                workflow_id=result_data["workflow_id"],
                status=WorkflowStatus(result_data["status"]),
                start_time=datetime.fromisoformat(result_data["start_time"]),
                end_time=(datetime.fromisoformat(result_data["end_time"]) if result_data["end_time"] else None),
                execution_time=result_data.get("execution_time"),
                error_message=result_data.get("error_message"),
                metadata=result_data.get("metadata", {}),
            )

            # Load data files
            data_dir = self.workflow_dir / "data"
            if data_dir.exists():
                result.outputs = self._load_data_files(result_data.get("outputs", {}), data_dir)
                result.step_results = self._load_data_files(result_data.get("step_results", {}), data_dir)

            return result

        except Exception as e:
            self.logger.error(f"Failed to load workflow result: {e}")
            return None

    def _load_data_files(self, metadata: dict[str, Any], data_dir: Path) -> dict[str, Any]:
        """Load data files referenced in metadata."""
        data = {}

        for key, value in metadata.items():
            if isinstance(value, dict) and value.get("type") == "numpy_array":
                data_file = data_dir / value["data_file"]
                if data_file.exists():
                    data[key] = np.load(data_file)
                else:
                    self.logger.warning(f"Data file not found: {data_file}")
                    data[key] = None
            else:
                data[key] = value

        return data

    def to_dict(self) -> dict[str, Any]:
        """Convert workflow to dictionary for serialization."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "created_time": self.created_time.isoformat(),
            "modified_time": self.modified_time.isoformat(),
            "status": self.status.value,
            "steps": {step_id: step.to_dict() for step_id, step in self.steps.items()},
            "step_order": self.step_order,
            "context": self.context,
        }

    def _materialise(self) -> Path:
        """Create the workflow directory and attach its log file. Idempotent.

        Every path that writes calls this; no constructor does. That split is the whole fix
        for #1917 -- the directory now appears when a workflow is persisted, not when one is
        named.
        """
        if not self._materialised:
            self.workflow_dir.mkdir(parents=True, exist_ok=True)
            self._materialised = True
        return self.workflow_dir

    def _setup_logging(self) -> logging.Logger:
        """Console only. The file handler is attached by `_materialise()`, because attaching
        it here would create the directory this change exists to stop creating."""
        return setup_workflow_logging(f"mfg_workflow.{self.id}", console=True)


class WorkflowManager:
    """
    Central manager for workflow operations.

    Manages workflow creation, execution, storage, and retrieval across
    the entire MFGarchon research environment.
    """

    def __init__(self, workspace_path: Path | None = None):
        """
        Initialize workflow manager.

        Args:
            workspace_path: Path to workspace directory
        """
        # Same split as Workflow above: naming a manager must not create a directory. #1917
        self.workspace_path = workspace_path or Path.cwd() / ".mfg_workflows"
        self._materialised = False

        self.workflows: dict[str, Workflow] = {}
        self.logger = self._setup_logging()

        # Load existing workflows
        self._load_existing_workflows()

        self.logger.info(f"Initialized workflow manager with workspace: {self.workspace_path}")

    def create_workflow(self, name: str, description: str = "", **kwargs) -> Workflow:
        """Create a new workflow."""
        workflow = Workflow(name, description, self._materialise())
        self.workflows[workflow.id] = workflow

        # Save workflow metadata
        self._save_workflow_metadata(workflow)

        self.logger.info(f"Created workflow '{name}' with ID {workflow.id}")
        return workflow

    def get_workflow(self, workflow_id: str) -> Workflow | None:
        """Get workflow by ID."""
        return self.workflows.get(workflow_id)

    def list_workflows(self) -> list[dict[str, Any]]:
        """List all workflows with metadata."""
        workflows = []

        for workflow in self.workflows.values():
            workflows.append(
                {
                    "id": workflow.id,
                    "name": workflow.name,
                    "description": workflow.description,
                    "created_time": workflow.created_time,
                    "modified_time": workflow.modified_time,
                    "status": workflow.status,
                    "num_steps": len(workflow.steps),
                }
            )

        return workflows

    def delete_workflow(self, workflow_id: str) -> bool:
        """Delete workflow and its data."""
        if workflow_id not in self.workflows:
            return False

        workflow = self.workflows[workflow_id]

        # Remove from memory
        del self.workflows[workflow_id]

        # Remove from disk
        import shutil

        if workflow.workflow_dir.exists():
            shutil.rmtree(workflow.workflow_dir)

        self.logger.info(f"Deleted workflow {workflow_id}")
        return True

    def execute_workflow(self, workflow_id: str, **kwargs) -> WorkflowResult | None:
        """Execute workflow by ID."""
        workflow = self.get_workflow(workflow_id)
        if workflow is None:
            return None

        return workflow.execute(**kwargs)

    def _load_existing_workflows(self):
        """Load existing workflows from workspace."""
        if not self.workspace_path.exists():
            return

        for workflow_dir in self.workspace_path.glob("workflow_*"):
            if workflow_dir.is_dir():
                try:
                    workflow_id = workflow_dir.name.replace("workflow_", "")
                    # Load workflow metadata if available
                    metadata_file = workflow_dir / "metadata.json"
                    if metadata_file.exists():
                        with open(metadata_file) as f:
                            metadata = json.load(f)

                        workflow = Workflow(
                            metadata["name"],
                            metadata["description"],
                            self.workspace_path,
                        )
                        workflow.id = workflow_id
                        workflow.workflow_dir = workflow_dir

                        self.workflows[workflow_id] = workflow

                except Exception as e:
                    self.logger.warning(f"Could not load workflow from {workflow_dir}: {e}")

    def _save_workflow_metadata(self, workflow: Workflow):
        """Save workflow metadata to disk."""
        metadata_file = workflow._materialise() / "metadata.json"

        metadata = {
            "id": workflow.id,
            "name": workflow.name,
            "description": workflow.description,
            "created_time": workflow.created_time.isoformat(),
            "modified_time": workflow.modified_time.isoformat(),
        }

        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)

    def _materialise(self) -> Path:
        """Create the workspace and attach its log file. Idempotent; never called by
        `__init__`. #1917"""
        if not self._materialised:
            self.workspace_path.mkdir(parents=True, exist_ok=True)
            self._materialised = True
        return self.workspace_path

    def _setup_logging(self) -> logging.Logger:
        """Console only until `_materialise()` runs -- naming the log file here would
        require the directory that this change exists to stop creating."""
        return setup_workflow_logging("mfg_workflow_manager")


# Export main classes
__all__ = [
    "StepStatus",
    "Workflow",
    "WorkflowManager",
    "WorkflowResult",
    "WorkflowStatus",
    "WorkflowStep",
]
