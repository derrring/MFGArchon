"""
Unit tests for Workflow Manager.

Tests the core workflow orchestration system including workflow creation,
step management, execution flow, and error handling.
"""

import tempfile
from pathlib import Path

import pytest

from mfgarchon.workflow.workflow_manager import (
    StepStatus,
    Workflow,
    WorkflowManager,
    WorkflowStatus,
)

# ============================================================================
# Test: Workflow Initialization
# ============================================================================


@pytest.mark.unit
@pytest.mark.fast
def test_workflow_creation_basic():
    """Test basic workflow creation."""
    wf = Workflow(name="test_workflow")

    assert wf.name == "test_workflow"
    assert wf.description == ""
    assert wf.status == WorkflowStatus.CREATED
    assert len(wf.steps) == 0
    assert wf.id is not None


@pytest.mark.unit
@pytest.mark.fast
def test_workflow_creation_with_description():
    """Test workflow creation with description."""
    wf = Workflow(name="analysis", description="Data analysis workflow")

    assert wf.name == "analysis"
    assert wf.description == "Data analysis workflow"


@pytest.mark.unit
@pytest.mark.fast
def test_workflow_creation_with_workspace():
    """Test workflow creation with custom workspace path.

    ~~`assert wf.workflow_dir.exists()`~~ [CORRECTED 2026-08-14] — construction no longer
    creates the directory (#1917). The path is computed; the directory appears when something
    is persisted into it. See `test_the_directory_appears_when_something_is_saved` below.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        workspace = Path(tmpdir)
        wf = Workflow(name="test", workspace_path=workspace)

        assert wf.workspace_path == workspace
        assert wf.workflow_dir.parent == workspace
        assert not wf.workflow_dir.exists(), "naming a workflow must not create its directory"


@pytest.mark.unit
@pytest.mark.fast
def test_workflow_generates_unique_ids():
    """Test that each workflow gets a unique ID."""
    wf1 = Workflow(name="workflow1")
    wf2 = Workflow(name="workflow2")

    assert wf1.id != wf2.id


@pytest.mark.unit
@pytest.mark.fast
def test_workflow_does_not_create_a_directory_when_merely_named():
    """[SUPERSEDED 2026-08-14] Was `test_workflow_creates_directory`, asserting the opposite.

    It pinned the defect, and it was also a producer of it: `Workflow(name="test")` passes no
    workspace, so every run of this test wrote `.mfg_workflows/workflow_<uuid>/` into whatever
    directory pytest was launched from. That is part of how 21,498 empty directories reached
    this repository (#1917). A workspace is named here so the assertion cannot depend on where
    the suite is run from.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        wf = Workflow(name="test", workspace_path=Path(tmpdir) / "ws")

        assert not wf.workflow_dir.exists()
        assert not wf.workspace_path.exists(), "naming a workflow created the workspace itself"


def test_the_directory_appears_when_something_is_saved():
    """The other direction: deferring the write must not mean never writing.

    A change that simply stopped persisting would satisfy every "creates nothing" assertion in
    this file and in tests/unit/test_no_writes_at_import.py.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        workspace = Path(tmpdir) / "ws"
        manager = WorkflowManager(workspace_path=workspace)
        assert not workspace.exists(), "the manager created its workspace at construction"

        wf = manager.create_workflow("real", "persisted on purpose")

        assert wf.workflow_dir.exists(), "create_workflow did not materialise the workflow"
        assert (wf.workflow_dir / "metadata.json").exists(), "metadata was not written"


# ============================================================================
# Test: Step Management
# ============================================================================


@pytest.mark.unit
@pytest.mark.fast
def test_add_step_basic():
    """Test adding a basic step to workflow."""
    wf = Workflow(name="test")

    def dummy_function(**kwargs):
        return {"value": 42}

    step_id = wf.add_step("compute", dummy_function)

    assert step_id in wf.steps
    assert wf.steps[step_id].name == "compute"
    assert wf.steps[step_id].function == dummy_function
    assert wf.steps[step_id].status == StepStatus.PENDING


@pytest.mark.unit
@pytest.mark.fast
def test_add_step_with_inputs():
    """Test adding step with input parameters."""
    wf = Workflow(name="test")

    def process(x, y, **kwargs):
        return x + y

    inputs = {"x": 10, "y": 20}
    step_id = wf.add_step("add", process, inputs=inputs)

    assert wf.steps[step_id].inputs == inputs


@pytest.mark.unit
@pytest.mark.fast
def test_add_step_with_dependencies():
    """Test adding step with dependencies."""
    wf = Workflow(name="test")

    step1_id = wf.add_step("step1", lambda **kwargs: 42)
    step2_id = wf.add_step("step2", lambda **kwargs: 100, dependencies=[step1_id])

    assert step1_id in wf.steps[step2_id].dependencies


@pytest.mark.unit
@pytest.mark.fast
def test_add_step_with_metadata():
    """Test adding step with metadata."""
    wf = Workflow(name="test")

    metadata = {"author": "researcher", "version": "1.0"}
    step_id = wf.add_step("compute", lambda **kwargs: 1, metadata=metadata)

    assert wf.steps[step_id].metadata == metadata


@pytest.mark.unit
@pytest.mark.fast
def test_add_multiple_steps():
    """Test adding multiple steps maintains order."""
    wf = Workflow(name="test")

    step1_id = wf.add_step("first", lambda **kwargs: 1)
    step2_id = wf.add_step("second", lambda **kwargs: 2)
    step3_id = wf.add_step("third", lambda **kwargs: 3)

    assert wf.step_order == [step1_id, step2_id, step3_id]


@pytest.mark.unit
@pytest.mark.fast
def test_step_id_generation():
    """Test step ID generation format."""
    wf = Workflow(name="test")

    step_id = wf.add_step("my_step", lambda **kwargs: 1)

    assert step_id.startswith("step_")
    assert "my_step" in step_id


# ============================================================================
# Test: Input/Output Management
# ============================================================================


@pytest.mark.unit
@pytest.mark.fast
def test_set_input():
    """Test setting input value for a step."""
    wf = Workflow(name="test")

    step_id = wf.add_step("compute", lambda x, **kwargs: x * 2)
    wf.set_input(step_id, "x", 10)

    assert wf.steps[step_id].inputs["x"] == 10


@pytest.mark.unit
@pytest.mark.fast
def test_set_input_invalid_step_raises():
    """Test setting input for non-existent step raises ValueError."""
    wf = Workflow(name="test")

    with pytest.raises(ValueError, match=r"Step .* not found"):
        wf.set_input("nonexistent_step", "x", 10)


@pytest.mark.unit
@pytest.mark.fast
def test_get_output_after_execution():
    """Test getting output from completed step."""
    wf = Workflow(name="test")

    def compute(**kwargs):
        return {"result": 42}

    step_id = wf.add_step("compute", compute)
    wf.execute(save_results=False)

    output = wf.get_output(step_id, "result")
    assert output == 42


@pytest.mark.unit
@pytest.mark.fast
def test_get_output_invalid_step_raises():
    """Test getting output from non-existent step raises ValueError."""
    wf = Workflow(name="test")

    with pytest.raises(ValueError, match=r"Step .* not found"):
        wf.get_output("nonexistent_step", "result")


@pytest.mark.unit
@pytest.mark.fast
def test_get_output_before_completion_raises():
    """Test getting output from uncompleted step raises ValueError."""
    wf = Workflow(name="test")

    step_id = wf.add_step("compute", lambda **kwargs: 42)

    with pytest.raises(ValueError, match="has not completed successfully"):
        wf.get_output(step_id, "result")


@pytest.mark.unit
@pytest.mark.fast
def test_get_output_nonexistent_output_raises():
    """Test getting non-existent output raises ValueError."""
    wf = Workflow(name="test")

    step_id = wf.add_step("compute", lambda **kwargs: {"value": 42})
    wf.execute(save_results=False)

    with pytest.raises(ValueError, match=r"Output '.*' not found"):
        wf.get_output(step_id, "nonexistent")


# ============================================================================
# Test: Workflow Execution
# ============================================================================


@pytest.mark.unit
def test_execute_empty_workflow():
    """Test executing workflow with no steps."""
    wf = Workflow(name="empty")

    result = wf.execute(save_results=False)

    assert result.status == WorkflowStatus.COMPLETED
    assert wf.status == WorkflowStatus.COMPLETED


@pytest.mark.unit
def test_execute_single_step():
    """Test executing workflow with single step."""
    wf = Workflow(name="test")

    def compute(**kwargs):  # Accept workflow_context kwarg
        return {"value": 42}

    wf.add_step("compute", compute)

    result = wf.execute(save_results=False)

    assert result.status == WorkflowStatus.COMPLETED
    assert wf.status == WorkflowStatus.COMPLETED


@pytest.mark.unit
def test_execute_multiple_steps():
    """Test executing workflow with multiple steps."""
    wf = Workflow(name="test")

    def step1(**kwargs):
        return {"a": 10}

    def step2(**kwargs):
        return {"b": 20}

    wf.add_step("first", step1)
    wf.add_step("second", step2)

    result = wf.execute(save_results=False)

    assert result.status == WorkflowStatus.COMPLETED
    assert len(result.step_results) == 2


@pytest.mark.unit
def test_execute_with_dependencies():
    """Test executing workflow respects dependencies."""
    wf = Workflow(name="test")

    execution_order = []

    def step1(**kwargs):
        execution_order.append(1)
        return {"value": 10}

    def step2(**kwargs):
        execution_order.append(2)
        return {"value": 20}

    step1_id = wf.add_step("first", step1)
    wf.add_step("second", step2, dependencies=[step1_id])

    wf.execute(save_results=False)

    assert execution_order == [1, 2]


@pytest.mark.unit
def test_execute_tracks_execution_time():
    """Test workflow tracks execution time."""
    wf = Workflow(name="test")

    def compute(**kwargs):
        return 42

    wf.add_step("compute", compute)

    result = wf.execute(save_results=False)

    assert result.execution_time is not None
    assert result.execution_time >= 0


@pytest.mark.unit
def test_execute_sets_start_and_end_times():
    """Test workflow sets start and end times."""
    wf = Workflow(name="test")

    wf.add_step("compute", lambda: 42)

    result = wf.execute(save_results=False)

    assert result.start_time is not None
    assert result.end_time is not None
    assert result.end_time >= result.start_time


# ============================================================================
# Test: Error Handling
# ============================================================================


@pytest.mark.unit
def test_execute_handles_step_failure():
    """Test workflow handles step execution failures."""
    wf = Workflow(name="test")

    def failing_step(**kwargs):
        raise ValueError("Intentional failure")

    wf.add_step("fail", failing_step)

    result = wf.execute(save_results=False)

    assert result.status == WorkflowStatus.FAILED
    assert wf.status == WorkflowStatus.FAILED
    assert result.error_message is not None


@pytest.mark.unit
def test_execute_stops_on_failure():
    """Test workflow stops execution after step failure."""
    wf = Workflow(name="test")

    executed = []

    def step1(**kwargs):
        executed.append(1)
        raise ValueError("Failure")

    def step2(**kwargs):
        executed.append(2)
        return 42

    wf.add_step("first", step1)
    wf.add_step("second", step2)

    wf.execute(save_results=False)

    assert executed == [1]  # Second step should not execute


@pytest.mark.unit
def test_step_stores_error_message():
    """Test failed step stores error message."""
    wf = Workflow(name="test")

    def failing(**kwargs):
        raise ValueError("Test error")

    step_id = wf.add_step("fail", failing)

    wf.execute(save_results=False)

    assert wf.steps[step_id].status == StepStatus.FAILED
    assert wf.steps[step_id].error_message == "Test error"


# ============================================================================
# Test: Step Status Transitions
# ============================================================================


@pytest.mark.unit
def test_step_status_pending_to_completed():
    """Test step status transitions from PENDING to COMPLETED."""
    wf = Workflow(name="test")

    def compute(**kwargs):
        return 42

    step_id = wf.add_step("compute", compute)

    assert wf.steps[step_id].status == StepStatus.PENDING

    wf.execute(save_results=False)

    assert wf.steps[step_id].status == StepStatus.COMPLETED


@pytest.mark.unit
def test_step_status_pending_to_failed():
    """Test step status transitions from PENDING to FAILED on error."""
    wf = Workflow(name="test")

    def failing(**kwargs):
        raise RuntimeError("Error")

    step_id = wf.add_step("fail", failing)

    assert wf.steps[step_id].status == StepStatus.PENDING

    wf.execute(save_results=False)

    assert wf.steps[step_id].status == StepStatus.FAILED


@pytest.mark.unit
def test_workflow_status_transitions():
    """Test workflow status transitions through execution."""
    wf = Workflow(name="test")

    assert wf.status == WorkflowStatus.CREATED

    wf.add_step("compute", lambda **kwargs: 42)  # Accept workflow_context kwarg
    result = wf.execute(save_results=False)

    assert result.status == WorkflowStatus.COMPLETED
    assert wf.status == WorkflowStatus.COMPLETED


# ============================================================================
# Test: Result Collection
# ============================================================================


@pytest.mark.unit
def test_collects_step_outputs():
    """Test workflow collects outputs from all steps."""
    wf = Workflow(name="test")

    def step1(**kwargs):  # Accept workflow_context kwarg
        return {"value": 10}

    def step2(**kwargs):  # Accept workflow_context kwarg
        return {"value": 20}

    wf.add_step("first", step1)
    wf.add_step("second", step2)

    result = wf.execute(save_results=False)

    assert "first_value" in result.outputs
    assert "second_value" in result.outputs
    assert result.outputs["first_value"] == 10
    assert result.outputs["second_value"] == 20


@pytest.mark.unit
def test_handles_non_dict_return_values():
    """Test workflow handles functions returning non-dict values."""
    wf = Workflow(name="test")

    def compute(**kwargs):  # Accept workflow_context kwarg
        return 42

    step_id = wf.add_step("compute", compute)

    wf.execute(save_results=False)

    assert wf.steps[step_id].outputs == {"result": 42}


# ============================================================================
# Test: Execution Order
# ============================================================================


@pytest.mark.unit
def test_execution_order_no_dependencies():
    """Test execution order with no dependencies follows add order."""
    wf = Workflow(name="test")

    wf.add_step("step1", lambda **kwargs: 1)
    wf.add_step("step2", lambda **kwargs: 2)
    wf.add_step("step3", lambda **kwargs: 3)

    execution_order = wf._compute_execution_order()

    assert len(execution_order) == 3


@pytest.mark.unit
def test_execution_order_with_dependencies():
    """Test execution order respects dependencies."""
    wf = Workflow(name="test")

    step1_id = wf.add_step("step1", lambda **kwargs: 1)
    step2_id = wf.add_step("step2", lambda **kwargs: 2)
    wf.add_step("step3", lambda **kwargs: 3, dependencies=[step1_id, step2_id])

    execution_order = wf._compute_execution_order()

    # step3 must come after step1 and step2
    assert execution_order.index(step1_id) < execution_order.index("step_002_step3")
    assert execution_order.index(step2_id) < execution_order.index("step_002_step3")


@pytest.mark.unit
def test_execution_order_circular_dependency_raises():
    """Test circular dependency detection raises ValueError."""
    wf = Workflow(name="test")

    step1_id = wf.add_step("step1", lambda **kwargs: 1)
    step2_id = wf.add_step("step2", lambda **kwargs: 2, dependencies=[step1_id])

    # Manually create circular dependency
    wf.steps[step1_id].dependencies.append(step2_id)

    with pytest.raises(ValueError, match="Circular dependency"):
        wf._compute_execution_order()
