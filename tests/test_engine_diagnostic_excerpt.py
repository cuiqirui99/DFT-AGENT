from vasp_slurm_agent.engine_context import _log_excerpt


def test_root_error_survives_a_long_stack_trace(tmp_path):
    path = tmp_path / "slurm.err"
    path.write_text("Fortran runtime error: Unit number is negative\n" + "stack frame\n" * 2000)
    excerpt = _log_excerpt(path)
    assert "Unit number is negative" in excerpt
    assert "stack frame" in excerpt
    assert len(excerpt) <= 2000
