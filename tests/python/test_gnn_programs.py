from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from analysis.figures.gnn_programs import load_program_embeddings, _snapshot, _failed_nodes, _leader_nodes, PROGRAM_FOLDERS


def test_identical_embeddings_have_identical_colours_across_programs(tmp_path):
    # Independently fitted PCAs could map the same vector to different colours.
    for i, folder in enumerate(PROGRAM_FOLDERS.values()):
        path = tmp_path / 'generalization' / folder
        path.mkdir(parents=True)
        pd.DataFrame({'tEp': [95, 103, 108], 'nodeId': [0, 0, 0],
                      'h_0': [0., 1., 2. + i], 'h_1': [1., 0., 1.],
                      'nextWakeUp': [5., 2., 1.]}).to_csv(path / 'gnn_spatiotemporal_embeddings.csv', index=False)
    frames = load_program_embeddings(SimpleNamespace(spatial_dir=tmp_path))
    assert len(frames) == len(PROGRAM_FOLDERS)
    reference = frames['gradient'].pc1.iloc[:2].to_numpy()
    for frame in frames.values():
        np.testing.assert_allclose(frame.pc1.iloc[:2], reference, atol=1e-12)


def test_missing_instant_is_not_silently_replaced():
    frame = pd.DataFrame({'tEp': [95., 96.], 'nodeId': [0, 0]})
    with pytest.raises(ValueError, match='No snapshot'):
        _snapshot(frame, 138.)


def test_failed_marker_uses_removed_node_position_without_marking_survivors():
    before = pd.DataFrame({"tEp": [95., 95.], "nodeId": [0, 1], "x": [5., 10.], "y": [3., 7.]})
    after = before[before.nodeId == 1].assign(tEp=103.)
    frame = pd.concat([before, after], ignore_index=True)
    failed = _failed_nodes(frame, after)
    assert failed.nodeId.tolist() == [0]
    assert failed[["x", "y"]].to_numpy().tolist() == [[5., 3.]]


def test_leader_markers_use_the_election_flag_instead_of_broadcast_values():
    nodes = pd.DataFrame({"nodeId": [10, 20, 30], "gradient": [10., 10., 30.], "isLeader": [1, 0, 0]})
    assert _leader_nodes(nodes).nodeId.tolist() == [10]
