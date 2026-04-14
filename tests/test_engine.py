import pytest
import numpy as np
from minkowski.engine import MinkowskiEngine, Event, Segment, FrameManager

@pytest.fixture
def engine():
    return MinkowskiEngine()

@pytest.fixture
def fm(engine):
    return FrameManager(engine)

@pytest.fixture(params=[(1, 0),(5, 3),(2, 5),(1, 1),(10, -8)])
def vec(request):
    t, x = request.param
    return np.array([[t], [x]])

@pytest.fixture(params=[0.0, 0.5, -0.8, 0.99, -0.99])
def v(request):
    return request.param

@pytest.fixture
def event(engine, vec):
        idx = engine.add_event(vec[0,0], vec[1,0])
        return idx, Event(engine, idx)
    
class TestMinkowskiEngine:
    def test_causality_classification(self, engine, vec):
        s2_expected = vec[0,0]**2 - vec[1,0]**2
        s2, cau = engine.causal_structure(vec)
        assert np.isclose(s2_expected, s2)
        if s2 > 1e-9: assert cau == "timelike"
        elif s2 < -1e-9: assert cau == "spacelike"
        else: assert cau == "lightlike"

    def test_gamma_limit(self, engine):
        with pytest.raises(ValueError, match="Velocity must be strictly between -1 and 1."):
            engine.lorentz_matrix(1.0)
        with pytest.raises(ValueError, match="Velocity must be strictly between -1 and 1."):
            engine.lorentz_matrix(-1.0)
        with pytest.raises(ValueError, match="Velocity must be strictly between -1 and 1."):
            engine.lorentz_matrix(1.5)

    def test_lorentz_transformation(self, engine, vec, v):
        s2_original, _ = engine.causal_structure(vec)
        vec_prime = engine.boost(vec, v)
        s2_prime, _ = engine.causal_structure(vec_prime)
        expected_vec = engine.lorentz_matrix(v) @ vec
        assert np.allclose(vec_prime, expected_vec)
        assert np.isclose(s2_original, s2_prime, atol=1e-12)
    
    def test_memory_allocation(self, engine):
        initial_capacity = engine._capacity
        assert engine.rest.shape == (2, initial_capacity)
        for i in range(initial_capacity + 1):
            engine.add_event(i, i)
        
        assert engine._next_index == initial_capacity + 1
        assert engine._capacity == initial_capacity * 2
        assert engine.rest.shape == (2, initial_capacity * 2)

        engine.remove_event(0)
        assert np.isnan(engine.rest[:, 0]).all()
        assert not np.isnan(engine.rest[:, 1]).any()
        with pytest.raises(IndexError):
            engine.remove_event(999)
        
    def test_event_management(self, engine):
        event1 = engine.add_event(2.0, 1.0)
        idx2 = engine.add_event(3.0, -1.0)
        event3 = engine.add_event(1.5, 3.5)

        engine.remove_event(idx2)
        assert idx2 in engine._free_indices
        assert engine._next_index == 3

        event2_new = engine.add_event(-1.5, 2.5)
        assert event2_new == idx2
        assert engine._next_index == 3
        assert len(engine._free_indices) == 0

    def test_active_coordinates(self, engine, v):
        event1 = engine.add_event(0.5, -2.0)
        idx2 = engine.add_event(1.5, 1.0)
        event3 = engine.add_event(2.0, 4.0)

        engine.remove_event(idx2)
        active_prime = engine.active_coordinates(v)
        assert active_prime.shape == (2, 2)

        expected_event1 = engine.boost(np.array([[0.5], [-2.0]]), v)
        expected_event3 = engine.boost(np.array([[2.0], [4.0]]), v)
        assert np.allclose(active_prime[:, 0:1], expected_event1)
        assert np.allclose(active_prime[:, 1:2], expected_event3)


class TestEvents:
    def test_event_boost(self, engine, vec, v, event):
        idx, ev = event
        expected_coordinates = engine.boost(vec, v)
        assert ev.index == idx
        assert np.allclose(ev.coordinates(v), expected_coordinates)


class TestSegments:
    def test_segment_interval_invariance(self, engine, vec, v):
        idx1 = engine.add_event(0, 0)
        idx2 = engine.add_event(vec[0,0], vec[1,0])
        seg = Segment(engine, idx1, idx2)

        dt_rest, dx_rest = seg.coordinate_deltas(0.0)
        dt_prime, dx_prime = seg.coordinate_deltas(v)
        s2_rest = dt_rest**2 - dx_rest**2
        s2_prime = dt_prime**2 - dx_prime**2
        assert np.isclose(s2_rest, s2_prime, atol=1e-12)


class TestFrameManager:
    def test_frame_management(self, fm):
        S1 = fm.add_frame(v=0.5)
        S2 = fm.add_frame(v=-0.5)
        assert fm.frames[0].index == 0
        assert S1.index == 1
        assert S2.index == 2
        
        fm.remove_frame(1)
        S3 = fm.add_frame(0.8)
        indices = [f.index for f in fm.frames]
        assert 2 in indices
        assert S3.index == 1

    def test_frame_management_exceptions(self, fm):
        for _ in range(10):
            fm.add_frame(v=0.3)
        with pytest.raises(RuntimeError, match="Maximum number of reference frames reached."):
            fm.add_frame(v=-0.6)
    
        with pytest.raises(ValueError, match="The fundamental rest frame cannot be removed."):
            fm.remove_frame(0)
        with pytest.raises(IndexError, match="The reference frame does not exist."):
            fm.remove_frame(99)