# operators/test_nonparametric.py
"""Tests for operators._nonparametric."""

import abc
import pytest
import numpy as np
import scipy.linalg as la
import scipy.sparse as sparse

import opinf

try:
    from .test_base import _TestOpInfOperator
except ImportError:
    from test_base import _TestOpInfOperator


_module = opinf.operators._nonparametric


class _TestNonparametricOperator(_TestOpInfOperator):

    def get_operator(self, r=None, m=None):
        if r is None:
            return self.Operator()
        return self.Operator(self.get_entries(r, m))

    def get_entries(self, r, m=None):
        d = self.Operator.operator_dimension(r, m)
        return np.random.standard_normal((r, d))

    @abc.abstractmethod
    def test_set_entries(self):
        raise NotImplementedError

    def test_in_model(self, r=3, k=100, m=2):
        """See if we can fit a model with this operator."""
        model = opinf.models.ContinuousModel(operators=[self.get_operator()])

        Q = np.random.random((r, k))
        dQ = np.random.random((r, k))

        if self.has_inputs:
            U = np.random.random((m, k))
            model.fit(states=Q, ddts=dQ, inputs=U)
        else:
            model.fit(states=Q, ddts=dQ, inputs=None)


# No dependence on state or input =============================================
class TestConstantOperator(_TestNonparametricOperator):
    """Test operators._nonparametric.ConstantOperator."""

    Operator = _module.ConstantOperator
    has_inputs = False

    def test_set_entries(self):
        """Test set_entries()."""
        op = self.Operator()

        # Too many dimensions.
        cbad = np.arange(12).reshape((4, 3))
        with pytest.raises(ValueError) as ex:
            op.set_entries(cbad)
        assert ex.value.args[0] == (
            "ConstantOperator entries must be one-dimensional"
        )

        # Case 1: one-dimensional array.
        c = np.arange(12)
        op.set_entries(c)
        assert op.entries is c

        # Case 2: two-dimensional array that can be flattened.
        op.set_entries(c.reshape((-1, 1)))
        assert op.shape == (12,)
        assert op.state_dimension == 12
        assert np.all(op.entries == c)

        op.set_entries(c.reshape((1, -1)))
        assert op.shape == (12,)
        assert op.state_dimension == 12
        assert np.all(op.entries == c)

        # Case 3: r = 1 and c is a scalar.
        c = np.random.random()
        op.set_entries(c)
        assert op.shape == (1,)
        assert op.state_dimension == 1
        assert op.entries[0] == c

    def test_apply(self, k=20):
        """Test apply()/__call__()."""
        op = self.Operator()

        def _test_single(r):
            c = np.random.random(r)
            op.set_entries(c)
            assert np.allclose(op.apply(), c)
            # Evaluation for a single vector.
            q = np.random.random(r)
            assert np.allclose(op.apply(q), op.entries)
            # Vectorized evaluation.
            Q = np.random.random((r, k))
            ccc = np.column_stack([c for _ in range(k)])
            out = op.apply(Q)
            assert out.shape == (r, k)
            assert np.all(out == ccc)

        _test_single(10)
        _test_single(20)
        _test_single(1)

        # Special case: r = 1, scalar q.
        c = np.random.random()
        op.set_entries(c)
        # Evaluation for a single vector.
        q = np.random.random()
        out = op.apply(q)
        assert np.isscalar(out)
        assert out == c
        # Vectorized evaluation.
        Q = np.random.random(k)
        out = op.apply(Q)
        assert out.shape == (k,)
        assert np.all(out == c)

    def test_datablock(self, k=20):
        """Test datablock()."""
        op = self.Operator()
        ones = np.ones((1, k))

        def _test_single(out):
            assert out.shape == ones.shape
            assert np.all(out == ones)

        _test_single(op.datablock(np.random.random((5, k))))
        _test_single(op.datablock(np.random.random((3, k))))
        _test_single(op.datablock(np.random.random(k)))

    def test_operator_dimension(self):
        """Test operator_dimension()."""
        assert self.Operator.operator_dimension() == 1
        assert self.Operator.operator_dimension(4) == 1
        assert self.Operator.operator_dimension(1, 6) == 1


# Dependent on state but not on input =========================================
class TestLinearOperator(_TestNonparametricOperator):
    """Test operators._nonparametric.LinearOperator."""

    Operator = _module.LinearOperator
    has_inputs = False

    def test_set_entries(self):
        """Test set_entries()."""
        op = self.Operator()

        # Too many dimensions.
        Abad = np.arange(12).reshape((2, 2, 3))
        with pytest.raises(ValueError) as ex:
            op.set_entries(Abad)
        assert ex.value.args[0] == (
            "LinearOperator entries must be two-dimensional"
        )

        # Nonsquare.
        Abad = Abad.reshape((4, 3))
        with pytest.raises(ValueError) as ex:
            op.set_entries(Abad)
        assert ex.value.args[0] == (
            "LinearOperator entries must be square (r x r)"
        )

        # Correct square usage.
        A = Abad[:3, :3]
        op.set_entries(A)
        assert op.entries is A
        assert op.state_dimension == 3

        # Special case: r = 1, scalar A.
        a = np.random.random()
        op.set_entries(a)
        assert op.shape == (1, 1)
        assert op.state_dimension == 1
        assert op[0, 0] == a

        # Sparse matrix.
        A = np.random.random((100, 100))
        A[A < 0.95] = 0
        A = sparse.csr_matrix(A)
        op.set_entries(A)
        assert op.state_dimension == 100
        assert op.shape == (100, 100)
        assert sparse.issparse(op.entries)

    def test_apply(self, k=20):
        """Test apply()/__call__()."""
        op = self.Operator()

        def _test_single(r):
            A = np.random.random((r, r))
            op.set_entries(A)
            # Evaluation for a single vector.
            q = np.random.random(r)
            assert np.allclose(op.apply(q), A @ q)
            # Vectorized evaluation.
            Q = np.random.random((r, k))
            assert np.allclose(op.apply(Q), A @ Q)

        _test_single(10)
        _test_single(4)
        _test_single(1)

        # Special case: A is 1x1 and q is a scalar.
        A = np.random.random()
        op.set_entries(A)
        # Evaluation for a single vector.
        q = np.random.random()
        out = op.apply(q)
        assert np.isscalar(out)
        assert np.allclose(out, A * q)
        # Vectorized evaluation.
        Q = np.random.random(k)
        out = op.apply(Q)
        assert out.shape == (k,)
        assert np.allclose(out, A * Q)

    def test_jacobian(self, r=9):
        """Test jacobian()."""
        A = np.random.random((r, r))
        op = self.Operator(A)
        jac = op.jacobian(np.random.random(r))
        assert jac.shape == A.shape
        assert np.all(jac == A)

    def test_datablock(self, m=3, k=20, r=10):
        """Test datablock()."""
        op = self.Operator()
        state_ = np.random.random((r, k))
        input_ = np.random.random((m, k))

        assert op.datablock(state_, input_) is state_
        assert op.datablock(state_, None) is state_

        # Special case: r = 1.
        state_ = np.random.random(k)
        block = op.datablock(state_)
        assert block.shape == (1, k)
        assert np.all(block[0] == state_)

    def test_operator_dimension(self):
        """Test operator_dimension()."""
        assert self.Operator.operator_dimension(2) == 2
        assert self.Operator.operator_dimension(4, 6) == 4


class TestQuadraticOperator(_TestNonparametricOperator):
    """Test operators._nonparametric.QuadraticOperator."""

    Operator = _module.QuadraticOperator
    has_inputs = False

    def test_set_entries(self, r=4):
        """Test set_entries()."""
        op = self.Operator()

        # Too many dimensions.
        Hbad = np.arange(16).reshape((2, 2, 2, 2))
        with pytest.raises(ValueError) as ex:
            op.set_entries(Hbad)
        assert ex.value.args[0] == (
            "QuadraticOperator entries must be two-dimensional"
        )

        # Two-dimensional but invalid shape.
        Hbad = Hbad.reshape((r, r))
        with pytest.raises(ValueError) as ex:
            op.set_entries(Hbad)
        assert ex.value.args[0] == (
            "invalid QuadraticOperator entries dimensions"
        )

        # Special case: r = 1, H a scalar.
        H = np.random.random()
        op.set_entries(H)
        assert op.state_dimension == 1
        assert op.shape == (1, 1)
        assert op.entries[0, 0] == H

        # Full operator, compressed internally.
        H = np.random.random((r, r**2))
        H_ = self.Operator.compress_entries(H)
        op.set_entries(H)
        r2_ = r * (r + 1) // 2
        assert op.state_dimension == r
        assert op.shape == (r, r2_)
        assert np.allclose(op.entries, H_)

        # Three-dimensional tensor.
        op.set_entries(H.reshape((r, r, r)))
        assert op.state_dimension == r
        assert op.shape == (r, r2_)
        assert np.allclose(op.entries, H_)

        # Compressed operator.
        H = np.random.random((r, r2_))
        op.set_entries(H)
        assert op.entries is H
        assert op.state_dimension == r

        # Test _clear().
        op._clear()
        assert op.entries is None
        assert op._mask is None
        assert op._prejac is None
        assert op.state_dimension is None
        assert op.shape is None

    def test_apply(self, k=10, ntrials=10):
        """Test apply()/__call__()."""
        op = self.Operator()

        def _test_single(r):
            H = np.random.random((r, r**2))
            op.set_entries(H)
            for _ in range(ntrials):
                # Evaluation for a single vector.
                q = np.random.random(r)
                evaltrue = H @ np.kron(q, q)
                evalgot = op.apply(q)
                assert evalgot.shape == (r,)
                assert np.allclose(evalgot, evaltrue)
                # Vectorized evaluation.
                Q = np.random.random((r, k))
                evaltrue = H @ la.khatri_rao(Q, Q)
                evalgot = op.apply(Q)
                assert evalgot.shape == (r, k)
                assert np.allclose(evalgot, evaltrue)

        _test_single(5)
        _test_single(2)
        _test_single(1)

        # Special case: r = 1 and q is a scalar.
        H = np.random.random()
        op.set_entries(H)
        for _ in range(ntrials):
            # Evaluation for a single vector.
            q = np.random.random()
            evaltrue = H * q**2
            evalgot = op.apply(q)
            assert np.isscalar(evalgot)
            assert np.isclose(evalgot, evaltrue)
            # Vectorized evaluation.
            Q = np.random.random(k)
            evaltrue = H * Q**2
            evalgot = op.apply(Q)
            assert evalgot.shape == (k,)
            assert np.allclose(evalgot, evaltrue)

    def test_jacobian(self, r=5, ntrials=10):
        """Test jacobian()."""
        H = np.random.random((r, r**2))
        op = self.Operator(H)
        assert op._prejac is None

        # r > 1
        Id = np.eye(r)
        for _ in range(ntrials):
            q = np.random.random(r)
            jac_true = H @ (np.kron(Id, q) + np.kron(q, Id)).T
            jac = op.jacobian(q)
            assert jac.shape == (r, r)
            assert np.allclose(jac, jac_true)

        # Special case: r = 1
        H = np.random.random()
        op.set_entries(H)
        for _ in range(ntrials):
            q = np.random.random()
            jac_true = 2 * H * q
            jac = op.jacobian(q)
            assert jac.shape == (1, 1)
            assert np.isclose(jac[0, 0], jac_true)

    def test_datablock(self, k=20, r=10):
        """Test datablock()."""
        op = self.Operator()
        state_ = np.random.random((r, k))
        r2_ = r * (r + 1) // 2

        block = op.datablock(state_)
        assert block.shape == (r2_, k)
        op.entries = np.random.random((r, r2_))
        mult = op.entries @ block
        evald = op.apply(state_)
        assert mult.shape == evald.shape
        assert np.allclose(mult, evald)

        # Special case: r = 1.
        state_ = state_[0]
        block = op.datablock(state_)
        assert block.shape == (1, k)
        op.entries = np.random.random()
        mult = op.entries[0, 0] * block[0]
        evald = op.apply(state_)
        assert mult.shape == evald.shape
        assert np.allclose(mult, evald)

    def test_operator_dimension(self):
        """Test operator_dimension()."""
        assert self.Operator.operator_dimension(1) == 1
        assert self.Operator.operator_dimension(3) == 6
        assert self.Operator.operator_dimension(5, 7) == 15

    def test_ckron(self, n_tests=20):
        """Test ckron()."""

        def _check(q, q2):
            for i in range(len(q)):
                assert np.allclose(
                    q2[i * (i + 1) // 2 : (i + 1) * (i + 2) // 2],
                    q[i] * q[: i + 1],
                )

        for r in np.random.randint(2, 10, n_tests):
            q = np.random.random(r)
            q2 = self.Operator.ckron(q)
            r2 = r * (r + 1) // 2
            assert q2.shape == (r2,)
            _check(q, q2)

            k = np.random.randint(1, 10)
            Q = np.random.random((r, k))
            Q2 = self.Operator.ckron(Q)
            assert Q2.shape == (r2, k)
            _check(Q, Q2)

    def test_ckron_indices(self, n_tests=20):
        """Test ckron_indices()."""
        # Manufactured test.
        mask = self.Operator.ckron_indices(4)
        assert np.all(
            mask
            == np.array(
                [
                    [0, 0],
                    [1, 0],
                    [1, 1],
                    [2, 0],
                    [2, 1],
                    [2, 2],
                    [3, 0],
                    [3, 1],
                    [3, 2],
                    [3, 3],
                ],
                dtype=int,
            )
        )
        submask = self.Operator.ckron_indices(3)
        assert np.allclose(submask, mask[: submask.shape[0]])

        # Random test.
        for _ in range(n_tests):
            r = np.random.randint(2, 10)
            _r2 = r * (r + 1) // 2
            mask = self.Operator.ckron_indices(r)
            assert mask.shape == (_r2, 2)
            assert mask.sum(axis=0)[0] == sum(i * (i + 1) for i in range(r))
            q = np.random.random(r)
            assert np.allclose(
                np.prod(q[mask], axis=1), self.Operator.ckron(q)
            )

    def test_compress_entries(self, n_tests=20):
        """Test compress_entries()."""
        # Try with bad second dimension.
        r = 5
        r2bad = r**2 + 1
        H = np.empty((r, r2bad))
        with pytest.raises(ValueError) as ex:
            self.Operator.compress_entries(H)
        assert ex.value.args[0] == (
            f"invalid shape (a, r2) = {(r, r2bad)} "
            "with r2 not a perfect square"
        )

        # One-dimensional H (r = 1).
        Hc = self.Operator.compress_entries([5])
        assert Hc.shape == (1, 1)
        assert Hc[0, 0] == 5

        # Random tests.
        for r in np.random.randint(2, 10, n_tests):
            # Check dimensions.
            a = np.random.randint(2, 10)
            H = np.random.random((a, r**2))
            r2 = r * (r + 1) // 2
            Hc = self.Operator.compress_entries(H)
            assert Hc.shape == (a, r2)

            # Check that Hc(q^2) == H(q ⊗ q).
            for _ in range(5):
                q = np.random.random(r)
                Hq2 = H @ np.kron(q, q)
                assert np.allclose(Hq2, Hc @ self.Operator.ckron(q))

            # Check that expand_entries() and compress_quadrati()
            # are inverses up to symmetry.
            H2 = self.Operator.expand_entries(Hc)
            Ht = H.reshape((a, r, r))
            H2sym = np.reshape([(Hti + Hti.T) / 2 for Hti in Ht], H.shape)
            assert np.allclose(H2, H2sym)

    def test_expand_entries(self, n_tests=20):
        """Test expand_entries()."""
        # Try with bad second dimension.
        r = 5
        r2bad = (r * (r + 1) // 2) + 1
        Hc = np.empty((r, r2bad))
        with pytest.raises(ValueError) as ex:
            self.Operator.expand_entries(Hc)
        assert ex.value.args[0] == (
            f"invalid shape (a, r2) = {(r, r2bad)} "
            "with r2 != r(r+1)/2 for any integer r"
        )

        # One-dimensional H (r = 1).
        H = self.Operator.expand_entries([5])
        assert H.shape == (1, 1)
        assert H[0, 0] == 5

        # Random tests.
        for r in np.random.randint(2, 10, n_tests):
            # Check dimensions.
            a = np.random.randint(2, 10)
            Hc = np.random.random((a, r * (r + 1) // 2))
            H = self.Operator.expand_entries(Hc)
            assert H.shape == (a, r**2)

            # Check that Hc(q^2) == H(q ⊗ q).
            for _ in range(5):
                q = np.random.random(r)
                Hq2 = H @ np.kron(q, q)
                assert np.allclose(Hq2, Hc @ self.Operator.ckron(q))

            # Check that expand_entries() and compress_entries() are inverses.
            Hc2 = self.Operator.compress_entries(H)
            assert np.allclose(Hc2, Hc)


class TestCubicOperator(_TestNonparametricOperator):
    """Test operators._nonparametric.CubicOperator."""

    Operator = _module.CubicOperator
    has_inputs = False

    def test_set_entries(self, r=4):
        """Test set_entries()."""
        op = self.Operator()

        # Too many dimensions.
        Gbad = np.arange(4).reshape((1, 2, 1, 2))
        with pytest.raises(ValueError) as ex:
            op.set_entries(Gbad)
        assert ex.value.args[0] == (
            "CubicOperator entries must be two-dimensional"
        )

        # Two-dimensional but invalid shape.
        Gbad = np.random.random((3, 8))
        with pytest.raises(ValueError) as ex:
            op.set_entries(Gbad)
        assert ex.value.args[0] == "invalid CubicOperator entries dimensions"

        # Special case: r = 1
        G = np.random.random((1, 1))
        op.set_entries(G)
        assert op.shape == (1, 1)
        assert np.allclose(op.entries, G)

        # Full operator, compressed internally.
        G = np.random.random((r, r**3))
        G_ = self.Operator.compress_entries(G)
        op.set_entries(G)
        r3_ = r * (r + 1) * (r + 2) // 6
        assert op.shape == (r, r3_)
        assert np.allclose(op.entries, G_)

        # Three-dimensional tensor.
        op.set_entries(G.reshape((r, r, r, r)))
        assert op.shape == (r, r3_)
        assert np.allclose(op.entries, G_)

        # Compressed operator.
        G = np.random.random((r, r3_))
        op.set_entries(G)
        assert op.entries is G

        # Test _clear().
        op._clear()
        assert op.entries is None
        assert op._mask is None
        assert op._prejac is None

    def test_apply(self, k=20, ntrials=10):
        """Test apply()/__call__()."""
        op = self.Operator()

        def _test_single(r):
            G = np.random.random((r, r**3))
            op.set_entries(G)
            for _ in range(ntrials):
                # Evaluation for a single vector.
                q = np.random.random(r)
                evaltrue = G @ np.kron(np.kron(q, q), q)
                evalgot = op.apply(q)
                assert np.allclose(evalgot, evaltrue)
                # Vectorized evaluation.
                Q = np.random.random((r, k))
                evaltrue = G @ la.khatri_rao(Q, la.khatri_rao(Q, Q))
                evalgot = op.apply(Q)
                assert evalgot.shape == (r, k)
                assert np.allclose(evalgot, evaltrue)

        _test_single(5)
        _test_single(3)
        _test_single(1)

        # Special case: r = 1 and q is a scalar.
        G = np.random.random()
        op.set_entries(G)
        for _ in range(ntrials):
            # Evaluation for a single vector.
            q = np.random.random()
            evaltrue = G * q**3
            evalgot = op.apply(q)
            assert np.isscalar(evalgot)
            assert np.allclose(evalgot, evaltrue)
            # Vectorized evaluation.
            Q = np.random.random(k)
            evaltrue = G * Q**3
            evalgot = op.apply(Q)
            assert evalgot.shape == (k,)
            assert np.allclose(evalgot, evaltrue)

    def test_jacobian(self, r=5, ntrials=10):
        """Test jacobian()."""
        G = np.random.random((r, r**3))
        op = self.Operator(G)
        assert op._prejac is None

        Id = np.eye(r)
        for _ in range(ntrials):
            q = np.random.random(r)
            qId = np.kron(q, Id)
            Idq = np.kron(Id, q)
            qqId = np.kron(q, qId)
            qIdq = np.kron(qId, q)
            Idqq = np.kron(Idq, q)
            jac_true = G @ (Idqq + qIdq + qqId).T
            jac = op.jacobian(q)
            assert jac.shape == (r, r)
            assert np.allclose(jac, jac_true)

        # Special case: r = 1
        G = np.random.random()
        op.set_entries(G)
        for _ in range(ntrials):
            q = np.random.random()
            jac_true = 3 * G * q**2
            jac = op.jacobian(q)
            assert jac.shape == (1, 1)
            assert np.isclose(jac[0, 0], jac_true)

    def test_datablock(self, k=20, r=10):
        """Test datablock()."""
        op = self.Operator()
        state_ = np.random.random((r, k))
        r3_ = r * (r + 1) * (r + 2) // 6

        # More thorough tests elsewhere for ckron().
        block = op.datablock(state_)
        assert block.shape == (r3_, k)
        op.entries = np.random.random((r, r3_))
        mult = op.entries @ block
        evald = op.apply(state_)
        assert mult.shape == evald.shape
        assert np.allclose(mult, evald)

        # Special case: r = 1.
        state_ = state_[0]
        block = op.datablock(state_)
        assert block.shape == (1, k)
        op.entries = np.random.random()
        mult = op.entries[0, 0] * block[0]
        evald = op.apply(state_)
        assert mult.shape == evald.shape
        assert np.allclose(mult, evald)

    def test_operator_dimension(self):
        """Test operator_dimension()."""
        assert self.Operator.operator_dimension(1) == 1
        assert self.Operator.operator_dimension(3) == 10
        assert self.Operator.operator_dimension(5, 2) == 35

    def test_ckron(self, n_tests=20):
        """Test ckron()."""

        def _check(q, q3):
            for i in range(len(q)):
                assert np.allclose(
                    q3[
                        i
                        * (i + 1)
                        * (i + 2)
                        // 6 : (i + 1)
                        * (i + 2)
                        * (i + 3)
                        // 6
                    ],
                    q[i] * TestQuadraticOperator.Operator.ckron(q[: i + 1]),
                )

        for r in np.random.randint(2, 10, n_tests):
            q = np.random.random(r)
            q3 = self.Operator.ckron(q)
            r3 = r * (r + 1) * (r + 2) // 6
            assert q3.shape == (r3,)
            _check(q, q3)

            k = np.random.randint(1, 10)
            Q = np.random.random((r, k))
            Q3 = self.Operator.ckron(Q)
            assert Q3.shape == (r3, k)
            _check(Q, Q3)

    def test_ckron_indices(self, n_tests=20):
        """Test ckron_indices()."""
        # Manufactured test.
        mask = self.Operator.ckron_indices(2)
        assert np.all(
            mask
            == np.array(
                [[0, 0, 0], [1, 0, 0], [1, 1, 0], [1, 1, 1]],
                dtype=int,
            )
        )

        # Random tests.
        for _ in range(n_tests):
            r = np.random.randint(2, 10)
            mask = self.Operator.ckron_indices(r)
            _r3 = r * (r + 1) * (r + 2) // 6
            mask = self.Operator.ckron_indices(r)
            assert mask.shape == (_r3, 3)
            q = np.random.random(r)
            assert np.allclose(
                np.prod(q[mask], axis=1), self.Operator.ckron(q)
            )

    def test_compress_entries(self, n_tests=20):
        """Test compress_entries()."""
        # Try with bad second dimension.
        r = 5
        r3bad = r**3 + 1
        G = np.empty((r, r3bad))
        with pytest.raises(ValueError) as ex:
            self.Operator.compress_entries(G)
        assert ex.value.args[0] == (
            f"invalid shape (a, r3) = {(r, r3bad)} "
            "with r3 not a perfect cube"
        )

        # One-dimensional G (r = 1).
        Gc = self.Operator.compress_entries([6])
        assert Gc.shape == (1, 1)
        assert Gc[0, 0] == 6

        # Random tests.
        for r in np.random.randint(2, 10, n_tests):
            # Check dimensions.
            a = np.random.randint(2, 10)
            G = np.random.random((a, r**3))
            r2 = r * (r + 1) * (r + 2) // 6
            Gc = self.Operator.compress_entries(G)
            assert Gc.shape == (a, r2)

            # Check that Gc(q^3) == G(q ⊗ q ⊗ q).
            for _ in range(5):
                q = np.random.random(r)
                Gq3 = G @ np.kron(q, np.kron(q, q))
                assert np.allclose(Gq3, Gc @ self.Operator.ckron(q))

    def test_expand_entries(self, n_tests=20):
        """Test expand_entries()."""
        # Try with bad second dimension.
        r = 5
        r3bad = (r * (r + 1) * (r + 2) // 6) + 1
        Gc = np.empty((r, r3bad))
        with pytest.raises(ValueError) as ex:
            self.Operator.expand_entries(Gc)
        assert ex.value.args[0] == (
            f"invalid shape (a, r3) = {(r, r3bad)} "
            "with r3 != r(r+1)(r+2)/6 for any integer r"
        )

        # One-dimensional G (r = 1).
        G = self.Operator.expand_entries([5])
        assert G.shape == (1, 1)
        assert G[0, 0] == 5

        # Random tests.
        for r in np.random.randint(2, 10, n_tests):
            # Check dimensions.
            a = np.random.randint(2, 10)
            Gc = np.random.random((a, r * (r + 1) * (r + 2) // 6))
            G = self.Operator.expand_entries(Gc)
            assert G.shape == (a, r**3)

            # Check that Gc[q^3] == G[q ⊗ q ⊗ q].
            for _ in range(5):
                q = np.random.random(r)
                Gq3 = G @ np.kron(q, np.kron(q, q))
                assert np.allclose(Gq3, Gc @ self.Operator.ckron(q))

            # Check that expand_entries() and compress_entries() are inverses.
            Gc2 = self.Operator.compress_entries(G)
            assert np.allclose(Gc2, Gc)


class TestQuarticOperator(_TestNonparametricOperator):

    Operator = _module.QuarticOperator
    has_inputs = False

    def test_set_entries(self, r=4):
        """Test set_entries()."""
        op = self.Operator()

        # Too many dimensions.
        Qbad = np.arange(4).reshape((1, 2, 1, 2, 1))
        with pytest.raises(ValueError) as ex:
            op.set_entries(Qbad)
        assert ex.value.args[0] == (
            "QuarticOperator entries must be two-dimensional"
        )

        # Two-dimensional but invalid shape.
        Gbad = np.random.random((3, 8))
        with pytest.raises(ValueError) as ex:
            op.set_entries(Gbad)
        assert ex.value.args[0] == "invalid QuarticOperator entries dimensions"

        # Special case: r = 1
        G = np.random.random((1, 1))
        op.set_entries(G)
        assert op.shape == (1, 1)
        assert np.allclose(op.entries, G)

        # Full operator, compressed internally.
        G = np.random.random((r, r**4))
        G_ = self.Operator.compress_entries(G)
        op.set_entries(G)
        r4_ = r * (r + 1) * (r + 2) * (r + 3) // 24
        assert op.shape == (r, r4_)
        assert np.allclose(op.entries, G_)

        # Three-dimensional tensor.
        op.set_entries(G.reshape((r, r, r, r, r)))
        assert op.shape == (r, r4_)
        assert np.allclose(op.entries, G_)

        # Compressed operator.
        G = np.random.random((r, r4_))
        op.set_entries(G)
        assert op.entries is G

        # Test _clear().
        op._clear()
        assert op.entries is None
        assert op._mask is None
        assert op._prejac is None

    def test_apply(self, k=20, ntrials=10):
        """Test apply()/__call__()."""
        op = self.Operator()

        def _test_single(r):
            G = np.random.random((r, r**4))
            op.set_entries(G)
            for _ in range(ntrials):
                # Evaluation for a single vector.
                q = np.random.random(r)
                evaltrue = G @ np.kron(np.kron(np.kron(q, q), q), q)
                evalgot = op.apply(q)
                assert np.allclose(evalgot, evaltrue)
                # Vectorized evaluation.
                Q = np.random.random((r, k))
                evaltrue = G @ la.khatri_rao(
                    Q, la.khatri_rao(Q, la.khatri_rao(Q, Q))
                )
                evalgot = op.apply(Q)
                assert evalgot.shape == (r, k)
                assert np.allclose(evalgot, evaltrue)

        _test_single(5)
        _test_single(3)
        _test_single(1)

        # Special case: r = 1 and q is a scalar.
        G = np.random.random()
        op.set_entries(G)
        for _ in range(ntrials):
            # Evaluation for a single vector.
            q = np.random.random()
            evaltrue = G * q**4
            evalgot = op.apply(q)
            assert np.isscalar(evalgot)
            assert np.allclose(evalgot, evaltrue)
            # Vectorized evaluation.
            Q = np.random.random(k)
            evaltrue = G * Q**4
            evalgot = op.apply(Q)
            assert evalgot.shape == (k,)
            assert np.allclose(evalgot, evaltrue)

    def test_jacobian(self, r=5, ntrials=10):
        """Test jacobian()."""
        G = np.random.random((r, r**4))
        op = self.Operator(G)
        assert op._prejac is None

        # TODO: DONE TO HERE!
        Id = np.eye(r)
        for _ in range(ntrials):
            q = np.random.random(r)
            Idq = np.kron(Id, q)
            qId = np.kron(q, Id)
            Idqq = np.kron(Idq, q)
            qqId = np.kron(q, qId)
            Idqqq = np.kron(Idqq, q)
            qIdqq = np.kron(q, Idqq)
            qqIdq = np.kron(qqId, q)
            qqqId = np.kron(q, qqId)
            jac_true = G @ (Idqqq + qIdqq + qqIdq + qqqId).T
            jac = op.jacobian(q)
            assert jac.shape == (r, r)
            assert np.allclose(jac, jac_true)

        # Special case: r = 1
        G = np.random.random()
        op.set_entries(G)
        for _ in range(ntrials):
            q = np.random.random()
            jac_true = 4 * G * q**3
            jac = op.jacobian(q)
            assert jac.shape == (1, 1)
            assert np.isclose(jac[0, 0], jac_true)

    def test_datablock(self, k=20, r=10):
        """Test datablock()."""
        op = self.Operator()
        state_ = np.random.random((r, k))
        r4_ = r * (r + 1) * (r + 2) * (r + 3) // 24

        # More thorough tests elsewhere for ckron().
        block = op.datablock(state_)
        assert block.shape == (r4_, k)
        op.entries = np.random.random((r, r4_))
        mult = op.entries @ block
        evald = op.apply(state_)
        assert mult.shape == evald.shape
        assert np.allclose(mult, evald)

        # Special case: r = 1.
        state_ = state_[0]
        block = op.datablock(state_)
        assert block.shape == (1, k)
        op.entries = np.random.random()
        mult = op.entries[0, 0] * block[0]
        evald = op.apply(state_)
        assert mult.shape == evald.shape
        assert np.allclose(mult, evald)

    def test_operator_dimension(self):
        """Test operator_dimension()."""
        assert self.Operator.operator_dimension(1) == 1
        assert self.Operator.operator_dimension(3) == 15
        assert self.Operator.operator_dimension(5, 2) == 70

    def test_ckron(self, n_tests=20):
        """Test ckron()."""

        def _check(q, q4):
            for i in range(len(q)):
                assert np.allclose(
                    q4[
                        i
                        * (i + 1)
                        * (i + 2)
                        * (i + 3)
                        // 24 : (i + 1)
                        * (i + 2)
                        * (i + 3)
                        * (i + 4)
                        // 24
                    ],
                    q[i] * TestCubicOperator.Operator.ckron(q[: i + 1]),
                )

        for r in np.random.randint(2, 10, n_tests):
            q = np.random.random(r)
            q4 = self.Operator.ckron(q)
            r4 = r * (r + 1) * (r + 2) * (r + 3) // 24
            assert q4.shape == (r4,)
            _check(q, q4)

            k = np.random.randint(1, 10)
            Q = np.random.random((r, k))
            Q4 = self.Operator.ckron(Q)
            assert Q4.shape == (r4, k)
            _check(Q, Q4)

    def test_ckron_indices(self, n_tests=20):
        """Test ckron_indices()."""
        # Manufactured test.
        mask = self.Operator.ckron_indices(2)
        assert np.all(
            mask
            == np.array(
                [
                    [0, 0, 0, 0],
                    [1, 0, 0, 0],
                    [1, 1, 0, 0],
                    [1, 1, 1, 0],
                    [1, 1, 1, 1],
                ],
                dtype=int,
            )
        )

        # Random tests.
        for _ in range(n_tests):
            r = np.random.randint(2, 10)
            mask = self.Operator.ckron_indices(r)
            _r4 = r * (r + 1) * (r + 2) * (r + 3) // 24
            mask = self.Operator.ckron_indices(r)
            assert mask.shape == (_r4, 4)
            q = np.random.random(r)
            assert np.allclose(
                np.prod(q[mask], axis=1), self.Operator.ckron(q)
            )

    def test_compress_entries(self, n_tests=20):
        """Test compress_entries()."""
        # Try with bad second dimension.
        r = 5
        r4bad = r**4 + 1
        G = np.empty((r, r4bad))
        with pytest.raises(ValueError) as ex:
            self.Operator.compress_entries(G)
        assert ex.value.args[0] == (
            f"invalid shape (a, r4) = {(r, r4bad)} "
            "with r4 not a perfect quartic"
        )

        # One-dimensional G (r = 1).
        Gc = self.Operator.compress_entries([6])
        assert Gc.shape == (1, 1)
        assert Gc[0, 0] == 6

        # Random tests.
        for r in np.random.randint(2, 10, n_tests):
            # Check dimensions.
            a = np.random.randint(2, 10)
            G = np.random.random((a, r**4))
            r4 = r * (r + 1) * (r + 2) * (r + 3) // 24
            Gc = self.Operator.compress_entries(G)
            assert Gc.shape == (a, r4)

            # Check that Gc(q^4) == G(q ⊗ q ⊗ q ⊗ q).
            for _ in range(5):
                q = np.random.random(r)
                Gq4 = G @ np.kron(q, np.kron(q, np.kron(q, q)))
                assert np.allclose(Gq4, Gc @ self.Operator.ckron(q))

    def test_expand_entries(self, n_tests=20):
        """Test expand_entries()."""
        # Try with bad second dimension.
        r = 5
        r4bad = (r * (r + 1) * (r + 2) * (r + 3) // 24) + 1
        Gc = np.empty((r, r4bad))
        with pytest.raises(ValueError) as ex:
            self.Operator.expand_entries(Gc)
        assert ex.value.args[0] == (
            f"invalid shape (a, r4) = {(r, r4bad)} "
            "with r4 != r(r+1)(r+2)(r+3)/24 for any integer r"
        )

        # One-dimensional G (r = 1).
        G = self.Operator.expand_entries([5])
        assert G.shape == (1, 1)
        assert G[0, 0] == 5

        # Random tests.
        for r in np.random.randint(2, 10, n_tests):
            # Check dimensions.
            a = np.random.randint(2, 10)
            Gc = np.random.random((a, r * (r + 1) * (r + 2) * (r + 3) // 24))
            G = self.Operator.expand_entries(Gc)
            assert G.shape == (a, r**4)

            # Check that Gc[q^4] == G[q ⊗ q ⊗ q ⊗ q].
            for _ in range(5):
                q = np.random.random(r)
                Gq4 = G @ np.kron(q, np.kron(q, np.kron(q, q)))
                assert np.allclose(Gq4, Gc @ self.Operator.ckron(q))

            # Check that expand_entries() and compress_entries() are inverses.
            Gc2 = self.Operator.compress_entries(G)
            assert np.allclose(Gc2, Gc)


# Dependent on input but not on state =========================================
class TestInputOperator(_TestNonparametricOperator):
    """Test operators._nonparametric.InputOperator."""

    Operator = _module.InputOperator
    has_inputs = True

    def test_set_entries(self):
        """Test set_entries()."""
        op = self.Operator()

        # Too many dimensions.
        Bbad = np.arange(12).reshape((2, 2, 3))
        with pytest.raises(ValueError) as ex:
            op.set_entries(Bbad)
        assert ex.value.args[0] == (
            "InputOperator entries must be two-dimensional"
        )

        # Nonsquare is OK.
        B = Bbad.reshape((4, 3))
        op.set_entries(B)
        assert op.entries is B
        assert op.input_dimension == 3

        # Special case: r > 1, m = 1
        B = np.random.random(5)
        op.set_entries(B)
        assert op.shape == (5, 1)
        assert op.input_dimension == 1
        assert np.allclose(op.entries[:, 0], B)

        # Special case: r = 1, m > 1
        B = np.random.random((1, 3))
        op.set_entries(B)
        assert op.shape == (1, 3)
        assert np.allclose(op.entries, B)

        # Special case: r = 1, m = 1 (scalar B).
        b = np.random.random()
        op.set_entries(b)
        assert op.shape == (1, 1)
        assert op[0, 0] == b

    def test_apply(self, k=20):
        """Test apply()/__call__()."""
        op = self.Operator()

        def _test_single(r, m):
            B = np.random.random((r, m))
            op.set_entries(B)
            # Evaluation for a single vector.
            q = np.random.random(r)
            u = np.random.random(m)
            evaltrue = B @ u
            evalgot = op.apply(q, u)
            assert evalgot.shape == (r,)
            assert np.allclose(evalgot, evaltrue)
            # Vectorized evaluation.
            Q = np.random.random((r, k))
            U = np.random.random((m, k))
            evaltrue = B @ U
            evalgot = op.apply(Q, U)
            assert evalgot.shape == (r, k)
            assert np.allclose(op.apply(Q, U), B @ U)

        _test_single(10, 2)
        _test_single(2, 5)
        _test_single(3, 1)
        _test_single(1, 4)
        _test_single(1, 1)

        # Special case: B is 1x1 and u is a scalar.
        B = np.random.random()
        op.set_entries(B)
        # Evaluation for a single vector.
        q = np.random.random()
        u = np.random.random()
        out = op.apply(q, u)
        assert np.isscalar(out)
        assert np.allclose(out, B * u)
        # Vectorized evaluation.
        U = np.random.random(k)
        out = op.apply(None, U)
        assert out.shape == (k,)
        assert np.allclose(out, B * U)

        # Special case: B is rx1, r>1, and u is a scalar.
        r = 10
        B = np.random.random(r)
        op.set_entries(B)
        # Evaluation for a single vector.
        q = np.random.random(r)
        u = np.random.random()
        out = op.apply(q, u)
        assert out.shape == (r,)
        assert np.allclose(out, B * u)
        # Vectorized evaluation.
        U = np.random.random(k)
        out = op.apply(None, U)
        assert out.shape == (r, k)
        assert np.allclose(out, np.column_stack([B * u for u in U]))

    def test_datablock(self, m=3, k=20, r=10):
        """Test datablock()."""
        op = self.Operator()
        state_ = np.random.random((r, k))
        input_ = np.random.random((m, k))

        assert op.datablock(state_, input_) is input_
        assert op.datablock(None, input_) is input_

        # Special case: m = 1.
        input_ = input_[0]
        block = op.datablock(state_, input_)
        assert block.shape == (1, k)
        assert np.all(block[0] == input_)

    def test_operator_dimension(self):
        """Test operator_dimension()."""
        assert self.Operator.operator_dimension(1, 3) == 3
        assert self.Operator.operator_dimension(3, 8) == 8
        assert self.Operator.operator_dimension(5, 2) == 2


# Dependent on state and input ================================================
class TestStateInputOperator(_TestNonparametricOperator):
    """Test operators._nonparametric.StateInputOperator."""

    Operator = _module.StateInputOperator
    has_inputs = True

    def test_set_entries(self):
        """Test set_entries()."""
        op = self.Operator()

        # Too many dimensions.
        Nbad = np.arange(12).reshape((2, 2, 3))
        with pytest.raises(ValueError) as ex:
            op.set_entries(Nbad)
        assert ex.value.args[0] == (
            "StateInputOperator entries must be two-dimensional"
        )

        # Two-dimensional but invalid shape.
        Nbad = np.random.random((3, 7))
        with pytest.raises(ValueError) as ex:
            op.set_entries(Nbad)
        assert ex.value.args[0] == (
            "invalid StateInputOperator entries dimensions"
        )

        # Correct dimensions.
        r, m = 5, 3
        N = np.random.random((r, r * m))
        op.set_entries(N)
        assert op.entries is N
        assert op.input_dimension == m

        # Special case: r = 1, m = 1 (scalar B).
        n = np.random.random()
        op.set_entries(n)
        assert op.shape == (1, 1)
        assert op[0, 0] == n
        assert op.input_dimension == 1

    def test_apply(self, k=20):
        """Test apply()/__call__()."""
        op = self.Operator()

        def _test_single(r, m):
            N = np.random.random((r, r * m))
            op.set_entries(N)
            # Evaluation for a single vector.
            q = np.random.random(r)
            u = np.random.random(m)
            evaltrue = N @ np.kron(u, q)
            evalgot = op.apply(q, u)
            assert evalgot.shape == (r,)
            assert np.allclose(evalgot, evaltrue)
            # Vectorized evaluation.
            Q = np.random.random((r, k))
            U = np.random.random((m, k))
            evaltrue = N @ la.khatri_rao(U, Q)
            evalgot = op.apply(Q, U)
            assert evalgot.shape == (r, k)
            assert np.allclose(evalgot, evaltrue)

        _test_single(10, 20)
        _test_single(2, 6)
        _test_single(3, 3)
        _test_single(2, 1)
        _test_single(1, 4)
        _test_single(1, 1)

        # Special case: N is 1x1 and q, u are scalars.
        N = np.random.random()
        op.set_entries(N)
        # Evaluation for a single vector.
        q = np.random.random()
        u = np.random.random()
        out = op.apply(q, u)
        assert np.isscalar(out)
        assert np.allclose(out, N * u * q)
        # Vectorized evaluation.
        Q = np.random.random(k)
        U = np.random.random(k)
        out = op.apply(Q, U)
        assert out.shape == (k,)
        assert np.allclose(out, N * U * Q)

        # Special case: N is rxr, r>1, and u is a scalar.
        r = 10
        N = np.random.random((r, r))
        op.set_entries(N)
        # Evaluation for a single vector.
        q = np.random.random(r)
        u = np.random.random()
        out = op.apply(q, u)
        assert out.shape == (r,)
        assert np.allclose(out, (N @ q) * u)
        # Vectorized evaluation.
        Q = np.random.random((r, k))
        U = np.random.random(k)
        out = op.apply(Q, U)
        assert out.shape == (r, k)
        assert np.allclose(
            out, np.column_stack([N @ q * u for q, u in zip(Q.T, U)])
        )

    def test_jacobian(self, r=9, m=4, ntrials=10):
        """Test jacobian()."""
        Ns = [np.random.random((r, r)) for _ in range(m)]
        N = np.hstack(Ns)
        op = self.Operator(N)

        with pytest.raises(ValueError) as ex:
            op.jacobian(np.random.random(r), np.random.random(m - 1))
        assert ex.value.args[0] == "invalid input_ shape"

        for _ in range(ntrials):
            q = np.random.random(r)
            u = np.random.random(m)
            jac_true = sum([Ni * uu for Ni, uu in zip(Ns, u)])
            jac = op.jacobian(q, u)
            assert jac.shape == (r, r)
            assert np.allclose(jac, jac_true)

        # Special case: r = 1, m > 1, q is a scalar.
        N = np.random.random((1, m))
        op.set_entries(N)
        for _ in range(ntrials):
            q = np.random.random()
            u = np.random.random(m)
            jac_true = N @ u
            jac = op.jacobian(q, u)
            assert jac.shape == (1, 1)
            assert np.allclose(jac, jac_true)

        # Special case: r > 1, m = 1, u is a scalar.
        N = np.random.random((r, r))
        op.set_entries(N)
        for _ in range(ntrials):
            q = np.random.random(r)
            u = np.random.random()
            jac_true = N * u
            jac = op.jacobian(q, u)
            assert jac.shape == (r, r)
            assert np.allclose(jac, jac_true)

        # Special case: r = m = 1, q and u are scalars.
        N = np.random.random()
        op.set_entries(N)
        for _ in range(ntrials):
            q = np.random.random()
            u = np.random.random()
            jac_true = N * u
            jac = op.jacobian(q, u)
            assert jac.shape == (1, 1)
            assert np.isclose(jac[0, 0], jac_true)

    def test_datablock(self, m=3, k=20, r=10):
        """Test datablock()."""
        op = self.Operator()
        state_ = np.random.random((r, k))
        input_ = np.random.random((m, k))
        rm = r * m

        block = op.datablock(state_, input_)
        assert block.shape == (rm, k)
        op.entries = np.random.random((r, rm))
        mult = op.entries @ block
        evald = op.apply(state_, input_)
        assert mult.shape == evald.shape
        assert np.allclose(mult, evald)

        # Special case: r = 1, m > 1
        state_ = state_[0]
        block = op.datablock(state_, input_)
        assert block.shape == (m, k)
        op.entries = np.random.random((1, m))
        mult = op.entries @ block
        evald = op.apply(state_, input_)
        assert mult.shape == evald.shape
        assert np.allclose(mult, evald)

        # Special case: r > 1, m = 1
        state_ = np.random.random((r, k))
        input_ = input_[0]
        block = op.datablock(state_, input_)
        assert block.shape == (r, k)
        op.entries = np.random.random((r, r))
        mult = op.entries @ block
        evald = op.apply(state_, input_)
        assert mult.shape == evald.shape
        assert np.allclose(mult, evald)

        # Special case: r = m = 1.
        state_ = state_[0]
        block = op.datablock(state_, input_)
        assert block.shape == (1, k)
        op.entries = np.random.random()
        mult = op.entries[0, 0] * block[0]
        evald = op.apply(state_, input_)
        assert mult.shape == evald.shape
        assert np.allclose(mult, evald)

    def test_operator_dimension(self):
        """Test operator_dimension()."""
        assert self.Operator.operator_dimension(1, 2) == 2
        assert self.Operator.operator_dimension(3, 6) == 18
        assert self.Operator.operator_dimension(5, 2) == 10


if __name__ == "__main__":
    pytest.main([__file__])
