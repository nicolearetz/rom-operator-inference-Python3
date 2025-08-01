# operators/_base.py
"""Abstract base classes for operators."""

__all__ = [
    "InputMixin",
    "OperatorTemplate",
    "OpInfOperator",
    "ParametricOperatorTemplate",
    "ParametricOpInfOperator",
]

import os
import abc
import copy
import types
import numpy as np
import scipy.linalg as la
import scipy.sparse as sparse
import matplotlib.pyplot as plt

from .. import errors, utils


class InputMixin(abc.ABC):
    r"""Mixin for operators whose ``apply()`` method acts on the input
    :math:`\u`.

    Operators that do not inherit from this Mixin do not have an
    ``input_dimension`` attribute, which indicates :math:`m = 0`.
    """

    @property
    @abc.abstractmethod
    def input_dimension(self) -> int:
        r"""Dimension :math:`m` of the input :math:`\u` that the operator
        acts on.
        """
        raise NotImplementedError  # pragma: no cover


def has_inputs(obj) -> bool:
    r"""Return ``True`` if ``obj`` is an operator object whose ``apply()``
    method acts on the ``input_`` argument, i.e.,
    :math:`\Ophat_{\ell}(\qhat,\u)` depends on :math:`\u`.
    """
    return isinstance(obj, InputMixin)


# Nonparametric operators =====================================================
class OperatorTemplate(abc.ABC):
    r"""Template for general operators :math:`\Ophat_{\ell}(\qhat,\u).`

    In this package, an "operator" is a function
    :math:`\Ophat_{\ell}: \RR^r \times \RR^m \to \RR^r` that acts on a state
    vector :math:`\qhat\in\RR^r` and (optionally) an input vector
    :math:`\u\in\RR^m`.

    Models are defined as the sum of several operators,
    for example, an :class:`opinf.models.ContinuousModel` object represents a
    system of ordinary differential equations:

    .. math::
       \ddt\qhat(t)
       = \sum_{\ell=1}^{n_\textrm{terms}}\Ophat_{\ell}(\qhat(t),\u(t)).

    Notes
    -----
    This class can be used for custom nonparametric model terms that are not
    learnable with Operator Inference.
    For parametric model terms, see :class:`ParametricOperatorTemplate`.
    For model terms that can be learned with Operator Inference, see
    :class:`OpInfOperator` or :class:`ParametricOpInfOperator`.
    """

    # Properties --------------------------------------------------------------
    @property
    @abc.abstractmethod
    def state_dimension(self) -> int:
        r"""Dimension :math:`r` of the state :math:`\qhat` that the operator
        acts on.
        """
        raise NotImplementedError  # pragma: no cover

    def __str__(self) -> str:
        """String representation: class name + dimensions."""
        out = [self.__class__.__name__]
        out.append(f"state_dimension: {self.state_dimension}")
        if has_inputs(self):
            out.append(f"input_dimension: {self.input_dimension}")
        return "\n  ".join(out)

    def __repr__(self) -> str:
        return utils.str2repr(self)

    @staticmethod
    def _str(statestr, inputstr=None):
        """String representation of the operator, used when printing out the
        structure of a model.

        Parameters
        ----------
        statestr : str
            String representation of the state, e.g., ``"q(t)"`` or ``"q_j"``.
        inputstr : str
            String representation of the input, e.g., ``"u(t)"`` or ``"u_j"``.

        Returns
        -------
        opstr : str
            String representation of the operator acting on the state/input,
            e.g., ``"Aq(t)"`` or ``"Bu(t)"`` or ``"H[q(t) ⊗ q(t)]"``.
        """
        return f"f({statestr}, {inputstr})"  # pragma: no cover

    # Evaluation --------------------------------------------------------------
    @abc.abstractmethod
    def apply(self, state: np.ndarray, input_=None) -> np.ndarray:
        """Apply the operator mapping to the given state / input.

        Parameters
        ----------
        state : (r,) or (r, k) ndarray
            State vector or matrix of state vectors.
        input_ : (m,) or (m, k) ndarray or None
            Input vector or matrix of input vectors.

        Returns
        -------
        out : (r,) or (r, k) ndarray
            Application of the operator to the state / input, with the same
            number of dimensions as ``state`` and (if provided) ``input_``.
        """
        raise NotImplementedError  # pragma: no cover

    def jacobian(self, state: np.ndarray, input_=None) -> np.ndarray:
        r"""Construct the state Jacobian of the operator.

        If :math:`[\![\q]\!]_{i}` denotes the :math:`i`-th entry of a vector
        :math:`\q`, then the :math:`(i,j)`-th entry of the state Jacobian is
        given by

        .. math::
           [\![\ddqhat\Ophat(\qhat,\u)]\!]_{i,j}
           = \frac{\partial}{\partial[\![\qhat]\!]_j}
           [\![\Ophat(\qhat,\u)]\!]_i.

        Parameters
        ----------
        state : (r,) ndarray
            State vector.
        input_ : (m,) ndarray or float or None
            Input vector.

        Returns
        -------
        jac : (r, r) ndarray
            State Jacobian.
        """
        raise NotImplementedError  # pragma: no cover

    # Dimensionality reduction ------------------------------------------------
    def galerkin(self, Vr: np.ndarray, Wr=None):
        r"""Get the (Petrov-)Galerkin projection of this operator.

        Consider an operator :math:`\Op(\q,\u)`, where :math:`\q\in\RR^n`
        is the state and :math:`\u\in\RR^m` is the input.
        Given a *trial basis* :math:`\Vr\in\RR^{n\times r}` and a *test basis*
        :math:`\Wr\in\RR^{n\times r}`, the Petrov-Galerkin projection of
        :math:`\Op` is the operator :math:`\Ophat:\RR^r\times\RR^m\to\RR^r`
        defined by

        .. math::
           \Ophat(\qhat, \u) = (\Wr\trp\Vr)^{-1}\Wr\trp\Op(\Vr\qhat, \u)

        where :math:`\qhat\in\RR^n` approximates the original state via
        :math:`\q \approx \Vr\qhat`.

        Parameters
        ----------
        Vr : (n, r) ndarray
            Basis for the trial space :math:`\Vr`.
        Wr : (n, r) ndarray or None
            Basis for the test space :math:`\Wr`.
            If ``None`` (default), use ``Vr`` as the test basis.

        Returns
        -------
        op : :class:`OperatorTemplate`
            New operator object whose ``state_dimension``
            attribute equals ``r``. If this operator acts on inputs, the
            ``input_dimension`` attribute of the new operator should be
            ``self.input_dimension``.
        """
        raise NotImplementedError  # pragma: no cover

    # Model persistence -------------------------------------------------------
    def copy(self):
        """Return a copy of the operator using :func:`copy.deepcopy()`."""
        return copy.deepcopy(self)  # pragma: no cover

    def save(self, savefile: str, overwrite: bool = False) -> None:
        """Save the operator to an HDF5 file.

        Parameters
        ----------
        savefile : str
            Path of the file to save the basis in.
        overwrite : bool
            If ``True``, overwrite the file if it already exists. If ``False``
            (default), raise a ``FileExistsError`` if the file already exists.
        """
        raise NotImplementedError  # pragma: no cover

    @classmethod
    def load(cls, loadfile: str):
        """Load an operator from an HDF5 file.

        Parameters
        ----------
        loadfile : str
            Path to the file where the operator was stored via :meth:`save()`.
        """
        raise NotImplementedError  # pragma: no cover

    # Verification ------------------------------------------------------------
    def verify(
        self,
        plot: bool = False,
        *,
        k: int = 10,
        fdifftol: float = 1e-5,
        ntests: int = 4,
    ) -> None:  # pragma: no cover
        """Verify consistency between dimension properties and required
        methods.

        This method verifies :meth:`apply()` and, if implemented,
        :meth:`jacobian()`, :meth:`galerkin()`, :meth:`copy()`,
        :meth:`save()`, and :meth:`load()`.

        Parameters
        ----------
        plot : bool
            If ``True``, plot the relative errors of the finite difference
            check for :meth:`jacobian()` as a function of the perturbation
            size.
            If ``False`` (default), print a report of the relative errors.
            Nothing is plotted or printed if :meth:`jacobian()` is not
            implemented.

        Notes
        -----
        This method does **not** verify the correctness of :meth:`apply()`,
        only that it returns an output with the expected shape. However,
        if :meth:`jacobian()` is implemented, a finite difference check is
        applied to check that :meth:`apply()` and :meth:`jacobian()` are
        consistent.
        """
        # Verify dimensions exist and are valid.
        if (
            not isinstance((r := self.state_dimension), int) or r <= 0
        ):  # pragma: no cover
            raise errors.VerificationError(
                "state_dimension must be a positive integer "
                f"(current value: {repr(r)}, of type '{type(r).__name__}')"
            )

        if hasinputs := has_inputs(self):
            if (
                not isinstance((m := self.input_dimension), int) or m <= 0
            ):  # pragma: no cover
                raise errors.VerificationError(
                    "input_dimension must be a positive integer "
                    f"(current value: {repr(m)}, of type '{type(r).__name__}')"
                )
        else:
            m = 0

        # Verify apply() - - - - - - - - - - - - - - - - - - - - - - - - - - -
        Q = np.random.random((r, k))
        q = Q[:, 0]
        U, u = None, None
        if hasinputs:
            U = np.random.random((m, k))
            u = U[:, 0]

        out = self.apply(q, u)
        if not isinstance(out, np.ndarray) or out.shape != (
            r,
        ):  # pragma: no cover
            _message = [
                "apply(q, u) must return array of shape (state_dimension,)",
                "when q.shape = (state_dimension,)",
                "and u = None",
            ]
            if hasinputs:
                _message[-1] = "and u.shape = (input_dimension,)"
            raise errors.VerificationError(" ".join(_message))

        out = self.apply(Q, U)
        if not isinstance(out, np.ndarray) or out.shape != (
            r,
            k,
        ):  # pragma: no cover
            _message = [
                "apply(Q, U) must return array of shape (state_dimension, k)",
                "when Q.shape = (state_dimension, k)",
                "and U = None",
            ]
            if hasinputs:
                _message[-1] = "and U.shape = (input_dimension, k)"
            raise errors.VerificationError(" ".join(_message))

        # Report successes.
        def _report_isconsistent(method):
            _message = f"{method} is consistent with state_dimension"
            if hasinputs:
                _message += " and input_dimension"
            print(_message)

        _report_isconsistent("apply()")

        # Verify jacobian() - - - - - - - - - - - - - - - - - - - - - - - - - -
        def _gradient(f, x, h=1e-8):
            """Estimate the Jacobian of f:R^n -> R^m at x using perturbations
            of magnitude h.
            """
            E = np.eye((n := x.size))
            return np.array([(f(x + h * E[i]) - f(x)) / h for i in range(n)]).T

        def _finite_difference_check(f, df, x, hs=None):
            """Compare analytical and numerical derivatives."""
            dfx = df(x)
            if hs is None:
                hs = np.logspace(-10, -1, 10)[::-1]
            return hs, np.array(
                [la.norm(_gradient(f, x, h) - dfx) / la.norm(dfx) for h in hs]
            )

        try:
            out = self.jacobian(q, u)
        except NotImplementedError:  # pragma: no cover
            print("jacobian() not implemented")
        else:
            if np.isscalar(out) and out == 0:  # pragma: no cover
                print("jacobian() = 0")
            elif not isinstance(out, np.ndarray) or out.shape != (
                r,
                r,
            ):  # pragma: no cover
                _message = [
                    "jacobian(q, u) must return array",
                    "of shape (state_dimension, state_dimension)",
                    "when q.shape = (state_dimension,)",
                    "and u = None",
                ]
                if hasinputs:
                    _message[-1] = "and u.shape = (input_dimension,)"
                raise errors.VerificationError(" ".join(_message))
            else:
                _report_isconsistent("jacobian()")

                # Finite difference check.
                hs, diffs = _finite_difference_check(
                    lambda x: self.apply(x, u),
                    lambda x: self.jacobian(x, u),
                    np.random.standard_normal(r),
                )
                if plot:  # pragma: no cover
                    plt.loglog(hs, diffs, ".-", markersize=5, linewidth=0.5)
                else:
                    print(
                        "jacobian() finite difference relative errors",
                        "  ------------------------------------------",
                        sep="\n",
                    )
                    for h, err in zip(hs, diffs):
                        print(f"  h = {h:.2e}\terror = {err:.4e}")
                if np.min(diffs) > fdifftol:  # pragma: no cover
                    raise errors.VerificationError(
                        "jacobian() finite difference check failed"
                    )

        # Verify galerkin() - - - - - - - - - - - - - - - - - - - - - - - - - -
        def _orth(n, k):
            """Get an n x k matrix with orthonormal columns."""
            return la.qr(np.random.standard_normal((n, k)), mode="economic")[0]

        if r > 1:
            rnew = r // 2
            Vr = _orth(r, rnew)
            Wr = _orth(r, rnew)
            try:
                out = self.galerkin(Vr, Wr)
            except NotImplementedError:  # pragma: no cover
                print("galerkin() not implemented")
            else:
                if not isinstance(out, OperatorTemplate):  # pragma: no cover
                    raise errors.VerificationError(
                        "galerkin() must return object "
                        "whose class inherits from OperatorTemplate"
                    )
                if out.state_dimension != rnew:  # pragma: no cover
                    raise errors.VerificationError(
                        "galerkin(Vr, Wr).state_dimension != Vr.shape[1]"
                    )
                if hasinputs and out.input_dimension != m:  # pragma: no cover
                    raise errors.VerificationError(
                        "self.galerkin(Vr, Wr).input_dimension "
                        "!= self.input_dimension"
                    )
                WrTVr_LU = la.lu_factor(Wr.T @ Vr)
                for _ in range(ntests):
                    qr = np.random.random(rnew)
                    full = la.lu_solve(WrTVr_LU, Wr.T @ self.apply(Vr @ qr, u))
                    reduced = out.apply(qr, u)
                    if not np.allclose(reduced, full):  # pragma: no cover
                        raise errors.VerificationError(
                            "op2.apply(qr, u) != "
                            "inv(Wr.T @ Vr) @ Wr.T @ self.apply(Vr @ qr, u) "
                            "where op2 = self.galerkin(Vr, Wr)"
                        )
                out = self.galerkin(Vr)
                for _ in range(ntests):
                    qr = np.random.random(rnew)
                    full = Vr.T @ self.apply(Vr @ qr, u)
                    reduced = out.apply(qr, u)
                    if not np.allclose(reduced, full):  # pragma: no cover
                        raise errors.VerificationError(
                            "op2.apply(qr, u) != "
                            "Vr.T @ self.apply(Vr @ qr, u) "
                            "where op2 = self.galerkin(Vr) and Vr.T @ Vr = I"
                        )
                print("galerkin() is consistent with apply()")
        else:  # pragma: no cover
            print("cannot test galerkin() when state_dimension = 1")

        # Verify copy() - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        out = self.copy()
        if out is self:  # pragma: no cover
            raise errors.VerificationError("self.copy() is self")
        if out.__class__ is not self.__class__:  # pragma: no cover
            raise errors.VerificationError(
                "type(self.copy()) is not type(self)"
            )
        if out.state_dimension != r:  # pragma: no cover
            raise errors.VerificationError(
                "self.copy().state_dimension != self.state_dimension"
            )
        if hasinputs and out.input_dimension != m:  # pragma: no cover
            raise errors.VerificationError(
                "self.copy().input_dimension != self.input_dimension"
            )
        _report_isconsistent("copy()")
        for _ in range(ntests):
            q = np.random.random(r)
            if not np.allclose(
                out.apply(q, u), self.apply(q, u)
            ):  # pragma: no cover
                raise errors.VerificationError(
                    "self.copy().apply() not consistent with self.apply()"
                )
        print("copy() preserves the results of apply()")

        # Verify save()/load() - - - - - - - - - - - - - - - - - - - - - - - -
        tempfile = "_operatorverification.h5"
        try:
            self.save(tempfile)
            out = self.load(tempfile)
        except NotImplementedError:  # pragma: no cover
            print("save() and/or load() not implemented")
        else:
            if out.__class__ is not self.__class__:  # pragma: no cover
                raise errors.VerificationError(
                    "save()/load() does not preserve object type"
                )
            if out.state_dimension != r:  # pragma: no cover
                raise errors.VerificationError(
                    "save()/load() does not preserve state_dimension"
                )
            if hasinputs and out.input_dimension != m:  # pragma: no cover
                raise errors.VerificationError(
                    "save()/load() does not preserve input_dimension"
                )
            _report_isconsistent("save()/load()")
            for _ in range(ntests):
                q = np.random.random(r)
                if not np.allclose(
                    out.apply(q, u), self.apply(q, u)
                ):  # pragma: no cover
                    raise errors.VerificationError(
                        "save()/load() does not preserve the result of apply()"
                    )
            print("save()/load() preserves the results of apply()")
        finally:
            if os.path.isfile(tempfile):  # pragma: no cover
                os.remove(tempfile)


def is_nonparametric(obj) -> bool:
    """Return ``True`` if ``obj`` is a nonparametric operator object."""
    return isinstance(obj, OperatorTemplate)


class OpInfOperator(OperatorTemplate):
    r"""Template for nonparametric operators that can be calibrated through
    Operator Inference, i.e.,
    :math:`\Ophat_{\ell}(\qhat, \u) = \Ohat_{\ell}\d(\qhat, \u)`.

    In this package, an "operator" is a function
    :math:`\Ophat_{\ell}: \RR^r \times \RR^m \to \RR^r` that acts on a state
    vector :math:`\qhat\in\RR^r` and (optionally) an input vector
    :math:`\u\in\RR^m`.

    Models are defined as the sum of several operators,
    for example, an :class:`opinf.models.ContinuousModel` object represents a
    system of ordinary differential equations:

    .. math::
       \ddt\qhat(t)
       = \sum_{\ell=1}^{n_\textrm{terms}}\Ophat_{\ell}(\qhat(t),\u(t)).

    Operator Inference calibrates operators that can be written as the product
    of a matrix and some known (possibly nonlinear) function of the state
    and/or input:

    .. math::
       \Ophat_{\ell}(\qhat, \u)
       = \Ohat_{\ell}\d(\qhat, \u),

    where :math:`\Ohat_{\ell}\in\RR^{r\times d}` is a constant matrix, called
    the *operator entries*, and :math:`\d_{\ell}:\RR^r\times\RR^m\to\RR^d`.

    Notes
    -----
    * To define operators with a more general structure than
      :math:`\Ohat_{\ell}\d(\qhat, \u),` see :class:`OperatorTemplate`.
    * If the operator entries :math:`\Ohat_{\ell}` depend on one or more
      external parameters, it is called a *parametric operator*.
      See :class:`ParametricOpInfOperator`.
    """

    # Initialization ----------------------------------------------------------
    def __init__(self, entries=None):
        """Initialize an empty operator."""
        self._clear()
        if entries is not None:
            self.set_entries(entries)

    def _clear(self):
        """Delete operator ``entries`` and related attributes."""
        self.__entries = None

    @staticmethod
    def _validate_entries(entries):
        """Ensure argument is a NumPy array and screen for NaN, Inf entries."""
        if sparse.issparse(entries):
            return
        if not isinstance(entries, np.ndarray):
            raise TypeError(
                "operator entries must be NumPy or scipy.sparse array"
            )
        if np.any(np.isnan(entries)):
            raise ValueError("operator entries must not be NaN")
        elif np.any(np.isinf(entries)):
            raise ValueError("operator entries must not be Inf")

    def set_entries(self, entries) -> None:
        """Set the :attr:`entries` attribute."""
        self.__entries = entries

    # Properties --------------------------------------------------------------
    @property
    def entries(self) -> np.ndarray:
        r"""Discrete representation of the operator,
        the matrix :math:`\Ohat`.
        """
        return self.__entries

    @entries.setter
    def entries(self, entries):
        """Set the ``entries`` attribute."""
        self.set_entries(entries)

    @entries.deleter
    def entries(self):
        """Reset the ``entries`` attribute."""
        self._clear()

    @property
    def shape(self) -> tuple:
        """Shape of the operator matrix."""
        return None if self.entries is None else self.entries.shape

    @property
    def state_dimension(self) -> int:
        r"""Dimension :math:`r` of the state :math:`\qhat` that the operator
        acts on.
        """
        return None if self.entries is None else self.entries.shape[0]

    # Magic methods -----------------------------------------------------------
    def __getitem__(self, key):
        """Slice into the entries of the operator."""
        return None if self.entries is None else self.entries[key]

    def __eq__(self, other):
        """Two OpInf operators are equal if they are of the same class
        and have the same ``entries`` array.
        """
        if not isinstance(other, self.__class__):
            return False
        if (self.entries is None and other.entries is not None) or (
            self.entries is not None and other.entries is None
        ):
            return False
        if self.entries is not None:
            if self.shape != other.shape:
                return False
            return np.all(self.entries == other.entries)
        return True

    def __add__(self, other):
        """Nonparametric operators are linear in their entries."""
        if (ocls := other.__class__) is not (scls := self.__class__):
            raise TypeError(
                f"can't add object of type '{ocls.__name__}' "
                f"to object of type '{scls.__name__}'"
            )
        return scls(self.entries + other.entries)

    def __str__(self):
        out = OperatorTemplate.__str__(self)
        return out + f"\n  entries.shape:   {self.shape}"

    # Evaluation --------------------------------------------------------------
    @utils.requires("entries")
    def jacobian(self, state, input_=None) -> np.ndarray:  # pragma: no cover
        r"""Construct the state Jacobian of the operator.

        If :math:`[\![\q]\!]_{i}` denotes the :math:`i`-th entry of a vector
        :math:`\q`, then the :math:`(i,j)`-th entry of the state Jacobian is
        given by

        .. math::
           [\![\ddqhat\Ophat(\qhat,\u)]\!]_{i,j}
           = \frac{\partial}{\partial[\![\qhat]\!]_j}
           [\![\Ophat(\qhat,\u)]\!]_i.

        If a child class does not implement this method, it is assumed that
        the Jacobian is zero (i.e., the operator does not act on the state).

        Parameters
        ----------
        state : (r,) ndarray
            State vector.
        input_ : (m,) ndarray or float or None
            Input vector.

        Returns
        -------
        jac : (r, r) ndarray
            State Jacobian.
        """
        return 0

    # Dimensionality reduction ------------------------------------------------
    def _galerkin(self, Vr, Wr, func):
        r"""Get the (Petrov-)Galerkin projection of this operator.

        Subclasses may implement this function as follows:

        .. code-block:: python

           @utils.requires("entries")
           def galerkin(self, Vr, Wr=None):
               '''Docstring'''
               return self._galerkin(Vr, Wr, lambda A, V: <TODO>)

        Parameters
        ----------
        Vr : (n, r) ndarray
            Basis for the trial space.
        Wr : (n, r) ndarray or None
            Basis for the test space. If ``None``, defaults to ``Vr``.
        func : callable
            Function that accepts the operator ``entries`` and the trial
            basis ``Vr`` and computes the right side of the projection,
            i.e, the result of substituting :math:`\q` with :math:`\Vr\qhat`.
            For example, for linear operator :math:`\Op_\ell(\q) = \A\q`,
            ``func`` should return :math:`\A\Vr`.

        Returns
        -------
        op : operator
            New object of the same class as ``self``.
        """
        if Wr is None:
            Wr = Vr
        n, r = Wr.shape
        if self.entries.shape[0] != n:  # pragma: no cover
            raise errors.DimensionalityError("basis and operator not aligned")
        if Vr.shape[1] != r:  # pragma: no cover
            raise errors.DimensionalityError(
                "trial and test bases not aligned"
            )

        entries = Wr.T @ func(self.entries, Vr)
        if not np.allclose((WrTVr := Wr.T @ Vr), np.eye(r)):
            entries = la.solve(WrTVr, entries)
        return self.__class__(entries)

    # Operator inference ------------------------------------------------------
    @staticmethod
    @abc.abstractmethod
    def operator_dimension(r: int, m: int = None) -> int:
        r"""Column dimension of the operator entries.

        Child classes should decorate this method with ``@staticmethod``.

        Parameters
        ----------
        r : int
            State dimension.
        m : int or None
            Input dimension.

        Returns
        -------
        d : int
            Number of columns in the operator entries matrix.
            This is also the number of rows in the data matrix block.
        """
        raise NotImplementedError  # pragma: no cover

    @staticmethod
    @abc.abstractmethod
    def datablock(states: np.ndarray, inputs=None) -> np.ndarray:
        r"""Construct the data matrix block corresponding to the operator.

        For a nonparametric operator
        :math:`\Ophat_{\ell}(\qhat,\u) = \Ohat_{\ell}\d_{\ell}(\qhat, \u)`,
        the data matrix block corresponding to the data pairs
        :math:`\{(\qhat_j,\u_j)\}_{j=0}^{k-1}` is the matrix

        .. math::
           \D\trp = \left[\begin{array}{c|c|c|c}
           & & & \\
           \d_{\ell}(\qhat_0,\u_0) & \d_{\ell}(\qhat_1,\u_1)
           & \cdots &
           \\d_{\ell}(\qhat_{k-1},\u_{k-1})
           \\ & & &
           \end{array}\right]
           \in \RR^{d \times k}.

        Here, ``states`` is the projected snapshot matrix
        :math:`[~\qhat_0~~\cdots~~\qhat_{k-1}~]`
        and ``inputs`` is the (optional) input matrix
        :math:`[~\u_0~~\cdots~~\u_{k-1}~]`.

        Child classes should decorate this method with ``@staticmethod``.

        Parameters
        ----------
        states : (r, k) or (k,) ndarray
            State vectors. Each column is a single state vector.
            If one dimensional, it is assumed that :math:`r = 1`.
        inputs : (m, k) or (k,) ndarray or None
            Input vectors. Each column is a single input vector.
            If one dimensional, it is assumed that :math:`m = 1`.

        Returns
        -------
        block : (d, k) or (d,) ndarray
            Data matrix block. Here, :math:`d` is ``entries.shape[1]``.
        """
        raise NotImplementedError  # pragma: no cover

    # Model persistence -------------------------------------------------------
    def copy(self):
        """Return a copy of the operator."""
        entries = self.entries.copy() if self.entries is not None else None
        return self.__class__(entries)

    def save(self, savefile: str, overwrite: bool = False) -> None:
        """Save the operator to an HDF5 file.

        Parameters
        ----------
        savefile : str
            Path of the file to save the basis in.
        overwrite : bool
            If ``True``, overwrite the file if it already exists. If ``False``
            (default), raise a ``FileExistsError`` if the file already exists.
        """
        with utils.hdf5_savehandle(savefile, overwrite) as hf:
            meta = hf.create_dataset("meta", shape=(0,))
            meta.attrs["class"] = self.__class__.__name__
            if self.entries is not None:
                hf.create_dataset("entries", data=self.entries)

    @classmethod
    def load(cls, loadfile: str):
        """Load an operator from an HDF5 file.

        Parameters
        ----------
        loadfile : str
            Path to the file where the operator was stored via :meth:`save()`.
        """
        with utils.hdf5_loadhandle(loadfile) as hf:
            if (
                ClassName := hf["meta"].attrs["class"]
            ) != cls.__name__:  # pragma: no cover
                raise TypeError(
                    f"file '{loadfile}' contains '{ClassName}' "
                    f"object, use '{ClassName}.load()"
                )
            return cls(hf["entries"][:] if "entries" in hf else None)

    # Verification ------------------------------------------------------------
    def verify(
        self,
        plot: bool = False,
        *,
        k: int = 10,
        ntests: int = 4,
        r: int = 6,
        m: int = 3,
    ) -> None:  # pragma: no cover
        """Verify consistency between dimension properties and required
        methods.

        This method verifies :meth:`apply()` and, if implemented,
        :meth:`jacobian()`, :meth:`galerkin()`, :meth:`copy()`,
        :meth:`save()`, and :meth:`load()`.
        If the :attr:`entries` are not set, no checks are made.

        Parameters
        ----------
        plot : bool
            If ``True``, plot the relative errors of the finite difference
            check for :meth:`jacobian()` as a function of the perturbation
            size.
            If ``False`` (default), print a report of the relative errors.
            Nothing is plotted or printed if :meth:`jacobian()` is not
            implemented.

        Notes
        -----
        This method does **not** verify the correctness of :meth:`apply()`,
        only that it returns an output with the expected shape. However,
        if :meth:`jacobian()` is implemented, a finite difference check is
        applied to check that :meth:`apply()` and :meth:`jacobian()` are
        consistent.
        """
        # Verify operator_dimension() - - - - - - - - - - - - - - - - - - - - -
        if not isinstance(
            self.operator_dimension, types.FunctionType
        ):  # pragma: no cover
            raise errors.VerificationError(
                "operator_dimension() must have @staticmethod decorator"
            )
        d = self.operator_dimension(r, m)
        if not isinstance(d, int) or d <= 0:  # pragma: no cover
            raise errors.VerificationError(
                "operator_dimension() must return a positive integer"
            )

        # Verify datablock() - - - - - - - - - - - - - - - - - - - - - - - - -
        if not isinstance(
            self.datablock, types.FunctionType
        ):  # pragma: no cover
            raise errors.VerificationError(
                "datablock() must have @staticmethod decorator"
            )
        Dt = self.datablock(np.random.random((r, k)), np.random.random((m, k)))
        if not isinstance(Dt, np.ndarray) or Dt.ndim != 2:  # pragma: no cover
            raise errors.VerificationError(
                "datablock() must return a two-dimensional array"
            )
        if Dt.shape != (d, k):  # pragma: no cover
            raise errors.VerificationError(
                "datablock().shape[0] != operator_dimension()"
            )
        print("operator_dimension() is consistent with datablock()")

        # Verify instance methods - - - - - - - - - - - - - - - - - - - - - - -
        if self.entries is None:  # pragma: no cover
            return print("cannot verify apply() when entries=None")

        OperatorTemplate.verify(self, plot=plot, k=k, ntests=ntests)


# Parametric operators ========================================================
class ParametricOperatorTemplate(abc.ABC):
    r"""Template for operators that depend on external parameters,
    :math:`\Ophat_{\ell}(\qhat,\u;\bfmu).`

    In this package, a parametric "operator" is a function
    :math:`\Ophat_{\ell}: \RR^r \times \RR^m \times \RR^p \to \RR^r` that acts
    on a state vector :math:`\qhat\in\RR^r`, an (optional) input vector
    :math:`\u\in\RR^m`, and a parameter vector :math:`\bfmu\in\RR^p`.

    Parametric models are defined as the sum of several operators, at least
    one of which is parametric.
    For example, a system of ODEs:

    .. math::
       \ddt\qhat(t;\bfmu)
       = \sum_{\ell=1}^{n_\textrm{terms}}\Ophat_{\ell}(\qhat(t),\u(t);\bfmu).

    Notes
    -----
    This class can be used for custom nonparametric model terms that are not
    learnable with Operator Inference.
    For nonparametric model terms, see :class:`OperatorTemplate`.
    For model terms that can be learned with Operator Inference, see
    :class:`OpInfOperator` or :class:`ParametricOpInfOperator`.
    """

    # Nonparametric operator class that this parametric operator evaluates to.
    _OperatorClass = NotImplemented

    # Properties --------------------------------------------------------------
    @property
    @abc.abstractmethod
    def state_dimension(self) -> int:
        r"""Dimension :math:`r` of the state :math:`\qhat` that the operator
        acts on.
        """
        raise NotImplementedError  # pragma: no cover

    @property
    @abc.abstractmethod
    def parameter_dimension(self) -> int:
        r"""Dimension :math:`p` of the parameter vector :math:`\bfmu` that the
        operator matrix depends on.
        """
        raise NotImplementedError  # pragma: no cover

    def __str__(self) -> str:
        """String representation: class name, dimensions, evaluation type."""
        out = [self.__class__.__name__]
        out.append(f"state_dimension:     {self.state_dimension}")
        if has_inputs(self):
            out.append(f"input_dimension:     {self.input_dimension}")
        out.append(f"parameter_dimension: {self.parameter_dimension}")
        out.append(f"evaluate(parameter) -> {self._OperatorClass.__name__}")
        return "\n  ".join(out)

    def __repr__(self) -> str:
        return utils.str2repr(self)

    # Evaluation --------------------------------------------------------------
    def _check_parametervalue_dimension(self, parameter):
        """Ensure a new parameter value has the expected shape."""
        if (pdim := self.parameter_dimension) is None:
            raise RuntimeError("parameter_dimension not set")
        if np.atleast_1d(parameter).shape[0] != pdim:
            raise ValueError(f"expected parameter of shape ({pdim:d},)")

    @abc.abstractmethod
    def evaluate(self, parameter):
        r"""Evaluate the operator at the given parameter value,
        resulting in a nonparametric operator.

        Parameters
        ----------
        parameter : (p,) ndarray or float
            Parameter value :math:`\bfmu` at which to evalute the operator.

        Returns
        -------
        op : nonparametric :mod:`opinf.operators` operator
            Nonparametric operator corresponding to the parameter value.
            This should be an instance of :class:`OperatorTemplate` (or
            a class that inherits from it).
        """
        raise NotImplementedError  # pragma: no cover

    def apply(self, parameter, state, input_=None):
        r"""Apply the operator to the given state and input
        at the specified parameter value, :math:`\Ophat_\ell(\qhat,\u;\bfmu)`.

        Parameters
        ----------
        parameter : (p,) ndarray or float
            Parameter value.
        state : (r,) ndarray
            State vector.
        input_ : (m,) ndarray or float or None
            Input vector.

        Returns
        -------
        (r,) ndarray

        Notes
        -----
        For repeated calls with the same parameter value, use
        :meth:`evaluate()` to first get the nonparametric operator
        corresponding to the parameter value.

        .. code-block::

           # Instead of this...
           >>> values = [parametric_operator.apply(parameter, q, u)
           ...           for q, u in zip(list_of_states, list_of_inputs)]
           # ...it is faster to do this.
           >>> operator_at_parameter = parametric_operator.evaluate(parameter)
           >>> values = [operator_at_parameter.apply(q, u)
           ...           for q, u in zip(list_of_states, list_of_inputs)]
        """
        return self.evaluate(parameter).apply(state, input_)

    def jacobian(self, parameter, state, input_=None):
        r"""Construct the state Jacobian of the operator,
        :math:`\ddqhat\Ophat_\ell(\qhat,\u;\bfmu)`.

        If :math:`[\![\q]\!]_{i}` denotes the entry :math:`i` of a vector
        :math:`\q`, then the entries of the state Jacobian are given by

        .. math::
           [\![\ddqhat\Ophat_\ell(\qhat,\u;\bfmu)]\!]_{i,j}
           = \frac{\partial}{\partial[\![\qhat]\!]_j}
           [\![\Ophat_\ell(\qhat,\u;\bfmu)]\!]_i.

        Parameters
        ----------
        parameter : (p,) ndarray or float
            Parameter value.
        state : (r,) ndarray
            State vector.
        input_ : (m,) ndarray or float or None
            Input vector.

        Returns
        -------
        jac : (r, r) ndarray
            State Jacobian.

        Notes
        -----
        For repeated calls with the same parameter value, use
        :meth:`evaluate()` to first get the nonparametric operator
        corresponding to the parameter value.

        .. code-block::

           # Instead of this...
           >>> values = [parametric_operator.jacobian(parameter, q, u)
           ...           for q, u in zip(list_of_states, list_of_inputs)]
           # ...it is faster to do this.
           >>> operator_at_parameter = parametric_operator.evaluate(parameter)
           >>> values = [operator_at_parameter.jacobian(q, u)
           ...           for q, u in zip(list_of_states, list_of_inputs)]
        """
        return self.evaluate(parameter).jacobian(state, input_)

    # Dimensionality reduction ------------------------------------------------
    def galerkin(self, Vr, Wr=None):
        r"""Get the (Petrov-)Galerkin projection of this operator.

        Consider an operator :math:`\Op(\q,\u)`, where :math:`\q\in\RR^n`
        is the state and :math:`\u\in\RR^m` is the input.
        Given a *trial basis* :math:`\Vr\in\RR^{n\times r}` and a *test basis*
        :math:`\Wr\in\RR^{n\times r}`, the Petrov-Galerkin projection of
        :math:`\Op` is the operator :math:`\Ophat:\RR^r\times\RR^m\to\RR^r`
        defined by

        .. math::
           \Ophat(\qhat, \u) = (\Wr\trp\Vr)^{-1}\Wr\trp\Op(\Vr\qhat, \u)

        where :math:`\qhat\in\RR^n` approximates the original state via
        :math:`\q \approx \Vr\qhat`.

        Parameters
        ----------
        Vr : (n, r) ndarray
            Basis for the trial space.
        Wr : (n, r) ndarray or None
            Basis for the test space. If ``None``, defaults to ``Vr``.

        Returns
        -------
        op : operator
            New object of the same class as ``self`` whose ``state_dimension``
            attribute equals ``r``. If this operator acts on inputs, the
            ``input_dimension`` attribute of the new operator should be
            ``self.input_dimension``.
        """
        raise NotImplementedError  # pragma: no cover

    # Model persistence -------------------------------------------------------
    def copy(self):
        """Return a copy of the operator."""
        return copy.deepcopy(self)  # pragma: no cover

    def save(self, savefile: str, overwrite: bool = False) -> None:
        """Save the operator to an HDF5 file.

        Parameters
        ----------
        savefile : str
            Path of the file to save the basis in.
        overwrite : bool
            If ``True``, overwrite the file if it already exists. If ``False``
            (default), raise a ``FileExistsError`` if the file already exists.
        """
        raise NotImplementedError  # pragma: no cover

    @classmethod
    def load(cls, loadfile: str):
        """Load an operator from an HDF5 file.

        Parameters
        ----------
        loadfile : str
            Path to the file where the operator was stored via :meth:`save()`.
        """
        raise NotImplementedError  # pragma: no cover

    # Verification ------------------------------------------------------------
    def verify(self, testparam=None):
        """Verify dimension attributes and :meth:`evaluate()`.

        Parameters
        ----------
        testparam : (p,) ndarray or None
            Test parameter at which to evaluate the operator.
            If ``None`` (default), draw test parameter entries from the
            standard Normal distribution.
        """
        # Check the _OperatorClass.
        if not issubclass(
            self._OperatorClass, OperatorTemplate
        ):  # pragma: no cover
            raise errors.VerificationError(
                "_OperatorClass must be a nonparametric operator type"
            )

        # Verify dimensions exist and are valid.
        if (
            not isinstance((r := self.state_dimension), int) or r <= 0
        ):  # pragma: no cover
            raise errors.VerificationError(
                "state_dimension must be a positive integer "
                f"(current value: {repr(r)}, of type '{type(r).__name__}')"
            )

        if hasinputs := has_inputs(self):
            if (
                not isinstance((m := self.input_dimension), int) or m <= 0
            ):  # pragma: no cover
                raise errors.VerificationError(
                    "input_dimension must be a positive integer "
                    f"(current value: {repr(m)}, of type '{type(r).__name__}')"
                )
        else:
            m = 0

        # Get a test parameter.
        if testparam is None:
            testparam = np.random.standard_normal(self.parameter_dimension)
        if np.shape(testparam) != (
            self.parameter_dimension,
        ):  # pragma: no cover
            raise ValueError("testparam.shape != (parameter_dimension,)")

        # Evaluate the operator at the test parameter.
        op_evaluated = self.evaluate(testparam)
        if not isinstance(
            op_evaluated, self._OperatorClass
        ):  # pragma: no cover
            raise errors.VerificationError(
                "evaluate() must return instance of type _OperatorClass"
            )
        if not is_nonparametric(op_evaluated):  # pragma: no cover
            raise errors.VerificationError(
                "_OperatorClass must be a nonparametric operator type"
            )
        if (
            op_evaluated.state_dimension != self.state_dimension
        ):  # pragma: no cover
            raise errors.VerificationError(
                "result of evaluate() does not retain the state_dimension"
            )
        if hasinputs:
            if not has_inputs(op_evaluated):  # pragma: no cover
                raise errors.VerificationError(
                    "result of evaluate() should depend on inputs"
                )
            if op_evaluated.input_dimension != m:  # pragma: no cover
                raise errors.VerificationError(
                    "result of evaluate() does not retain the input_dimension"
                )
        else:
            if has_inputs(op_evaluated):  # pragma: no cover
                raise errors.VerificationError(
                    "result of evaluate() should not depend on inputs"
                )


def is_parametric(obj) -> bool:
    """Return ``True`` if ``obj`` is a parametric operator object."""
    return isinstance(obj, ParametricOperatorTemplate)


class ParametricOpInfOperator(ParametricOperatorTemplate):
    r"""Template for operators that depend on external parameters, and which
    can be calibrated through operator inference, i.e.,
    :math:`\Ophat_\ell(\qhat,\u;\bfmu) = \Ohat_\ell(\bfmu)\d_\ell(\qhat,\u)`.
    """

    # Initialization ----------------------------------------------------------
    def __init__(self):
        """Initialize the parameter_dimension."""
        self._clear()

    def _clear(self) -> None:
        """Reset the operator to its post-constructor state."""
        self.__p = None
        self.__entries = None

    def _set_parameter_dimension_from_values(self, parameters) -> None:
        """Extract and save the dimension of the parameter space from a set of
        one or more parameter values.

        Parameters
        ----------
        parameters : (s, p) or (s,) ndarray
            Parameter value(s).
        """
        if (dim := len(shape := np.shape(parameters))) == 1:
            self.__p = 1
        elif dim == 2:
            self.__p = shape[1]
        else:
            raise ValueError("parameter values must be scalars or 1D arrays")

    # Verification ------------------------------------------------------------
    @staticmethod
    def _check_shape_consistency(iterable, prefix: str) -> None:
        """Ensure that each array in `iterable` has the same shape."""
        shape = np.shape(iterable[0])
        if any(np.shape(A) != shape for A in iterable):
            raise ValueError(f"{prefix} shapes inconsistent")

    # Properties --------------------------------------------------------------
    @property
    def state_dimension(self) -> int:
        r"""Dimension :math:`r` of the state :math:`\qhat` that the operator
        acts on.
        """
        return None if self.entries is None else self.entries[0].shape[0]

    @property
    def parameter_dimension(self) -> int:
        r"""Dimension :math:`p` of the parameter vector :math:`\bfmu` that the
        operator matrix depends on.
        """
        return self.__p

    @parameter_dimension.setter
    def parameter_dimension(self, p):
        """Set :attr:`parameter_dimension`.
        Only allowed if :attr:`parameter_dimension` is currently ``None``.
        """
        if self.__p is not None:
            raise AttributeError(
                "can't set property 'parameter_dimension' twice"
            )
        if not isinstance(p, int) or p < 1:
            raise ValueError("parameter_dimension must be a positive integer")
        self.__p = p

    @property
    def shape(self) -> tuple:
        """Shape of the operator matrix when evaluated
        at a parameter value.
        """
        return None if self.entries is None else self.entries[0].shape

    @property
    def entries(self):
        r"""Arrays that define the operator matrix as a function of the
        parameter vector.
        """
        return self.__entries

    @abc.abstractmethod
    def set_entries(self, entries, fromblock: bool = False) -> None:
        r"""Set the arrays that define the operator matrix as a function of
        the parameter vector.

        Parameters
        ----------
        entries : list of s (r, d) ndarrays, or (r, sd) ndarray
            Operator entries, either as a list of arrays
            (``fromblock=False``, default)
            or as a horizontal concatenatation of arrays (``fromblock=True``).
        fromblock : bool
            If ``True``, interpret ``entries`` as a horizontal concatenation
            of arrays; if ``False`` (default), interpret ``entries`` as a list
            of arrays.
        """
        self.__entries = entries

    # Operator inference ------------------------------------------------------
    @abc.abstractmethod
    def operator_dimension(self, s: int, r: int, m: int = None) -> int:
        r"""Number of columns in the total operator matrix.

        Parameters
        ----------
        s : int
            Number of training parameter values.
        r : int
            State dimension.
        m : int or None
            Input dimension.

        Returns
        -------
        d : int
            Number of columns in the total operator entries matrix.
            This is also the number of rows in the data matrix block.
        """
        raise NotImplementedError  # pragma: no cover

    @abc.abstractmethod
    def datablock(self, parameters, states, inputs=None):
        r"""Return the data matrix block corresponding to the operator.

        Parameters
        ----------
        parameters : (s, p) ndarray
            Traning parameter values :math:`\bfmu_{0},\ldots,\bfmu_{s-1}`.
        states : list of s (r, k_i) ndarrays
            State snapshots for each of the :math:`s` training parameter
            values.
        inputs : list of s (m, k_i) ndarrays
            Inputs corresponding to the state snapshots.

        Returns
        -------
        block : (D, K) ndarray
            Data block for the parametric operator.
            Here, :math:`D` is the total operator matrix dimension and
            :math:`K = \sum_{i=0}^{s-1}k_i`, the total number of state
            snapshots.
        """
        raise NotImplementedError  # pragma: no cover


def is_opinf(obj) -> bool:
    """Return ``True`` if ``obj`` is an OpInf operator (whether or not the
    entries are calibrated).
    """
    return isinstance(obj, (OpInfOperator, ParametricOpInfOperator))


def is_uncalibrated(obj) -> bool:
    """Return ``True`` if ``obj`` is an OpInf operator with empty entries."""
    return is_opinf(obj) and obj.entries is None
