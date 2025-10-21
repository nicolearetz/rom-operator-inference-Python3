# from .. import utils
from ._base import OpInfOperator

import numpy as np

import scipy.linalg as la
import itertools

# import scipy.sparse as sparse
from scipy.special import comb

__all__ = ["PolynomialOperator"]


class PolynomialOperator(OpInfOperator):
    r"""Polynomial state operator
    :math:`\Ophat_{\ell}(\qhat,\u) = \Phat[\qhat^{\otimes p}]`
    where :math:`\otimes p` indicates the Kronecker product of a vector with
    itself :math:`p` times. The matrix :math:`\Phat` is
    :math:`r \times \binom{r+p-1}{p}`.

    This class is equivalent to the following.

    * :math:`p = 1`: :class:`ConstantOperator`
    * :math:`p = 2`: :class:`QuadraticOperator`
    * :math:`p = 3`: :class:`CubicOperator`
    * :math:`p = 4`: :class:`QuarticOperator`

    Parameters
    ----------
    polynomial_order : int
        Order of the Kronecker product.
    entries : (r, (r + p - 1 choose p)) ndarray or None
        Operator matrix :math:`\Phat`.

    Examples
    --------
    >>> import numpy as np
    >>> H = opinf.operators.QuadraticOperator()
    >>> P = opinf.operators.PolynomialOperator(2)
    >>> entries = np.random.random((10, 100))
    >>> H.set_entries(entries)
    >>> H.shape
    (10, 55)
    >>> P.set_entries(H.entries)
    >>> q = np.random.random(10)
    >>> outH = H.apply(q)
    >>> out = P.apply(q)
    >>> np.allclose(out, outH)
    True
    """

    def __init__(self, polynomial_order: int, entries=None):
        """Initialize an empty operator."""
        if polynomial_order < 0 or (
            not np.isclose(polynomial_order, p := int(polynomial_order))
        ):
            raise ValueError(
                "expected non-negative integer polynomial order"
                + f" polynomial_order. Got p={polynomial_order}"
            )

        self.polynomial_order = p

        super().__init__(entries=entries)

    def operator_dimension(self, r: int, m=None) -> int:
        """
        computes the number of non-redundant terms in a vector of length r
        that is taken to the power p with the Kronecker product

        Parameters
        ----------
        r : int
            State dimension.
        m : int or None
            Input dimension -- currently not used
        """
        if r < 0 or (not np.isclose(r, int(r))):
            raise ValueError(
                f"expected non-negative integer reduced dimension r. Got r={r}"
            )

        # for constant operators the dimension does not matter
        if (p := self.polynomial_order) == 0:
            return 1

        return comb(r, p, repetition=True, exact=True)

    def datablock(self, states: np.ndarray, inputs=None) -> np.ndarray:
        r"""Return the data matrix block corresponding to
        this operator's polynomial order,
        with ``states`` being the projected snapshots.

        Parameters
        ----------
        states : (r, k) or (k,) ndarray
            State vectors. Each column is a single state vector.
            If one dimensional, it is assumed that :math:`r = 1`.
        inputs : (m, k) or (k,) ndarray or None
            Input vectors (not used).

        Returns
        -------
        datablock : (self.operator_dimension(r), k) ndarray
            where p is the polynomial order for this operator.
        """
        # if constant, we just return an array containing ones
        # of shape 1 x <number of data points>
        if self.polynomial_order == 0:
            return np.ones((1, np.atleast_1d(states).shape[-1]))

        # make sure data is in 2D
        states = np.atleast_2d(states)

        if states.shape[0] == 0:
            return np.empty(shape=(0, states.shape[1]))

        # compute data matrix
        return PolynomialOperator.exp_p(states, self.polynomial_order)

    @staticmethod
    def keptIndices_p(r, p):
        """
        returns the non-redundant indices in a kronecker-product with
        exponent p when the dimension of the vector is r
        """
        if p == 0:
            return np.array([0])

        dim_if_p_was_one_smaller = PolynomialOperator(
            polynomial_order=(p - 1)
        ).operator_dimension(r=r)
        indexmatrix = np.reshape(
            np.arange(r * dim_if_p_was_one_smaller),
            (r, dim_if_p_was_one_smaller),
        )
        return np.hstack(
            [
                indexmatrix[
                    i,
                    : PolynomialOperator(
                        polynomial_order=(p - 1)
                    ).operator_dimension(i + 1),
                ]
                for i in range(r)
            ]
        )

    @staticmethod
    def exp_p(x, p, kept=None):
        """
        recursively computes x^p without the redundant terms
        (it still computes them but then takes them out)
        the result has shape

        if x is 1-dimensional:
        (PolynomialOperator.operator_dimension(x.shape[0]),)

        otherwise:
        (PolynomialOperator.operator_dimension(x.shape[0]), x.shape[1])
        """
        # for a constant operator, we just return 1 (x^0 = 1)
        if p == 0:
            return np.ones([1])

        # for a linear operator, x^1 = 1
        if p == 1:
            return x

        # identify kept entries in condensed Kronecker product for
        # this reduced dimension
        # for all polynomial orders up to self.polynomial order
        if kept is None:
            r = x.shape[0]
            kept = [
                PolynomialOperator.keptIndices_p(r=r, p=i)
                for i in range(p + 1)
            ]

        # distinguish between the shapes of the input
        if len(x.shape) == 1:
            # this gets called when the ROM is run
            return np.kron(x, PolynomialOperator.exp_p(x, p - 1, kept))[
                kept[p]
            ]
        else:
            # this gets called for constructing the data matrix
            return la.khatri_rao(x, PolynomialOperator.exp_p(x, p - 1, kept))[
                kept[p]
            ]

    @staticmethod
    def ckron_indices(r, p):
        """Construct a mask for efficiently computing the compressed, p-times
        repeated Kronecker product.

        This method provides a faster way to evaluate :meth:`ckron`
        when the state dimension ``r`` is known *a priori*.

        Parameters
        ----------
        r : int
            State dimension.
        p: int
            polynomial order

        Returns
        -------
        mask : ndarray
            Compressed Kronecker product mask.

        """
        mask = np.array(
            [*itertools.combinations_with_replacement([*range(r)][::-1], p)]
        )[::-1, :]
        return mask

    def apply(self, state: np.ndarray, input_=None) -> np.ndarray:
        r"""Apply the operator to the given state. Input is not used.
        See OpInfOperator.apply for description.
        """
        # note: having all these if conditions here to distinguish between
        # p=0, p=1, p>1
        # really makes things slower than necessary (apply gets called a lot)
        # it might be worth excluding these special cases from the
        # PolynomialOperator
        # class and defaulting to ConstantOperator and LinearOperator instead.

        if state.shape[0] != self.state_dimension:
            raise ValueError(
                f"Expected state of dimension r={self.state_dimension}."
                + f"Got state.shape={state.shape}"
            )

        # constant
        if self.polynomial_order == 0:
            if np.ndim(state) == 2:  # r, k > 1.
                return np.outer(self.entries, np.ones(state.shape[-1]))
            if np.ndim(self.entries) == 2:
                return self.entries[:, 0]
            return self.entries
        # note: no need to go through the trouble of identifying the
        # non-redundant indices

        # linear
        if self.polynomial_order == 1:
            return self.entries @ state
        # note: no need to go through the trouble of identifying the
        # non-redundant indices

        restricted_kronecker_product = np.prod(
            state[self.nonredudant_entries_mask], axis=1
        )
        return self.entries @ restricted_kronecker_product

    # Properties --------------------------------------------------------------
    @property
    def nonredudant_entries(self) -> list:
        r"""list containing at index i a list of the indices that are kept
        when restricting the i-times Kronecker product of a vector of
        shape self.state_dimension() with itself.
        """
        return [
            PolynomialOperator.keptIndices_p(r=self.state_dimension, p=i)
            for i in range(self.polynomial_order + 1)
        ]

    @property
    def nonredudant_entries_mask(self) -> np.ndarray:
        r"""list containing at index i a list of the indices that are kept
        when restricting the i-times Kronecker product of a vector of
        shape self.state_dimension() with itself.
        """
        return self.ckron_indices(
            r=self.state_dimension, p=self.polynomial_order
        )

    def restrict_me_to_subspace(self, indices_trial, indices_test=None):

        if indices_test is None:
            indices_test = indices_trial

        if max(indices_trial) >= self.state_dimension:
            raise RuntimeError(
                f"""
                               In PolynomialOperator.restrict_to_subspace:
                               Encountered restriction onto unknown trial basis
                               vector number {max(indices_trial)}.
                               Reduced dimension is {self.state_dimension}"""
            )

        if max(indices_test) >= self.state_dimension:
            raise RuntimeError(
                f"""
                               In PolynomialOperator.restrict_to_subspace:
                               Encountered restriction onto unknown test basis
                               vector number {max(indices_test)}.
                               Reduced dimension is {self.state_dimension}"""
            )

        new_entries = PolynomialOperator.restrict_matrix_to_subspace(
            indices_trial=indices_trial,
            entries=self.entries,
            polynomial_order=self.polynomial_order,
            indices_test=indices_test,
        )

        return PolynomialOperator(
            entries=new_entries, polynomial_order=self.polynomial_order
        )

    @staticmethod
    def restrict_matrix_to_subspace(
        indices_trial, entries, polynomial_order, indices_test=None
    ):
        """
        - not checking for duplicate indices
        """
        if indices_test is None:
            indices_test = indices_trial

        # constant
        if polynomial_order == 0:
            if np.ndim(entries) == 2:
                return entries[indices_test, 0]
            return entries[indices_test]

        # higher-order polynomials
        entries_sub = entries[indices_test, :]
        # restrict to test indices only

        col_indices = PolynomialOperator.columnIndices_p(
            indices=indices_trial, p=polynomial_order
        )
        # find out which columns to keep

        return entries_sub[:, col_indices]

    @staticmethod
    def columnIndices_p(indices, p):
        if p == 1:
            return indices

        if p == 0:
            return [0]

        sub = PolynomialOperator.columnIndices_p(indices, p - 1)
        return [
            comb(indices[i], p, repetition=True, exact=True) + sub[j]
            for i in range(len(indices))
            for j in range(comb(i + 1, p - 1, repetition=True, exact=True))
        ]
