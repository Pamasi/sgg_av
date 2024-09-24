"""
.. _fuzzyop:

The `ltn.fuzzy_ops` module contains the PyTorch implementation of some common fuzzy logic operators and aggregators.
Refer to the `LTN paper <https://arxiv.org/abs/2012.13635>`_ for a detailed description of these operators
(see the Appendix).

All the operators included in this module support the traditional NumPy/PyTorch broadcasting.

The operators have been designed to be used with :class:`ltn.core.Connective` or :class:`ltn.core.Quantifier`.
"""

import torch
from ltn_log import LTNObject

# these are the projection functions to make the Product Real Logic stable. These functions help to change the input
# of particular fuzzy operators in such a way they do not lead to gradient problems (vanishing, exploding).
eps = 1e-4  # epsilon is set to small value in such a way to not change the input too much


def pi_0(x):
    """
    Function that has to be used when we need to assure that the truth value in input to a fuzzy operator is never equal
    to zero, in such a way to avoid gradient problems. It maps the interval [0, 1] in the interval ]0, 1], where the 0
    is excluded.

    Parameters
    -----------
    x: :class:`torch.Tensor`
        A truth value.

    Returns
    -----------
    :class:`torch.Tensor`
        The input truth value changed in such a way to prevent gradient problems (0 is changed with a small number
        near 0).
    """
    return (1 - eps) * x + eps


def pi_1(x):
    """
    Function that has to be used when we need to assure that the truth value in input to a fuzzy operator is never equal
    to one, in such a way to avoid gradient problems. It maps the interval [0, 1] in the interval [0, 1[, where the 1
    is excluded.

    Parameters
    -----------
    x: :class:`torch.Tensor`
        A truth value.

    Returns
    -----------
    :class:`torch.Tensor`
        The input truth value changed in such a way to prevent gradient problems (1 is changed with a small number
        near 1).
    """
    return (1 - eps) * x

def check_values(*values):
    """
    This function checks the input values are in the range [0., 1.] and raises an exception if it is not the case.

    Parameters
    -----------
    values: :obj:`list` or :obj:`tuple`
        List or tuple of :class:`torch.Tensor` containing the truth values of the operands.

    Raises
    -----------
    :class:`ValueError`
        Raises when the values of the input parameters are incorrect.
    """

    values = list(values)
    """
    for v in values:
        if not torch.all(torch.where(torch.logical_and(v >= 0., v <= 1.), 1., 0.)):
            raise ValueError("Expected inputs of connectives and quantifiers to be tensors of truth values in the range"
                             " [0., 1.], but got some values outside this range.")
    """



class ConnectiveOperator:
    """
    Abstract class for connective operators.

    Every connective operator implemented in LTNtorch must inherit from this class and implements
    the `__call__()` method.

    Raises
    -----------
    :class:`NotImplementedError`
        Raised when `__call__()` is not implemented in the sub-class.
    """

    def __call__(self, *args, **kwargs):
        """
        Implements the behavior of the connective operator.

        Parameters
        ----------
        args : :class:`torch.Tensor`, or :obj:`tuple` of :class:`torch.Tensor`
            Operand (operands) on which the unary (binary) connective operator has to be applied.
        """
        raise NotImplementedError()


class UnaryConnectiveOperator(ConnectiveOperator):
    """
    Abstract class for unary connective operators.

    Every unary connective operator implemented in LTNtorch must inherit from this class and
    implement the `__call__()` method.

    Raises
    -----------
    :class:`NotImplementedError`
        Raised when `__call__()` is not implemented in the sub-class.
    """

    def __call__(self, *args, **kwargs):
        """
        Implements the behavior of the unary connective operator.

        Parameters
        ----------
        args : :class:`torch.Tensor`
            Operand on which the unary connective operator has to be applied.
        """
        raise NotImplementedError()


class BinaryConnectiveOperator(ConnectiveOperator):
    """
    Abstract class for binary connective operators.

    Every binary connective operator implemented in LTNtorch must inherit from this class
    and implement the `__call__()` method.

    Raises
    -----------
    :class:`NotImplementedError`
        Raised when `__call__()` is not implemented in the sub-class.
    """

    def __call__(self, *args, **kwargs):
        """
        Implements the behavior of the binary connective operator.

        Parameters
        ----------
        args : :obj:`tuple` of :class:`torch.Tensor`
            Operands on which the binary connective operator has to be applied.
        """
        raise NotImplementedError()


class AggregationOperator:
    """
    Abstract class for aggregation operators.

    Every aggregation operator implemented in LTNtorch must inherit from this class
    and implement the `__call__()` method.

    Raises
    -----------
    :class:`NotImplementedError`
        Raised when `__call__()` is not implemented in the sub-class.
    """

    def __call__(self, *args, **kwargs):
        """
        Implements the behavior of the aggregation operator.

        Parameters
        ----------
        args: :class:`torch.Tensor`
            :ref:`Grounding <notegrounding>` of formula on which the aggregation operator has to be applied.
        """
        raise NotImplementedError()





"LOG LTN OPERATORS/AGGREGATORS"
class OR_LogMeanExp(BinaryConnectiveOperator):

    def __repr__(self):
        return "OR_LogMeanExp()"

    def __call__(self, x, y):
        """
        It applies the Godel fuzzy disjunction operator to the given operands.

        Parameters
        -----------
        x : :class:`torch.Tensor`
            First operand on which the operator has to be applied.
        y : :class:`torch.Tensor`
            Second operand on which the operator has to be applied.

        Returns
        -----------
        :class:`torch.Tensor`
            The Godel fuzzy disjunction of the two operands.
        """
        max_val = torch.maximum(x, y)
        return max_val + torch.log(torch.mean(torch.stack([torch.exp(x - max_val), torch.exp(y - max_val)]),dim=0))

class OrLogExp(BinaryConnectiveOperator):
    """
    Lukasiewicz fuzzy disjunction operator.

    :math:`\lor_{Lukasiewicz}(x, y) = \operatorname{min}(x + y, 1)`

    Examples
    --------
    Note that:

    - variable `x` has two individuals;
    - variable `y` has three individuals;
    - the shape of the result of the conjunction is `(2, 3)` due to the :ref:`LTN broadcasting <broadcasting>`. The first dimension is dedicated two variable `x`, while the second dimension to variable `y`;
    - at index `(0, 0)` there is the evaluation of the formula on first individual of `x` and first individual of `y`, at index `(0, 1)` there is the evaluation of the formula on first individual of `x` and second individual of `y`, and so forth.

    >>> import ltn
    >>> import torch
    >>> Or = ltn.Connective(ltn.fuzzy_ops.OrLuk())
    >>> print(Or)
    Connective(connective_op=OrLuk())
    >>> p = ltn.Predicate(func=lambda x: torch.nn.Sigmoid()(
    ...                                     torch.sum(x, dim=1)
    ...                                  ))
    >>> x = ltn.Variable('x', torch.tensor([[0.56], [0.9]]))
    >>> y = ltn.Variable('y', torch.tensor([[0.7], [0.2], [0.1]]))
    >>> print(p(x).value)
    tensor([0.6365, 0.7109])
    >>> print(p(y).value)
    tensor([0.6682, 0.5498, 0.5250])
    >>> print(Or(p(x), p(y)).value)
    tensor([[1., 1., 1.],
            [1., 1., 1.]])

    .. automethod:: __call__
    """

    def __repr__(self):
        return "OrLogExp()"

    def __call__(self, x, y):
        """
        It applies the Lukasiewicz fuzzy disjunction operator to the given operands.

        Parameters
        -----------
        x : :class:`torch.Tensor`
            First operand on which the operator has to be applied.
        y : :class:`torch.Tensor`
            Second operand on which the operator has to be applied.

        Returns
        -----------
        :class:`torch.Tensor`
            The Lukasiewicz fuzzy disjunction of the two operands.
        """
        max_val = torch.max(torch.stack([x, y]))
        return max_val + torch.log(torch.mean(torch.exp(x - max_val), torch.exp(y - max_val)))

class AggregLogSum(AggregationOperator):
    """
    Min fuzzy aggregation operator.

    :math:`A_{T_{M}}(x_1, \\dots, x_n) = \\operatorname{min}(x_1, \\dots, x_n)`

    Examples
    --------
    >>> import ltn
    >>> import torch
    >>> Forall = ltn.Quantifier(ltn.fuzzy_ops.AggregMin(), quantifier='f')
    >>> print(Forall)
    Quantifier(agg_op=AggregMin(), quantifier='f')
    >>> p = ltn.Predicate(func=lambda x: torch.nn.Sigmoid()(
    ...                                     torch.sum(x, dim=1)
    ...                                  ))
    >>> x = ltn.Variable('x', torch.tensor([[0.56], [0.9], [0.7]]))
    >>> print(p(x).value)
    tensor([0.6365, 0.7109, 0.6682])
    >>> print(Forall(x, p(x)).value)
    tensor(0.6365)

    .. automethod:: __call__
    """

    def __repr__(self):
        return "AggregMin()"

    def __call__(self, xs, p=2, dim=None, keepdim=False, mask=None):
        """
        It applies the min fuzzy aggregation operator to the given formula's :ref:`grounding <notegrounding>` on
        the selected dimensions.

        Parameters
        ----------
        xs : :class:`torch.Tensor`
            :ref:`Grounding <notegrounding>` of formula on which the aggregation has to be performed.
        dim : :obj:`tuple` of :obj:`int`, default=None
            Tuple containing the indexes of dimensions on which the aggregation has to be performed.
        keepdim : :obj:`bool`, default=False
            Flag indicating whether the output has to keep the same dimensions as the input after
            the aggregation.
        mask : :class:`torch.Tensor`, default=None
            Boolean mask for excluding values of 'xs' from the aggregation. It is internally used for guarded
            quantification. The mask must have the same shape of 'xs'. `False` means exclusion, `True` means inclusion.

        Returns
        ----------
        :class:`torch.Tensor`
            Min fuzzy aggregation of the formula.

        Raises
        ------
        :class:`ValueError`
            Raises when the :ref:`grounding <notegrounding>` of the formula ('xs') and the mask do not have the same
            shape.
            Raises when the 'mask' is not boolean.
        """
        if mask is not None:
            if mask.shape != xs.shape:
                raise ValueError("'xs' and 'mask' must have the same shape.")
            if not mask.dtype == torch.bool:
                raise ValueError("'mask' must be a torch.BoolTensor.")
            # here, we put 1 where the mask is not satisfied, since 1 is the maximum value for a truth value.
            # this is a way to exclude values from the minimum computation
            xs = torch.where(~mask, 1., xs.double())
        out = torch.sum(torch.log(xs), dim=dim, keepdim=keepdim)
        return out


class Aggreg_LogSumExp(AggregationOperator):
    """
    Min fuzzy aggregation operator.

    :math:`A_{T_{M}}(x_1, \\dots, x_n) = \\operatorname{min}(x_1, \\dots, x_n)`

    Examples
    --------
    >>> import ltn
    >>> import torch
    >>> Forall = ltn.Quantifier(ltn.fuzzy_ops.AggregMin(), quantifier='f')
    >>> print(Forall)
    Quantifier(agg_op=AggregMin(), quantifier='f')
    >>> p = ltn.Predicate(func=lambda x: torch.nn.Sigmoid()(
    ...                                     torch.sum(x, dim=1)
    ...                                  ))
    >>> x = ltn.Variable('x', torch.tensor([[0.56], [0.9], [0.7]]))
    >>> print(p(x).value)
    tensor([0.6365, 0.7109, 0.6682])
    >>> print(Forall(x, p(x)).value)
    tensor(0.6365)

    .. automethod:: __call__
    """

    def __repr__(self):
        return "Aggreg_LogSumExp()"

    def __call__(self, xs, p=2, dim=None, keepdim=False, mask=None, alpha=1):
        """
        It applies the min fuzzy aggregation operator to the given formula's :ref:`grounding <notegrounding>` on
        the selected dimensions.

        Parameters
        ----------
        xs : :class:`torch.Tensor`
            :ref:`Grounding <notegrounding>` of formula on which the aggregation has to be performed.
        dim : :obj:`tuple` of :obj:`int`, default=None
            Tuple containing the indexes of dimensions on which the aggregation has to be performed.
        keepdim : :obj:`bool`, default=False
            Flag indicating whether the output has to keep the same dimensions as the input after
            the aggregation.
        mask : :class:`torch.Tensor`, default=None
            Boolean mask for excluding values of 'xs' from the aggregation. It is internally used for guarded
            quantification. The mask must have the same shape of 'xs'. `False` means exclusion, `True` means inclusion.

        Returns
        ----------
        :class:`torch.Tensor`
            Min fuzzy aggregation of the formula.

        Raises
        ------
        :class:`ValueError`
            Raises when the :ref:`grounding <notegrounding>` of the formula ('xs') and the mask do not have the same
            shape.
            Raises when the 'mask' is not boolean.
        """
        if mask is not None:
            if mask.shape != xs.shape:
                raise ValueError("'xs' and 'mask' must have the same shape.")
            if not mask.dtype == torch.bool:
                raise ValueError("'mask' must be a torch.BoolTensor.")
            # here, we put 1 where the mask is not satisfied, since 1 is the maximum value for a truth value.
            # this is a way to exclude values from the minimum computation
            xs = torch.where(~mask, 1., xs.double())
        # out = torch.sum(xs,dim=dim,keepdim=keepdim)
        max_val = torch.max(alpha * xs, dim=dim, keepdim=keepdim)
        return (1 / alpha) * (max_val + torch.log(torch.exp((alpha * xs) - max_val)))

class Aggreg_Sum(AggregationOperator):
    """
    Min fuzzy aggregation operator.

    :math:`A_{T_{M}}(x_1, \\dots, x_n) = \\operatorname{min}(x_1, \\dots, x_n)`

    Examples
    --------
    >>> import ltn
    >>> import torch
    >>> Forall = ltn.Quantifier(ltn.fuzzy_ops.AggregMin(), quantifier='f')
    >>> print(Forall)
    Quantifier(agg_op=AggregMin(), quantifier='f')
    >>> p = ltn.Predicate(func=lambda x: torch.nn.Sigmoid()(
    ...                                     torch.sum(x, dim=1)
    ...                                  ))
    >>> x = ltn.Variable('x', torch.tensor([[0.56], [0.9], [0.7]]))
    >>> print(p(x).value)
    tensor([0.6365, 0.7109, 0.6682])
    >>> print(Forall(x, p(x)).value)
    tensor(0.6365)

    .. automethod:: __call__
    """

    def __repr__(self):
        return "Aggreg_LogSumExp()"

    def __call__(self, xs, p=2, dim=None, keepdim=False, mask=None, alpha=1):
        """
        It applies the min fuzzy aggregation operator to the given formula's :ref:`grounding <notegrounding>` on
        the selected dimensions.

        Parameters
        ----------
        xs : :class:`torch.Tensor`
            :ref:`Grounding <notegrounding>` of formula on which the aggregation has to be performed.
        dim : :obj:`tuple` of :obj:`int`, default=None
            Tuple containing the indexes of dimensions on which the aggregation has to be performed.
        keepdim : :obj:`bool`, default=False
            Flag indicating whether the output has to keep the same dimensions as the input after
            the aggregation.
        mask : :class:`torch.Tensor`, default=None
            Boolean mask for excluding values of 'xs' from the aggregation. It is internally used for guarded
            quantification. The mask must have the same shape of 'xs'. `False` means exclusion, `True` means inclusion.

        Returns
        ----------
        :class:`torch.Tensor`
            Min fuzzy aggregation of the formula.

        Raises
        ------
        :class:`ValueError`
            Raises when the :ref:`grounding <notegrounding>` of the formula ('xs') and the mask do not have the same
            shape.
            Raises when the 'mask' is not boolean.
        """
        if mask is not None:
            if mask.shape != xs.shape:
                raise ValueError("'xs' and 'mask' must have the same shape.")
            if not mask.dtype == torch.bool:
                raise ValueError("'mask' must be a torch.BoolTensor.")
            # here, we put 1 where the mask is not satisfied, since 1 is the maximum value for a truth value.
            # this is a way to exclude values from the minimum computation
            xs = torch.where(~mask, 1., xs.double())
        # out = torch.sum(xs,dim=dim,keepdim=keepdim)

        return torch.sum(xs,dim=dim, keepdim=keepdim)



class Aggreg_LogMeanExp(AggregationOperator):

    def __repr__(self):
        return "Aggreg_LogMeanExp()"

    def __call__(self, xs, dim=None, keepdim=False, mask=None, alpha=1,p=None):
        """
        It applies the min fuzzy aggregation operator to the given formula's :ref:`grounding <notegrounding>` on
        the selected dimensions.

        Parameters
        ----------
        xs : :class:`torch.Tensor`
            :ref:`Grounding <notegrounding>` of formula on which the aggregation has to be performed.
        dim : :obj:`tuple` of :obj:`int`, default=None
            Tuple containing the indexes of dimensions on which the aggregation has to be performed.
        keepdim : :obj:`bool`, default=False
            Flag indicating whether the output has to keep the same dimensions as the input after
            the aggregation.
        mask : :class:`torch.Tensor`, default=None
            Boolean mask for excluding values of 'xs' from the aggregation. It is internally used for guarded
            quantification. The mask must have the same shape of 'xs'. `False` means exclusion, `True` means inclusion.

        Returns
        ----------
        :class:`torch.Tensor`
            Min fuzzy aggregation of the formula.

        Raises
        ------
        :class:`ValueError`
            Raises when the :ref:`grounding <notegrounding>` of the formula ('xs') and the mask do not have the same
            shape.
            Raises when the 'mask' is not boolean.
        """
        if mask is not None:
            if mask.shape != xs.shape:
                raise ValueError("'xs' and 'mask' must have the same shape.")
            if not mask.dtype == torch.bool:
                raise ValueError("'mask' must be a torch.BoolTensor.")
            # here, we put 1 where the mask is not satisfied, since 1 is the maximum value for a truth value.
            # this is a way to exclude values from the minimum computation

            max_val= torch.amax(torch.where(~mask, -torch.inf, xs),dim=dim,keepdim=keepdim)
            max_val = torch.where(torch.isinf(max_val),torch.zeros_like(max_val),max_val)

            numerator = torch.sum(torch.where(~mask, torch.zeros_like(xs), xs), dim=dim, keepdim=keepdim)

            # we count the number of 1 in the mask
            denominator = torch.sum(mask, dim=dim, keepdim=keepdim)
            tensor=(1/alpha)*(max_val + torch.log(torch.div(torch.exp((alpha*numerator)-max_val),denominator)))
            return  torch.where(torch.isinf(tensor),torch.nan,tensor)

        else:
            max_val= torch.amax(xs,dim=dim,keepdim=keepdim)

            return (1 / alpha) * (max_val + torch.log(torch.mean(torch.exp(alpha * xs - max_val), dim=dim,keepdim=keepdim)))
        # out = torch.sum(xs,dim=dim,keepdim=keepdim)
        #max_val = torch.amax(xs,dim=dim,keepdim=keepdim)
        #return (1 / alpha) * (max_val + torch.log(torch.mean(torch.exp(alpha * xs - max_val), dim=dim,keepdim=keepdim)).view(-1,1)).view(-1)

class Not_log_negation_softmax(UnaryConnectiveOperator):

    def __repr__(self):
        return "Not_log_negation_softmax()"

    def __call__(self, x):
        #

        x = torch.exp(x)
        x = torch.clamp(x, min=0.001, max=0.99)
        return torch.log(1-x)



class Focalloss(AggregationOperator):
    """
    `pMeanError` fuzzy aggregation operator.

    :math:`A_{pME}(x_1, \\dots, x_n) = 1 - (\\frac{1}{n} \\sum_{i = 1}^n (1 - x_i)^p)^{\\frac{1}{p}}`

    Parameters
    ----------
    p : :obj:`int`, default=2
        Value of hyper-parameter `p` of the `pMeanError` fuzzy aggregation operator.
    stable : :obj:`bool`, default=True
        Flag indicating whether to use the :ref:`stable version <stable>` of the operator or not.

    Attributes
    ----------
    p : :obj:`int`
        See `p` parameter.
    stable : :obj:`bool`
        See `stable` parameter.

    Notes
    -----
    The `pMeanError` aggregation operator has been selected as an approximation of
    :math:`\\forall` with :math:`p \geq 1`. If :math:`p \\to \infty`, then the `pMeanError` operator tends to the
    minimum of the input values (classical behavior of :math:`\\forall`).

    Examples
    --------
    >>> import ltn
    >>> import torch
    >>> Forall = ltn.Quantifier(ltn.fuzzy_ops.AggregPMeanError(), quantifier='f')
    >>> print(Forall)
    Quantifier(agg_op=AggregPMeanError(p=2, stable=True), quantifier='f')
    >>> p = ltn.Predicate(func=lambda x: torch.nn.Sigmoid()(
    ...                                     torch.sum(x, dim=1)
    ...                                  ))
    >>> x = ltn.Variable('x', torch.tensor([[0.56], [0.9], [0.7]]))
    >>> print(p(x).value)
    tensor([0.6365, 0.7109, 0.6682])
    >>> print(Forall(x, p(x)).value)
    tensor(0.6704)

    .. automethod:: __call__
    """

    def __init__(self, p=2, stable=True):
        """
        This constructor has to be used to set whether it has to be used the stable version (it avoids gradient
        problems) of the p-mean error aggregator or not. Also, it is possible to set the value of the parameter p.

        Parameters
        ----------
        p: :obj:`int`
            Value of the parameter p.
        stable: :obj:`bool`
            A boolean flag indicating whether it has to be used the stable version of the aggregator or not.
        """
        self.p = p
        self.stable = stable
        self.alpha = 1
        self.gamma = 1

    def __repr__(self):
        return "AggregPMeanError(p=" + str(self.p) + ", stable=" + str(self.stable) + ")"

    def __call__(self, xs, dim=None, keepdim=False, mask=None, p=None, stable=None):
        """
        It applies the `pMeanError` aggregation operator to the given formula's :ref:`grounding <notegrounding>`
        on the selected dimensions.

        Parameters
        ----------
        xs : :class:`torch.Tensor`
            :ref:`Grounding <notegrounding>` of formula on which the aggregation has to be performed.
        dim : :obj:`tuple` of :obj:`int`, default=None
            Tuple containing the indexes of dimensions on which the aggregation has to be performed.
        keepdim : :obj:`bool`, default=False
            Flag indicating whether the output has to keep the same dimensions as the input after
            the aggregation.
        mask : :class:`torch.Tensor`, default=None
            Boolean mask for excluding values of 'xs' from the aggregation. It is internally used for guarded
            quantification. The mask must have the same shape of 'xs'. `False` means exclusion, `True` means inclusion.
        p : :obj:`int`, default=None
            Value of hyper-parameter `p` of the `pMeanError` fuzzy aggregation operator.
        stable: :obj:`bool`, default=None
            Flag indicating whether to use the :ref:`stable version <stable>` of the operator or not.

        Returns
        ----------
        :class:`torch.Tensor`
            `pMeanError` fuzzy aggregation of the formula.

        Raises
        ------
        :class:`ValueError`
            Raises when the :ref:`grounding <notegrounding>` of the formula ('xs') and the mask do not have the same
            shape.
            Raises when the 'mask' is not boolean.
        """
        p = self.p if p is None else p
        stable = self.stable if stable is None else stable
        xs = xs + 1e-6

        if mask is not None:
            if mask.shape != xs.shape:
                raise ValueError("'xs' and 'mask' must have the same shape.")
            if not mask.dtype == torch.bool:  # isinstance(mask, torch.BoolTensor):
                raise ValueError("'mask' must be a torch.BoolTensor.")

            masked = torch.where(~mask, torch.zeros_like(xs), xs)
            non_zero_values = masked != 0
            zeros = torch.zeros_like(xs)
            zeros[non_zero_values]=torch.log(xs[non_zero_values])





            return -self.alpha *torch.sum(torch.mul(torch.pow((1 - masked),self.gamma),zeros),dim=dim, keepdim=keepdim)

        else:
            return -self.alpha * torch.sum(torch.mul(torch.pow((1 - xs),self.gamma),torch.log(xs)),dim=dim, keepdim=keepdim)






class AggregMean(AggregationOperator):
    """
    `pMeanError` fuzzy aggregation operator.

    :math:`A_{pME}(x_1, \\dots, x_n) = 1 - (\\frac{1}{n} \\sum_{i = 1}^n (1 - x_i)^p)^{\\frac{1}{p}}`

    Parameters
    ----------
    p : :obj:`int`, default=2
        Value of hyper-parameter `p` of the `pMeanError` fuzzy aggregation operator.
    stable : :obj:`bool`, default=True
        Flag indicating whether to use the :ref:`stable version <stable>` of the operator or not.

    Attributes
    ----------
    p : :obj:`int`
        See `p` parameter.
    stable : :obj:`bool`
        See `stable` parameter.

    Notes
    -----
    The `pMeanError` aggregation operator has been selected as an approximation of
    :math:`\\forall` with :math:`p \geq 1`. If :math:`p \\to \infty`, then the `pMeanError` operator tends to the
    minimum of the input values (classical behavior of :math:`\\forall`).

    Examples
    --------
    >>> import ltn
    >>> import torch
    >>> Forall = ltn.Quantifier(ltn.fuzzy_ops.AggregPMeanError(), quantifier='f')
    >>> print(Forall)
    Quantifier(agg_op=AggregPMeanError(p=2, stable=True), quantifier='f')
    >>> p = ltn.Predicate(func=lambda x: torch.nn.Sigmoid()(
    ...                                     torch.sum(x, dim=1)
    ...                                  ))
    >>> x = ltn.Variable('x', torch.tensor([[0.56], [0.9], [0.7]]))
    >>> print(p(x).value)
    tensor([0.6365, 0.7109, 0.6682])
    >>> print(Forall(x, p(x)).value)
    tensor(0.6704)

    .. automethod:: __call__
    """

    def __init__(self, p=2, stable=True):
        """
        This constructor has to be used to set whether it has to be used the stable version (it avoids gradient
        problems) of the p-mean error aggregator or not. Also, it is possible to set the value of the parameter p.

        Parameters
        ----------
        p: :obj:`int`
            Value of the parameter p.
        stable: :obj:`bool`
            A boolean flag indicating whether it has to be used the stable version of the aggregator or not.
        """
        self.p = p
        self.stable = stable

    def __repr__(self):
        return "AggregPMeanError(p=" + str(self.p) + ", stable=" + str(self.stable) + ")"

    def __call__(self, xs, dim=None, keepdim=False, mask=None, p=None, stable=None):
        """
        It applies the `pMeanError` aggregation operator to the given formula's :ref:`grounding <notegrounding>`
        on the selected dimensions.

        Parameters
        ----------
        xs : :class:`torch.Tensor`
            :ref:`Grounding <notegrounding>` of formula on which the aggregation has to be performed.
        dim : :obj:`tuple` of :obj:`int`, default=None
            Tuple containing the indexes of dimensions on which the aggregation has to be performed.
        keepdim : :obj:`bool`, default=False
            Flag indicating whether the output has to keep the same dimensions as the input after
            the aggregation.
        mask : :class:`torch.Tensor`, default=None
            Boolean mask for excluding values of 'xs' from the aggregation. It is internally used for guarded
            quantification. The mask must have the same shape of 'xs'. `False` means exclusion, `True` means inclusion.
        p : :obj:`int`, default=None
            Value of hyper-parameter `p` of the `pMeanError` fuzzy aggregation operator.
        stable: :obj:`bool`, default=None
            Flag indicating whether to use the :ref:`stable version <stable>` of the operator or not.

        Returns
        ----------
        :class:`torch.Tensor`
            `pMeanError` fuzzy aggregation of the formula.

        Raises
        ------
        :class:`ValueError`
            Raises when the :ref:`grounding <notegrounding>` of the formula ('xs') and the mask do not have the same
            shape.
            Raises when the 'mask' is not boolean.
        """
        p = self.p if p is None else p
        stable = self.stable if stable is None else stable

        if mask is not None:
            if mask.shape != xs.shape:
                raise ValueError("'xs' and 'mask' must have the same shape.")
            if not mask.dtype == torch.bool:  # isinstance(mask, torch.BoolTensor):
                raise ValueError("'mask' must be a torch.BoolTensor.")
            # we sum the values of xs which are not filtered out by the mask
            numerator = torch.sum(torch.where(~mask, torch.zeros_like(xs), xs), dim=dim, keepdim=keepdim)
            # we count the number of 1 in the mask
            denominator = torch.sum(mask, dim=dim, keepdim=keepdim)
            return torch.div(numerator, denominator)
        else:
            return torch.mean(xs, dim=dim, keepdim=keepdim)


class And_Sum(BinaryConnectiveOperator):
    """
    Goguen fuzzy conjunction operator (product operator).

    :math:`\land_{Goguen}(x, y) = xy`

    Parameters
    ----------
    stable : :obj:`bool`, default=True
        Flag indicating whether to use the :ref:`stable version <stable>` of the operator or not.

    Attributes
    ----------
    stable : :obj:`bool`
        See `stable` parameter.

    Notes
    -----
    The Gougen fuzzy conjunction could have vanishing gradients if not used in its :ref:`stable <stable>` version.

    Examples
    --------
    Note that:

    - variable `x` has two individuals;
    - variable `y` has three individuals;
    - the shape of the result of the conjunction is `(2, 3)` due to the :ref:`LTN broadcasting <broadcasting>`. The first dimension is dedicated two variable `x`, while the second dimension to variable `y`;
    - at index `(0, 0)` there is the evaluation of the formula on first individual of `x` and first individual of `y`, at index `(0, 1)` there is the evaluation of the formula on first individual of `x` and second individual of `y`, and so forth.

    >>> import ltn
    >>> import torch
    >>> And = ltn.Connective(ltn.fuzzy_ops.AndProd())
    >>> print(And)
    Connective(connective_op=AndProd(stable=True))
    >>> p = ltn.Predicate(func=lambda x: torch.nn.Sigmoid()(
    ...                                     torch.sum(x, dim=1)
    ...                                  ))
    >>> x = ltn.Variable('x', torch.tensor([[0.56], [0.9]]))
    >>> y = ltn.Variable('y', torch.tensor([[0.7], [0.2], [0.1]]))
    >>> print(p(x).value)
    tensor([0.6365, 0.7109])
    >>> print(p(y).value)
    tensor([0.6682, 0.5498, 0.5250])
    >>> print(And(p(x), p(y)).value)
    tensor([[0.4253, 0.3500, 0.3342],
            [0.4751, 0.3910, 0.3733]])

    .. automethod:: __call__
    """

    def __init__(self, stable=True):
        """
        This constructor has to be used to set whether it has to be used the stable version (it avoids gradient
        problems) of the Goguen fuzzy conjunction or not.

        Parameters
        -----------
        stable : :obj:`bool`, default=True
            A boolean flag indicating whether it has to be used the stable version of the operator or not.
        """
        self.stable = stable

    def __repr__(self):
        return "AndSum(stable=" + str(self.stable) + ")"

    def __call__(self, x, y, stable=None):
        """
        It applies the Goguen fuzzy conjunction operator to the given operands.

        Parameters
        ----------
        x : :class:`torch.Tensor`
            First operand on which the operator has to be applied.
        y : :class:`torch.Tensor`
            Second operand on which the operator has to be applied.
        stable : :obj:`bool`, default=None
            Flag indicating whether to use the :ref:`stable version <stable>` of the operator or not.

        Returns
        -----------
        :class:`torch.Tensor`
            The Goguen fuzzy conjunction of the two operands.
        """
        stable = self.stable if stable is None else stable
        if stable:
            x, y = pi_0(x), pi_0(y)
        return x + y



class SatAgg:
    """
    `SatAgg` aggregation operator.

    :math:`\operatorname{SatAgg}_{\phi \in \mathcal{K}} \mathcal{G}_{\\theta} (\phi)`

    It aggregates the truth values of the closed formulas given in input, namely the formulas
    :math:`\phi_1, \dots, \phi_n` contained in the knowledge base :math:`\mathcal{K}`. In the notation,
    :math:`\mathcal{G}_{\\theta}` is the :ref:`grounding <notegrounding>` function, parametrized by :math:`\\theta`.

    Parameters
    ----------
    agg_op : :class:`ltn.fuzzy_ops.AggregationOperator`, default=AggregPMeanError(p=2)
        Fuzzy aggregation operator used by the `SatAgg` operator to perform the aggregation.

    Attributes
    ----------
    agg_op : :class:`ltn.fuzzy_ops.AggregationOperator`, default=AggregPMeanError(p=2)
        See `agg_op` parameter.

    Raises
    ----------
    :class:`TypeError`
        Raises when the type of the input parameter is not correct.

    Notes
    -----
    - `SatAgg` is particularly useful for computing the overall satisfaction level of a knowledge base when :ref:`learning <notelearning>` a Logic Tensor Network;
    - the result of the `SatAgg` aggregation is a scalar. It is the satisfaction level of the knowledge based composed of the closed formulas given in input.

    Examples
    --------
    `SatAgg` can be used to aggregate the truth values of formulas contained in a knowledge base. Note that:

    - `SatAgg` takes as input a tuple of :class:`ltn.core.LTNObject` and/or :class:`torch.Tensor`;
    - when some :class:`torch.Tensor` are given to `SatAgg`, they have to be scalars in [0., 1.] since `SatAgg` is designed to work with closed formulas;
    - in this example, our knowledge base is composed of closed formulas `f1`, `f2`, and `f3`;
    - `SatAgg` applies the `pMeanError` aggregation operator to the truth values of these formulas. The result is a new truth value which can be interpreted as a satisfaction level of the entire knowledge base;
    - the result of `SatAgg` is a :class:`torch.Tensor` since it has been designed for learning in PyTorch. The idea is to put the result of the operator directly inside the loss function of the LTN. See this `tutorial <https://nbviewer.jupyter.org/github/bmxitalia/LTNtorch/blob/main/tutorials/3-knowledgebase-and-learning.ipynb>`_ for a detailed example.

    >>> import ltn
    >>> import torch
    >>> x = ltn.Variable('x', torch.tensor([[0.1, 0.03],
    ...                                     [2.3, 4.3]]))
    >>> y = ltn.Variable('y', torch.tensor([[3.4, 2.3],
    ...                                     [5.4, 0.43]]))
    >>> p = ltn.Predicate(func=lambda x: torch.nn.Sigmoid()(
    ...                                     torch.sum(x, dim=1)
    ...                                  ))
    >>> q = ltn.Predicate(func=lambda x, y: torch.nn.Sigmoid()(
    ...                                         torch.sum(torch.cat([x, y], dim=1),
    ...                                     dim=1)))
    >>> Forall = ltn.Quantifier(ltn.fuzzy_ops.AggregPMeanError(), quantifier='f')
    >>> And = ltn.Connective(ltn.fuzzy_ops.AndProd())
    >>> f1 = Forall(x, p(x))
    >>> f2 = Forall([x, y], q(x, y))
    >>> f3 = And(Forall([x, y], q(x, y)), Forall(x, p(x)))
    >>> sat_agg = ltn.fuzzy_ops.SatAgg(ltn.fuzzy_ops.AggregPMeanError())
    >>> print(sat_agg)
    SatAgg(agg_op=AggregPMeanError(p=2, stable=True))
    >>> out = sat_agg(f1, f2, f3)
    >>> print(type(out))
    <class 'torch.Tensor'>
    >>> print(out)
    tensor(0.7294)

    In the previous example, some closed formulas (:class:`ltn.core.LTNObject`) have been given to the `SatAgg`
    operator.
    In this example, we show that `SatAgg` can take as input also :class:`torch.Tensor` containing the result of some
    closed formulas, namely scalars in [0., 1.]. Note that:

    - `f2` is just a :class:`torch.Tensor`;
    - since `f2` contains a scalar in [0., 1.], its value can be interpreted as a truth value of a closed formula. For this reason, it is possible to give `f2` to the `SatAgg` operator to get the aggregation of `f1` (:class:`ltn.core.LTNObject`) and `f2` (:class:`torch.Tensor`).

    >>> x = ltn.Variable('x', torch.tensor([[0.1, 0.03],
    ...                                     [2.3, 4.3]]))
    >>> p = ltn.Predicate(func=lambda x: torch.nn.Sigmoid()(
    ...                                     torch.sum(x, dim=1)
    ...                                  ))
    >>> Forall = ltn.Quantifier(ltn.fuzzy_ops.AggregPMeanError(), quantifier='f')
    >>> f1 = Forall(x, p(x))
    >>> f2 = torch.tensor(0.7)
    >>> sat_agg = ltn.fuzzy_ops.SatAgg(ltn.fuzzy_ops.AggregPMeanError())
    >>> print(sat_agg)
    SatAgg(agg_op=AggregPMeanError(p=2, stable=True))
    >>> out = sat_agg(f1, f2)
    >>> print(type(out))
    <class 'torch.Tensor'>
    >>> print(out)
    tensor(0.6842)

    .. automethod:: __call__
    """

    def __init__(self, agg_op=AggregMean()):
        """
        This is the constructor of the SatAgg operator.

        It takes as input an aggregation operator which define the behavior of SatAgg.

        Parameters
        ----------
        agg_op: :class:`AggregationOperator`
            Aggregation operator which implements the SatAgg aggregation. By default is the pMeanError with p=2.

        Raises
        ----------
        :class:`TypeError`
            Raises when the type of the input parameter is not correct.
        """
        if not isinstance(agg_op, AggregationOperator):
            raise TypeError("SatAgg() : argument 'agg_op' (position 1) must be an AggregationOperator, not " +
                            str(type(agg_op)))
        self.agg_op = agg_op

    def __repr__(self):
        return "SatAgg(agg_op=" + str(self.agg_op) + ")"

    def __call__(self, *closed_formulas, p=None):
        """
        It applies the `SatAgg` aggregation operator to the given closed formula's :ref:`groundings <notegrounding>`.

        Parameters
        ----------
        closed_formulas : :obj:`tuple` of :class:`ltn.core.LTNObject` and/or :class:`torch.Tensor`
            Tuple of closed formulas (`LTNObject` and/or tensors) for which the aggregation has to be computed.

        Returns
        ----------
        :class:`torch.Tensor`
            The result of the `SatAgg` aggregation.

        Raises
        ----------
        :class:`TypeError`
            Raises when the type of the input parameter is not correct.

        :class:`ValueError`
            Raises when the truth values of the formulas/tensors given in input are not in the range [0., 1.].
            Raises when the truth values of the formulas/tensors given in input are not scalars, namely some formulas
            are not closed formulas.
        """
        # The closed formulas are identifiable since they are just scalar because all the variables
        # have been quantified (i.e., all dimensions have been aggregated).


        if len(list(closed_formulas)) == 1:
            truth_values = [list(closed_formulas)[0]]
        else:
            truth_values = list(closed_formulas)

        if not all(isinstance(x, (LTNObject, torch.Tensor)) for x in truth_values):
            raise TypeError("Expected parameter 'closed_formulas' to be a tuple of LTNObject and/or tensors, "
                            "but got " + str([type(f) for f in closed_formulas]))
        truth_values2 = [o.value if isinstance(o, LTNObject) else o for o in truth_values]
        if not all([f.shape == torch.Size([]) for f in truth_values2]):
            raise ValueError("Expected parameter 'closed_formulas' to be a tuple of LTNObject and/or tensors "
                             "containing scalars, but got the following shapes: " +
                             str([f.shape() for f in closed_formulas]))
        """
        for t in range(len(truth_values)):
            if hasattr(truth_values[t], 'weight'):
                truth_values2[t].weight=truth_values[t].weight
        """

        truth_values2 = torch.stack(truth_values2, dim=0)
        truth_values2.weight = [f.weight for f in truth_values if hasattr(f, "weight")]
        # check truth values of operands are in [0., 1.] before computing the SatAgg aggregation
        check_values(truth_values2)

        return self.agg_op(truth_values2, dim=0, p=p)
