from dataclasses import dataclass
from typing import Any, Iterable, List, Tuple

from typing_extensions import Protocol

# ## Task 1.1
# Central Difference calculation


def central_difference(f: Any, *vals: Any, arg: int = 0, epsilon: float = 1e-6) -> Any:  
    r"""
    Computes an approximation to the derivative of `f` with respect to one arg.

    See :doc:`derivative` or https://en.wikipedia.org/wiki/Finite_difference for more details.

    Args:
        f : arbitrary function from n-scalar args to one value
        *vals : n-float values $x_0 \ldots x_{n-1}$
        arg : the number $i$ of the arg to compute the derivative
        epsilon : a small constant

    Returns:
        An approximation of $f'_i(x_0, \ldots, x_{n-1})$
    """
    # TODO: Implement for Task 1.1.
    # 求偏导
    tmp_vals = list(vals)
    tmp_vals[arg] += epsilon
    return (f(*tmp_vals)-f(*vals))/epsilon
    # raise NotImplementedError("Need to implement for Task 1.1")


variable_count = 1


class Variable(Protocol):
    def accumulate_derivative(self, x: Any) -> None:
        pass

    @property
    def unique_id(self) -> int:
        pass

    def is_leaf(self) -> bool:
        pass

    def is_constant(self) -> bool:
        pass

    @property
    def parents(self) -> Iterable["Variable"]:
        pass

    def chain_rule(self, d_output: Any) -> Iterable[Tuple["Variable", Any]]:
        pass


def topological_sort(variable: Variable) -> Iterable[Variable]:  # 计算图 - 有向无环图 -> 拓扑排序
    """
    Computes the topological order of the computation graph.

    Args:
        variable: The right-most variable

    Returns:
        Non-constant Variables in topological order starting from the right.
    """
    # TODO: Implement for Task 1.4.
    # e.g. b=(x+y)+z, c=(x+y)*w -> [x, y, a=x+y, z, b=a+z, w, c=a*w]
    vis = set()
    topo = []
    def dfs(u: Variable):
        if u.unique_id in vis or u.is_constant():  # 访问过了/常量
            return
        vis.add(u.unique_id)  # 标记vis
        for v in u.parents:  # 遍历父节点
            dfs(v)
        topo.append(u)  # 加入list
    dfs(variable)
    topo.reverse()
    return topo
    # raise NotImplementedError("Need to implement for Task 1.4")


def backpropagate(variable: Variable, deriv: Any) -> None:  # variable=f(x1, x2, …) - deriv = dL / dvariable
    """
    Runs backpropagation on the computation graph in order to
    compute derivatives for the leave nodes.

    Args:
        variable: The right-most variable
        deriv  : Its derivative that we want to propagate backward to the leaves.

    No return. Should write to its results to the derivative values of each leaf through `accumulate_derivative`.
    """
    # TODO: Implement for Task 1.4.
    d = {}  # 相当于cpp里的map - <v.unique_id, deriv>
    d[variable.unique_id] = deriv  # 初始化variable的导数
    topo = topological_sort(variable)
    for u in topo:
        if u.is_constant() or u.is_leaf():  # 常量/叶子节点不需要计算梯度
            continue
        for v, grad_v in u.chain_rule(d[u.unique_id]):
            if v.unique_id in d:
                d[v.unique_id] += grad_v  # 累加梯度
            else:  # v.unique_id第一次出现
                d[v.unique_id] = grad_v
    for u in topo:
        if u.is_leaf():
            u.accumulate_derivative(d[u.unique_id])
    #raise NotImplementedError("Need to implement for Task 1.4")


@dataclass
class Context:
    """
    Context class is used by `Function` to store information during the forward pass.
    """

    no_grad: bool = False
    saved_values: Tuple[Any, ...] = ()

    def save_for_backward(self, *values: Any) -> None:
        "Store the given `values` if they need to be used during backpropagation."
        if self.no_grad:
            return
        self.saved_values = values

    @property
    def saved_tensors(self) -> Tuple[Any, ...]:
        return self.saved_values
