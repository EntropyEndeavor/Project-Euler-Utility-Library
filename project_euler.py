"""A collection of useful functions for solving Project Euler problems.

All functions are experimental, and are not guaranteed to work in all cases.
A timer is automatically started when the module is imported and timing
information can be printed by calling stop_clock().
"""
from itertools import combinations, compress, count
from typing import Iterable, Generator, Optional
from platform import python_implementation
from math import factorial, isqrt, prod
from time import time, process_time


_TIMING = (time(), process_time())


def stop_clock() -> None:
    """Print timing information from the point when the module was first imported."""
    end_time = (time(), process_time())

    wall_time = end_time[0] - _TIMING[0]
    proc_time = end_time[1] - _TIMING[1]

    is_pypy = python_implementation() == 'PyPy'
    instance = " (PyPy)" if is_pypy else ""

    print(f"Wall time{instance}: {wall_time:.2f} seconds.")
    print(f"Process time{instance}: {proc_time:.4f} seconds.")


def is_square(x: int) -> bool:
    """Test if a number is a perfect square."""
    return x == isqrt(x)**2


def digit_sum(n: int) -> int:
    """Compute the sum of the digits of a number."""
    ans = 0
    while n != 0:
        ans += n % 10
        n //= 10

    return ans


def digital_root(n: int) -> int:
    """Compute the digital root of a number.

    The digital root is found by computing the sum of the digits
    iteratively until the result is a single digit number.
    """
    while n >= 10:
        n = digit_sum(n)
    return n


def polygonal_number(n: int, side: int) -> int:
    """Calculate the nth polygonal number of the given side length."""
    return ((side-2)*(n**2) - (side-4)*n)//2


def multinomial(n: int, rs: Iterable[int]) -> int:
    """Compute a multinomial coefficient.

    This function assumes that the the sum of the
    values in rs will be exactly equal to n.
    """
    return factorial(n)//prod(factorial(r) for r in rs)


def xor_mul(a: int, b: int) -> int:
    """Perform binary long multiplication with bitwise XOR instead of addition.

    The XOR multiplication was definied in Problem 810
    and has been used in a number of problems since.
    This function will not work correctly in SageMath due to 
    the bitwise XOR operator being treated as exponentiation.
    """
    ans = 0
    while a != 0:
        if a & 1:
            ans ^= b
        a >>= 1
        b <<= 1

    return ans


class ModFrac:
    """Store intermediate results as a fraction when a modular inverse would be used.

    Many Project Euler problems ask for an answer to be returned modulo some number.
    There are many reasons that computing the answer might require modular inverses,
    for example if the solution includes the computation of many choose functions.
    While there are efficient ways to compute a range of modular inverses, this can
    be avoided by leaving the entire computation as a fraction and only computing
    one modular inverse at the very end.
    """
    _mod = None

    def __init__(self, n: int = 0, d: int = 1):
        """Initialize a modular fraction.

            The default arguments are such that it will default to
            zero when no arguments are provided, similar to other
            numeric types, allowing it to be used easily as the value
            type in a defaultdict.

            When one argument is provided it will be equivalent to an
            integer with that value.

            The modulus used is global (i.e. all ModFrac objects will have 
            the same modulus) and must be set before a ModFrac is created.
        """
        if ModFrac._mod is None:
            raise RuntimeError("Global modulus must be set before creating a ModFrac.")
        self._n = n % ModFrac._mod
        self._d = d % ModFrac._mod

    @classmethod
    def set_mod(cls, mod: int) -> None:
        """Set the modulus used for all ModFrac objects.

        This must be done to create a ModFrac, and can only be done once.
        """
        if cls._mod is not None:
            raise RuntimeError("Once set, the modulus cannot be changed for ModFrac.")
        cls._mod = mod

    def __add__(self, other: "ModFrac") -> "ModFrac":
        return ModFrac(self._n*other._d + other._n*self._d, other._d*self._d)

    def __mul__(self, other: "ModFrac") -> "ModFrac":
        return ModFrac(self._n*other._n, other._d*self._d)

    def __int__(self) -> int:
        return (self._n * pow(self._d, -1, ModFrac._mod)) % ModFrac._mod

    def __repr__(self) -> str:
        return f"({self._n}/{self._d}) % {ModFrac._mod}"


class CachedModFactorials:
    """Cache the values of factorials reduced by the given modulus.

    Should be used when you expect to need the values of many different factorials.
    """
    def __init__(self, mod: int):
        self._mod = mod
        self._cache = [1]
        self._num = 0

    def extend(self, n: int) -> None:
        """Make sure the first n values are available in the cache."""
        while self._num < n:
            self._num += 1
            self._cache.append((self._cache[-1] * self._num) % self._mod)

    def get(self, n: int) -> int:
        """Get the remainder of factorial(n) with the given modulus.""" 
        self.extend(n)

        return self._cache[n]


class CachedModPowers:
    """Cache the values of powers of the base reduced by the given modulus.

    Should be used when you expect to need the values of many different powers of the base.
    """
    def __init__(self, base: int, mod: int):
        self._mod = mod
        self._base = base
        self._cache = [1]
        self._num = 0

    def extend(self, n: int) -> None:
        """Make sure the first n values are available in the cache."""
        while self._num < n:
            self._num += 1
            self._cache.append((self._cache[-1] * self._base) % self._mod)

    def get(self, n: int) -> int:
        """Get the remainder of base**n with the given modulus."""
        self.extend(n)

        return self._cache[n]


def prime_sieve(upperlimit: int) -> list[int]:
    """Use the Sieve of Eratosthenes to find all primes up to the limit."""
    primality = [True for _ in range(upperlimit+1)]
    primality[0] = False
    primality[1] = False

    for i in range(isqrt(upperlimit)+1):
        if primality[i]:
            j = i**2
            while j <= upperlimit:
                primality[j] = False
                j += i

    return list(compress(count(), primality))


def prime_factor(num: int) -> list[int]:
    """Completely factor a number.

    If the exponent of a prime factor is larger than one, then it will
    appear in the list a number of times equal to its exponent. For example,
    if asked to factor 12 the result will be [2,2,3].
    """
    primes = prime_sieve(isqrt(num))

    return factor_out_primes(num, primes)


def factor_out_primes(num: int, primes: list[int]):
    """Factor out all instances of the provided primes from a number.

    If the exponent of a prime factor is larger than one, then it will
    appear in the list a number of times equal to its exponent. For example,
    if asked to factor 12 the result will be [2,2,3].

    If the number contains more than one prime factor that is not
    included in the list of primes, then the number will not be fully
    factored.
    """
    limit = isqrt(num)

    factors = []

    for p in primes:
        if p > limit:
            break

        while (num % p) == 0:
            factors.append(p)
            num //= p
            limit = isqrt(num)

    if 1 != num:
        factors.append(num)

    return factors


def partitions(n: int, smallest_allowed: Optional[int] = None,
               biggest_allowed: Optional[int] = None) -> Generator[list[int],None,None]:
    """Find the partitions of a number.

    By default all partitions are generated. The smallest and largest
    values for the elements inside the partion can also be controlled.
    """
    if smallest_allowed is None:
        smallest_allowed = 1

    if biggest_allowed is None:
        biggest_allowed = n

    if n == 0:
        yield []
        return

    for e in range(min(n, biggest_allowed), smallest_allowed - 1, -1):
        for rest in partitions(n-e, smallest_allowed, e):
            yield [e] + rest

def factorization_exponent_sequences() -> Generator[list[int],None,None]:
    """Find unique exponent sequences for prime factorizations.

    All the exponents for a sequence are in order from largest to
    smallest. The sequences will be produced in batches with the same
    number of total prime factors. The smallest representative example
    of a number having each sequence can be found by taking a list of
    primes from smallest to largest and applying each exponent to the
    corresponding prime.
    """
    for num_primes in count(1):
        yield from partitions(num_primes)


def subsets_less_when_mul(numbers: list[float], limit: float,
                          min_num_elements: int = 0, max_num_elements: Optional[int] = None,
                          *, _next_pos: int = 0) -> Generator[list[float],None,None]:
    """Find all subsets where the product of the elements is less than the limit.

    By default subsets of any size will be generated, but limits can be provided for the size.

    The _next_pos argument is for internal use only, and should not be provided.
    The numbers array must be in sorted order with the least element first.
    WARNING: This function does not work correctly if the limit is less than one. 
    """
    if max_num_elements is None:
        max_num_elements = len(numbers)

    if min_num_elements > max_num_elements:
        return

    if min_num_elements > len(numbers) - _next_pos:
        # We don't have enough elements left to fulfill our minimum.
        return

    if max_num_elements == 0 or len(numbers) == _next_pos:
        # We aren't allowed to place any more elements.
        yield []
        return

    if min_num_elements == 0:
        # We are allowed to return an empty set.
        yield []

    for ix in range(_next_pos, len(numbers)):
        num = numbers[ix]
        if num > limit:
            return

        # Recursively find answers within the new limit. Since we are looking at the
        # smallest number left, if we can't find any solutions at all then we
        # won't find any solutions by picking a larger number next, so exit early.
        seen_any = False
        for s in subsets_less_when_mul(numbers, limit/num, max(0, min_num_elements - 1),
                                       max(0, max_num_elements - 1), _next_pos = ix + 1):
            seen_any = True
            yield [num] + s

        if not seen_any:
            return


def split_combinations(items: list, left_size: int) -> Generator:
    """Find all combinations of elements while also retriving the non-included elements."""
    for indices in combinations(range(len(items)), left_size):
        yield ([i for ix, i in enumerate(items) if ix in indices],
               [i for ix, i in enumerate(items) if not ix in indices])


def all_subsets(items: list, min_size: int = 0, max_size: Optional[int] = None) -> Generator:
    """Generate all subsets of a list.

    By default all subsets are generated, but limits can be placed on the subset size.
    """
    if max_size is None:
        max_size = len(items)

    for size in range(min_size, max_size + 1):
        yield from combinations(items, size)


_MILLER_RABIN_TESTS = ((2047, (2,)),
                       (1373653, (2,3)),
                       (9080191, (31,73)),
                       (25326001, (2,3,5)),
                       (3215031751, (2,3,5,7)),
                       (4759123141, (2,7,61)),
                       (1122004669633, (2,13,23,1662803)),
                       (2152302898747, (2,3,5,7,11)),
                       (3474749660383, (2,3,5,7,11,13)),
                       (341550071728321, (2,3,5,7,11,13,17)),
                       (3825123056546413051, (2,3,5,7,11,13,17,19,23)),
                       (18446744073709551616, (2,3,5,7,11,13,17,19,23,29,31,37)),
                       (318665857834031151167461, (2,3,5,7,11,13,17,19,23,29,31,37)),
                       (3317044064679887385961981, (2,3,5,7,11,13,17,19,23,29,31,37,41)))


def miller_rabin(n: int) -> bool:
    """Use Miller Rabin to test for primality.

    The checks that should be used are determined based on the size of the value.
    """
    for limit, checks in _MILLER_RABIN_TESTS:
        if n < limit:
            return miller_rabin_c(n, checks)

    raise ValueError("Argument exceeds limits of known checks.")


def miller_rabin_c(n: int, to_check: Iterable[int]) -> bool:
    """Use the provided checks to test for primality with Miller Rabin."""
    if n == 1:
        return False

    if n in to_check:
        return True

    d = n-1
    s = 0
    while d % 2 == 0:
        d = d//2
        s += 1

    for a in to_check:
        if 1 != pow(a,d,n) and all(n-1 != pow(a, 2**r * d, n) for r in range(s)):
            return False

    return True


def present_in_sorted(n: float, s: list[float]) -> bool:
    """Determine if a given element is present inside a sorted list."""
    possible_index = smallest_index_ge_in_sorted(n, s)

    if possible_index is None:
        return False

    return s[possible_index] == n


def smallest_index_ge_in_sorted(target: float, s: list[float]) -> int:
    """Find the index of the first number in a sorted list greater than or equal to the target."""
    if not s or s[-1] < target:
        return None

    if s[0] >= target:
        return 0

    bottom = 0
    top = len(s) - 1

    while top - bottom > 1:
        middle = (top + bottom)//2

        if s[middle] < target:
            bottom = middle
        else:
            top = middle

    return top


def crt(a: int, m: int, b: int, n: int) -> int:
    """Find the remainder that satisfies two modular constraints with Chinese Remainder Theorem."""
    k = pow(m,-1,n)*(b-a)

    return (a + k*m) % (m*n)


class RedBlackTree:
    """Implement a self-balancing binary search tree.

    This tree implementation is heavily based on the version in
    Introduction to Algorithms (3rd ed) by Cormen, Leiserson,
    Rivest and Stein.

    Be aware that the tree functions may sometimes return a NIL node.
    A NIL node may have arbitrary and meaningless values in the
    parent, left, right and value attributes. A NIL node will evaluate
    to boolean False, and all other nodes will evaulate to boolean True.

    An extra attribute is reserved inside each node in case you want to
    store additional information about a tree state. For example you might
    want to store the sum of all values below a node in the tree, which can
    be transformed to find the sum of the values less than the value in the
    node. Such extra data must be kept up to date as the tree changes. Hook
    functions starting with _extra_info are provided to make updates at the
    relavent points in the tree functions. They do nothing in the base
    implementation, but you can override any/all of them as necessary.

    It is possible to disable the self balancing features in this tree.
    This may be helpful if you want to store extra information, but do
    not want to implement the more complicated logic needed to update this
    information for the tree rotations.
    """
    # pylint: disable=protected-access
    class Node:
        """A Node in a self balancing binary search tree.

        Attribute value contains the value of the node.
        
        Tree functions will reutrn nodes to you, but you
        should never be creating a node yourself. Returned
        nodes should be checked for NIL. 

        The tree exposes hooks into certain functions to allow
        you to store extra information in the extra attribute,
        and then potentially update it as the tree changes.
        """
        def __init__(self, value, nil_node, is_nil: bool = False):
            self.parent = nil_node
            self.left = nil_node
            self.right = nil_node
            self.value = value
            self.extra = None
            self._is_nil = is_nil
            self._red = not is_nil

        def is_nil(self):
            """Check if this is a NIL node.

            NIL nodes will automatically evaulate to boolean False, and all other
            nodes will evaluate to True (even if they contain value zero, for example).

            NIL nodes may have arbitrary data in value, parent, left and right.
            None of these nodes should be traversed from a NIL node.
            """
            return self._is_nil

        def _is_red(self):
            return self._red

        def _is_black(self):
            return not self._red

        def __bool__(self) -> bool:
            return not self.is_nil()

        def __str__(self) -> str:
            """Print just the value of this node."""
            if self.is_nil():
                return "Node(nil)"

            return f"Node({self.value!r})"

        def __repr__(self) -> str:
            """Print the value of this and all child nodes."""
            if self.is_nil():
                return "nil"

            return f"[{self.value!r} {self.left!r} {self.right!r}]"

    def __init__(self, *, balance: bool = True):
        self._nil = self.Node(None, None, is_nil = True)

        self.root = self._nil
        self._balance = balance

    def _iterate_subtree(self, node):
        if node:
            yield from self._iterate_subtree(node.left)

            yield node.value

            yield from self._iterate_subtree(node.right)

    def __iter__(self):
        """Iterate through all the values in the tree in sorted order."""
        yield from self._iterate_subtree(self.root)

    def find(self, value, *, direction: float = 0):
        """Find the node containing a certain value in the tree.

        The tree may contain multiple nodes with the same value, in
        which case any one of them may be returned by this function.

        By default this function will reutrn the NIL node if the value
        is not found in the tree. The direction argument can be used
        to have this function return the node with the largest value
        below the target value (if negative) or the smallest value above
        (if positive). In all cases, it is still possible for the NIL
        node to be returned if there are no nodes meeting the criteria.
        """
        x = self.root

        if not x:
            return x

        while x.value != value:
            if value < x.value:
                if x.left:
                    x = x.left
                else:
                    break
            else:
                if x.right:
                    x = x.right
                else:
                    break

        if x.value != value:
            if 0 == direction:
                return self._nil

            if direction < 0 and x.value > value:
                return self.prev_node(x)

            if direction > 0 and x.value < value:
                return self.next_node(x)

        return x

    def minimum(self, node = None):
        """Return the node with the smallest value.

        By default, the smallest value in the entire tree is found.
        You can optionally provided a node to only search the subtree
        rooted at that node. The NIL node can be returned if the tree is empty.
        """
        if node is None:
            node = self.root
            if not node:
                return node

        while node.left:
            node = node.left

        return node

    def maximum(self, node = None):
        """Return the node with the largest value.

        By default, the largest value in the entire tree is found.
        You can optionally provided a node to only search the subtree
        rooted at that node. The NIL node can be returned if the tree is empty.
        """
        if node is None:
            node = self.root
            if not node:
                return node

        while node.right:
            node = node.right

        return node

    def prev_node(self, node):
        """Find the previous node.

        Can return the NIL node if the node with smallest value was provided.
        """
        if node.left:
            return self.maximum(node.left)

        y = node.parent
        while y and node == y.left:
            node = y
            y = y.parent

        return y

    def next_node(self, node):
        """Find the next node.

        Can return the NIL node if the node with largest value was provided.
        """
        if node.right:
            return self.minimum(node.right)

        y = node.parent
        while y and node == y.right:
            node = y
            y = y.parent

        return y

    def insert(self, value):
        """Add a new node with the given value into the tree."""
        y = self._nil
        x = self.root

        while x:
            y = x
            if value < x.value:
                self._extra_info_insert_traversed_node(x, value, -1)
                x = x.left
            else:
                self._extra_info_insert_traversed_node(x, value, 1)
                x = x.right

        node = self.Node(value, self._nil)
        node.parent = y

        if not y:
            # Tree was empty.
            self.root = node
        elif value < y.value:
            y.left = node
        else:
            y.right = node

        self._extra_info_insert_created_node(node)

        if self._balance:
            self._rb_insert_fixup(node)

        return node

    def delete(self, node):
        """Remove a node from the tree."""
        if not node:
            raise ValueError("Can't delete the NIL node.")

        y = node
        y_was_black = y._is_black()

        self._extra_info_delete_removed_node(node)
        if not node.left:
            x = node.right
            self._extra_info_delete_moved_node(x, node.parent, None, None)
            self._transplant(node, x)
        elif not node.right:
            x = node.left
            self._extra_info_delete_moved_node(x, node.parent, None, None)
            self._transplant(node, x)
        else:
            y = self.minimum(node.right)
            y_was_black = y._is_black()
            x = y.right
            if y.parent == node:
                x.parent = y
                self._extra_info_delete_moved_node(y, node.parent, node.left, None)
            else:
                self._extra_info_delete_removed_node(y)
                self._extra_info_delete_moved_node(x, y.parent, None, None)
                self._extra_info_delete_moved_node(y, node.parent, node.left, node.right)
                self._transplant(y, x)
                y.right = node.right
                y.right.parent = y
            self._transplant(node, y)
            y.left = node.left
            y.left.parent = y
            y._red = node._red

        if self._balance and y_was_black:
            self._rb_delete_fixup(x)

    def _rotate_left(self, node):
        self._extra_info_balance_left_rotate(node)
        y = node.right
        node.right = y.left

        if y.left:
            y.left.parent = node

        y.parent = node.parent
        if not node.parent:
            self.root = y
        elif node == node.parent.left:
            node.parent.left = y
        else:
            node.parent.right = y

        y.left = node
        node.parent = y

    def _rotate_right(self, node):
        self._extra_info_balance_right_rotate(node)
        y = node.left
        node.left = y.right

        if y.right:
            y.right.parent = node

        y.parent = node.parent
        if not node.parent:
            self.root = y
        elif node == node.parent.right:
            node.parent.right = y
        else:
            node.parent.left = y

        y.right = node
        node.parent = y

    def _rb_insert_fixup(self, node):
        while node.parent._is_red():
            if node.parent == node.parent.parent.left:
                y = node.parent.parent.right

                if y._is_red():
                    node.parent._red = False
                    y._red = False
                    node.parent.parent._red = True
                    node = node.parent.parent
                else:
                    if node == node.parent.right:
                        node = node.parent
                        self._rotate_left(node)
                    node.parent._red = False
                    node.parent.parent._red = True
                    self._rotate_right(node.parent.parent)
            else:
                y = node.parent.parent.left

                if y._is_red():
                    node.parent._red = False
                    y._red = False
                    node.parent.parent._red = True
                    node = node.parent.parent
                else:
                    if node == node.parent.left:
                        node = node.parent
                        self._rotate_right(node)
                    node.parent._red = False
                    node.parent.parent._red = True
                    self._rotate_left(node.parent.parent)

        self.root._red = False

    def _transplant(self, u, v):
        if not u.parent:
            # u was the root.
            self.root = v
        elif u == u.parent.left:
            u.parent.left = v
        else:
            u.parent.right = v

        v.parent = u.parent

    def _rb_delete_fixup(self, node):
        while node != self.root and node._is_black():
            if node == node.parent.left:
                w = node.parent.right
                if w._is_red():
                    w._red = False
                    node.parent._red = True
                    self._rotate_left(node.parent)
                    w = node.parent.right
                if w.left._is_black() and w.right._is_black():
                    w._red = True
                    node = node.parent
                else:
                    if w.right._is_black():
                        w.left._red = False
                        w._red = True
                        self._rotate_right(w)
                        w = node.parent.right
                    w._red = node.parent._red
                    node.parent._red = False
                    w.right._red = False
                    self._rotate_left(node.parent)
                    node = self.root
            else:
                w = node.parent.left
                if w._is_red():
                    w._red = False
                    node.parent._red = True
                    self._rotate_right(node.parent)
                    w = node.parent.left
                if w.right._is_black() and w.left._is_black():
                    w._red = True
                    node = node.parent
                else:
                    if w.left._is_black():
                        w.right._red = False
                        w._red = True
                        self._rotate_left(w)
                        w = node.parent.left
                    w._red = node.parent._red
                    node.parent._red = False
                    w.left._red = False
                    self._rotate_right(node.parent)
                    node = self.root

        node._red = False

    def _extra_info_insert_traversed_node(self, _node, _new_value, _direction: int):
        """Called when we are about to branch from a node while finding insertion point.

        new_value is the value we are trying to insert that does not yet exist as a node.
        Direction is negative if we will take left branch from node, positive for right.
        """
        return

    def _extra_info_insert_created_node(self, _node):
        """Called after we have placed a new node in the tree."""
        return

    def _extra_info_delete_removed_node(self, _node):
        """Called when a node is about to be removed from the tree.

        The node may be the node that is about to be deleted from the tree entirely,
        but it may also be a node that is about to be moved to a higher position
        and placed back into the the tree later. The function will only be called on
        a moved node if that node will be severed from one of its non-NIL children.

        All nodes in the subtrees of the children will also be losing
        an ancestor, which might need to be taken into account.

        Consider if modifications need to be made further up the tree as well.
        """
        return

    def _extra_info_delete_moved_node(self, _node, _parent, _left_child, _right_child):
        """Called when a node is about to be moved to a new location in the tree.

        parent will be the new parent. The parent always changes when a node is moved.
        left_child and right_child might be None if those value are not changing when the node
        is moved (meaning they can be directly accessed from the node if needed). If they
        are not None then they are the new children for the new location of the node in the tree.
        """
        return

    def _extra_info_balance_left_rotate(self, _node):
        """Called when a left rotate is about to be applied to the node."""
        return

    def _extra_info_balance_right_rotate(self, _node):
        """Called when a right rotate is about to be applied to the node."""
        return

    def __repr__(self) -> str:
        return f"{self.root!r}"
