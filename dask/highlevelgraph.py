from collections.abc import Mapping
from typing import Hashable, List, Set, Union

import tlz as toolz

from .utils import ignoring
from .base import is_dask_collection
from .core import reverse_dict, get_dependencies


class HlgKeys:
    """ The base class for high level graph keys

    Parameters
    ----------
    value: set
        The set of keys.
    name: str
        Common name of ALL the keys in `value` or None if `value` doesn't
        share a common key name.
    """

    def __init__(self, value=set(), name=None):
        self._value = value
        self._name = name

    def name(self):
        """ Return the common name of ALL keys in self or None if they
        doesn't share name."""
        return self._name

    def named(self):
        """ Whether the keys share a commen name or not """
        return self._name is not None

    def value(self) -> Set:
        """ Return a regular set of keys

        Notice, this function will trigger a materialization of the set
        """
        return self._value

    def _may_contain(self, key: Hashable):
        """ Fast track check if self contains `key` """

        if self.named():
            if not (
                isinstance(key, tuple)
                and len(key) > 0
                and isinstance(key[0], str)
                and key[0] == self.name()
            ):
                return False
        else:
            return True

    def _may_intersect(self, other: "HlgKeys"):
        """ Fast track check if self intersect `key` """
        return not (
            (self.named() and self.name() != other.name())
            or len(self) == 0
            or len(other) == 0
        )

    def __len__(self):
        return len(self.value())

    def __contains__(self, key: Hashable):
        return self._may_contain(key) and key in self.value()

    def __and__(self, other):
        """Return a new set with keys common in self and other"""
        if isinstance(other, HlgKeys):
            if self._may_intersect(other):
                return other & self.value()
            else:
                return HlgKeys()
        else:
            return HlgKeys(name=None, value=other & self.value())

    def __or__(self, other):
        """Return a new set with keys from self and other"""
        if len(other) == 0:
            return self
        if isinstance(other, HlgKeys):
            if len(self) == 0:
                return other
            else:
                return other | self.value()
        else:
            return HlgKeys(name=None, value=other | self.value())

    def __repr__(self):
        return (
            f"<{type(self).__name__} name={repr(self._name)} "
            f"keys={repr(self.value())}>"
        )


class HlgKeysList(HlgKeys):
    """ List of sets that act as a `HlgKeys`

    Parameters
    ----------
    list_of_sets: list of sets or HlgKeys
        The sets and/or HlgKeys that makes up this `HlgKeysList`.
    name: str
        Common name of ALL the keys in `list_of_sets` or None if
        `list_of_sets` doesn't share a common key name.
    """

    def __init__(self, list_of_keys: List[Union[HlgKeys, Set]] = [], name=None):
        self._list_of_keys = list_of_keys
        super().__init__(name=name)

    def value(self) -> Set:
        ret = set()
        for keys in self._list_of_keys:
            if isinstance(keys, HlgKeys):
                ret |= keys.value()
            else:
                ret |= keys
        return ret

    def __contains__(self, key: Hashable):
        if self._may_contain(key):
            for keys in self._list_of_keys:
                if key in keys:
                    return True
        return False

    def __len__(self):
        return sum([len(k) for k in self._list_of_keys], 0)

    def __and__(self, other):
        if isinstance(other, HlgKeys):
            if not self._may_intersect(other):
                return HlgKeysList()
            name = self.name() if self.name() == other.name() else None
            ret = []
            for keys in self._list_of_keys:
                ret.append(other & keys)
            return HlgKeysList(list_of_keys=ret, name=name)
        else:
            ret = []
            for keys in self._list_of_keys:
                ret.append(keys & other)
            return HlgKeysList(list_of_keys=ret)

    def __or__(self, other):
        if len(other) == 0:
            return self
        name = None
        if isinstance(other, HlgKeys):
            if len(self) == 0:
                return other
            name = self.name() if self.name() == other.name() else None
            if isinstance(other, HlgKeysList):
                return HlgKeysList(self._list_of_sets + other._list_of_sets, name=name)

        return HlgKeysList(self._list_of_sets + [other], name=name)

    def __repr__(self):
        return (
            f"<{type(self).__name__} name={repr(self._name)} "
            f"sets={repr(self._list_of_keys)}>"
        )


class Layer(Mapping):
    """ High level graph layer

    This abstract class establish a protocol for high level graph layers.
    """

    def __contains__(self, key):
        return key in self.get_key_set()

    def get_key_set(self) -> HlgKeys:
        """ Return the keys of the layer as a set """
        return HlgKeys(set(self.keys()))

    def cull(self, keys: HlgKeys) -> "Layer":
        """ Return a new Layer with only the tasks required to calculate `keys`.

        In other words, remove unnecessary tasks from the layer.

        Examples
        --------
        >>> d = Layer({'x': 1, 'y': (inc, 'x'), 'out': (add, 'x', 10)})
        >>> dsk = d.cull(HlgKeys(set('out')))
        >>> dsk  # doctest: +SKIP
        {'x': 1, 'out': (add, 'x', 10)}

        Returns
        -------
        layer: Layer
            Culled layer
        """
        seen = set()
        out = {}
        work = keys.value()

        while len(work) > 0:
            new_work = []
            for k in work:
                out[k] = self[k]
                for d in get_dependencies(self, k, as_list=True):
                    if d not in seen:
                        seen.add(d)
                        new_work.append(d)
            work = new_work
        return BasicLayer(out)

    def get_external_dependencies(self, known_keys: HlgKeys):
        """Get external dependencies

        Parameters
        ----------
        known_keys : HlgKeys
            Set of known keys (typically all keys in a HighLevelGraph)

        Returns
        -------
        deps: HlgKeys
            Set of dependencies
        """
        ret = set()
        work = list(self.values())

        while work:
            new_work = []
            for w in work:
                typ = type(w)
                if typ is tuple and w and callable(w[0]):  # istask(w)
                    new_work.extend(w[1:])
                elif typ is list:
                    new_work.extend(w)
                elif typ is dict:
                    new_work.extend(w.values())
                else:
                    try:
                        if w in known_keys and w not in self.keys():
                            ret.add(w)
                    except TypeError:  # not hashable
                        pass
            work = new_work
        return HlgKeys(ret)


class BasicLayer(Layer):
    """ Basic implementation of `Layer` that takes a mapping and a name """

    def __init__(self, mapping, name=None):
        self.__mapping = mapping
        self.__name = name

    def __getitem__(self, k):
        return self.__mapping[k]

    def __iter__(self):
        return iter(self.__mapping)

    def __len__(self):
        return len(self.__mapping)

    def __repr__(self):
        return (
            f"<{type(self).__name__} name={repr(self.__name)} "
            f"mapping={repr(self.__mapping)}>"
        )

    def get_key_set(self) -> HlgKeys:
        return HlgKeys(set(self.keys()), name=self.__name)


class HighLevelGraph(Mapping):
    """ Task graph composed of layers of dependent subgraphs

    This object encodes a Dask task graph that is composed of layers of
    dependent subgraphs, such as commonly occurs when building task graphs
    using high level collections like Dask array, bag, or dataframe.

    Typically each high level array, bag, or dataframe operation takes the task
    graphs of the input collections, merges them, and then adds one or more new
    layers of tasks for the new operation.  These layers typically have at
    least as many tasks as there are partitions or chunks in the collection.
    The HighLevelGraph object stores the subgraphs for each operation
    separately in sub-graphs, and also stores the dependency structure between
    them.

    Parameters
    ----------
    layers : Dict[str, Mapping]
        The subgraph layers, keyed by a unique name
    dependencies : Dict[str, Set[str]]
        The set of layers on which each layer depends

    Examples
    --------

    Here is an idealized example that shows the internal state of a
    HighLevelGraph

    >>> import dask.dataframe as dd

    >>> df = dd.read_csv('myfile.*.csv')  # doctest: +SKIP
    >>> df = df + 100  # doctest: +SKIP
    >>> df = df[df.name == 'Alice']  # doctest: +SKIP

    >>> graph = df.__dask_graph__()  # doctest: +SKIP
    >>> graph.layers  # doctest: +SKIP
    {
     'read-csv': {('read-csv', 0): (pandas.read_csv, 'myfile.0.csv'),
                  ('read-csv', 1): (pandas.read_csv, 'myfile.1.csv'),
                  ('read-csv', 2): (pandas.read_csv, 'myfile.2.csv'),
                  ('read-csv', 3): (pandas.read_csv, 'myfile.3.csv')},
     'add': {('add', 0): (operator.add, ('read-csv', 0), 100),
             ('add', 1): (operator.add, ('read-csv', 1), 100),
             ('add', 2): (operator.add, ('read-csv', 2), 100),
             ('add', 3): (operator.add, ('read-csv', 3), 100)}
     'filter': {('filter', 0): (lambda part: part[part.name == 'Alice'], ('add', 0)),
                ('filter', 1): (lambda part: part[part.name == 'Alice'], ('add', 1)),
                ('filter', 2): (lambda part: part[part.name == 'Alice'], ('add', 2)),
                ('filter', 3): (lambda part: part[part.name == 'Alice'], ('add', 3))}
    }

    >>> graph.dependencies  # doctest: +SKIP
    {
     'read-csv': set(),
     'add': {'read-csv'},
     'filter': {'add'}
    }

    See Also
    --------
    HighLevelGraph.from_collections :
        typically used by developers to make new HighLevelGraphs
    """

    def __init__(self, layers, dependencies):
        self.layers = layers
        self.dependencies = dependencies

    @property
    def dependents(self):
        return reverse_dict(self.dependencies)

    @property
    def dicts(self):
        # Backwards compatibility for now
        return self.layers

    @classmethod
    def _from_collection(cls, name, layer, collection):
        """ `from_collections` optimized for a single collection """
        if is_dask_collection(collection):
            graph = collection.__dask_graph__()
            if isinstance(graph, HighLevelGraph):
                layers = graph.layers.copy()
                layers.update({name: layer})
                deps = graph.dependencies.copy()
                with ignoring(AttributeError):
                    deps.update({name: set(collection.__dask_layers__())})
            else:
                try:
                    [key] = collection.__dask_layers__()
                except AttributeError:
                    key = id(graph)
                layers = {name: layer, key: graph}
                deps = {name: {key}, key: set()}
        else:
            raise TypeError(type(collection))

        return cls(layers, deps)

    @classmethod
    def from_collections(cls, name, layer, dependencies=()):
        """ Construct a HighLevelGraph from a new layer and a set of collections

        This constructs a HighLevelGraph in the common case where we have a single
        new layer and a set of old collections on which we want to depend.

        This pulls out the ``__dask_layers__()`` method of the collections if
        they exist, and adds them to the dependencies for this new layer.  It
        also merges all of the layers from all of the dependent collections
        together into the new layers for this graph.

        Parameters
        ----------
        name : str
            The name of the new layer
        layer : Mapping
            The graph layer itself
        dependencies : List of Dask collections
            A lit of other dask collections (like arrays or dataframes) that
            have graphs themselves

        Examples
        --------

        In typical usage we make a new task layer, and then pass that layer
        along with all dependent collections to this method.

        >>> def add(self, other):
        ...     name = 'add-' + tokenize(self, other)
        ...     layer = {(name, i): (add, input_key, other)
        ...              for i, input_key in enumerate(self.__dask_keys__())}
        ...     graph = HighLevelGraph.from_collections(name, layer, dependencies=[self])
        ...     return new_collection(name, graph)
        """
        if len(dependencies) == 1:
            return cls._from_collection(name, layer, dependencies[0])
        layers = {name: layer}
        deps = {}
        deps[name] = set()
        for collection in toolz.unique(dependencies, key=id):
            if is_dask_collection(collection):
                graph = collection.__dask_graph__()
                if isinstance(graph, HighLevelGraph):
                    layers.update(graph.layers)
                    deps.update(graph.dependencies)
                    with ignoring(AttributeError):
                        deps[name] |= set(collection.__dask_layers__())
                else:
                    try:
                        [key] = collection.__dask_layers__()
                    except AttributeError:
                        key = id(graph)
                    layers[key] = graph
                    deps[name].add(key)
                    deps[key] = set()
            else:
                raise TypeError(type(collection))

        return cls(layers, deps)

    def __getitem__(self, key):
        for d in self.layers.values():
            if key in d:
                return d[key]
        raise KeyError(key)

    def __len__(self):
        return sum(1 for _ in self)

    def items(self):
        items = []
        seen = set()
        for d in self.layers.values():
            for key in d:
                if key not in seen:
                    seen.add(key)
                    items.append((key, d[key]))
        return items

    def __iter__(self):
        return toolz.unique(toolz.concat(self.layers.values()))

    def keys(self):
        return [key for key, _ in self.items()]

    def values(self):
        return [value for _, value in self.items()]

    @classmethod
    def merge(cls, *graphs):
        layers = {}
        dependencies = {}
        for g in graphs:
            if isinstance(g, HighLevelGraph):
                layers.update(g.layers)
                dependencies.update(g.dependencies)
            elif isinstance(g, Mapping):
                layers[id(g)] = g
                dependencies[id(g)] = set()
            else:
                raise TypeError(g)
        return cls(layers, dependencies)

    def visualize(self, filename="dask.pdf", format=None, **kwargs):
        from .dot import graphviz_to_file

        g = to_graphviz(self, **kwargs)
        return graphviz_to_file(g, filename, format)

    def validate(self):
        # Check dependencies
        for layer_name, deps in self.dependencies.items():
            if layer_name not in self.layers:
                raise ValueError(
                    f"dependencies[{repr(layer_name)}] not found in layers"
                )
            for dep in deps:
                if dep not in self.dependencies:
                    raise ValueError(f"{repr(dep)} not found in dependencies")


def to_graphviz(
    hg,
    data_attributes=None,
    function_attributes=None,
    rankdir="BT",
    graph_attr={},
    node_attr=None,
    edge_attr=None,
    **kwargs,
):
    from .dot import graphviz, name, label

    if data_attributes is None:
        data_attributes = {}
    if function_attributes is None:
        function_attributes = {}

    graph_attr = graph_attr or {}
    graph_attr["rankdir"] = rankdir
    graph_attr.update(kwargs)
    g = graphviz.Digraph(
        graph_attr=graph_attr, node_attr=node_attr, edge_attr=edge_attr
    )

    cache = {}

    for k in hg.dependencies:
        k_name = name(k)
        attrs = data_attributes.get(k, {})
        attrs.setdefault("label", label(k, cache=cache))
        attrs.setdefault("shape", "box")
        g.node(k_name, **attrs)

    for k, deps in hg.dependencies.items():
        k_name = name(k)
        for dep in deps:
            dep_name = name(dep)
            g.edge(dep_name, k_name)
    return g
