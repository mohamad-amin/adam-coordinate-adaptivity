import jax


def get_or_make(d, key):
    if key in d:
        return d[key]
    else:
        d[key] = dict()
        return d[key]


def tree_path_to_name(path):
    return '.'.join(map(lambda x: x.key, path))


def random_split_like_tree(rng_key, target, treedef=None):
    if treedef is None:
        treedef = jax.tree_structure(target)
    keys = jax.random.split(rng_key, treedef.num_leaves)
    return jax.tree_unflatten(treedef, keys)


def count_params(params_pytree):
    flat_params = jax.tree_util.tree_leaves(params_pytree)
    return sum(x.size for x in flat_params)
