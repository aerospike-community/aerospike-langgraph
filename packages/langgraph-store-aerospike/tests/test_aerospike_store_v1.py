# tests/test_store_basic.py
from langgraph.store.base import GetOp, ListNamespacesOp, MatchCondition, PutOp, SearchOp


def cleanup(store):
    """Delete all records in this namespace/set."""
    scan = store.client.scan(store.ns, store.set)
    for key, _meta, _bins in scan.results():
        store.client.remove(key)


def test_write_and_get(store):
    cleanup(store)
    ns = ("demo", "user1")
    key = "k1"
    value = {"foo": "bar"}
    store.batch([PutOp(namespace=ns, key=key, value=value)])
    item = store.get(ns, key)
    assert item is not None
    assert item.namespace == ns
    assert item.key == key
    assert item.value == value


def test_list_namespaces(store):
    cleanup(store)
    ns = ("demo", "user2")
    store.batch([PutOp(namespace=ns, key="x", value={"a": 1})])
    out = store.list_namespaces()
    assert ns in out
    prefixed = store.list_namespaces(prefix=("demo",))
    assert ns in prefixed


def test_search(store):
    cleanup(store)
    ns = ("demo", "user3")
    store.batch([PutOp(namespace=ns, key="x", value={"type": "chat", "user": "u3"})])
    store.batch([PutOp(namespace=ns, key="y", value={"type": "chat", "user": "u3"})])
    res = store.batch([SearchOp(namespace_prefix=("demo", "user3"))])
    print(len(res[0]))
    assert len(res[0]) == 2
    filtered = store.batch(
        [
            SearchOp(
                namespace_prefix=("demo",),
                filter={"type": "chat"},
            )
        ]
    )
    assert len(filtered[0]) == 2


def test_batch(store):
    cleanup(store)
    ns = ("demo", "user4")
    ops = [
        PutOp(namespace=ns, key="a", value={"v": 1}),
        PutOp(namespace=ns, key="b", value={"v": 2}),
        GetOp(namespace=ns, key="a"),
        SearchOp(namespace_prefix=("demo",)),
        ListNamespacesOp(
            match_conditions=[MatchCondition(match_type="prefix", path=("demo",))],
            limit=10,
            offset=0,
            max_depth=None,
        ),
    ]
    results = store.batch(ops)
    # Check PutOps return None
    assert results[0] is None
    assert results[1] is None
    # Check GetOp result
    got = results[2]
    assert got.key == "a"
    assert got.value == {"v": 1}
    # Search result
    search_items = results[3]
    assert len(search_items) >= 2
    # Namespace listing
    ns_list = results[4]
    assert ns in ns_list
