import aerospike
import pytest
from langgraph.store.aerospike.base import AerospikeStore
from langgraph.store.base import GetOp, ListNamespacesOp, MatchCondition, PutOp, SearchOp

# Configuration for local Docker instance
AEROSPIKE_CONFIG = {"hosts": [("localhost", 3000)]}
TEST_NAMESPACE = "test"  # Aerospike default namespace is often 'test'
TEST_SET = "langgraph_store"


@pytest.fixture(scope="session")
def aerospike_client():
    """
    Creates a single connection for the whole test session.
    """
    client = aerospike.client(AEROSPIKE_CONFIG).connect()
    yield client
    client.close()


@pytest.fixture
def store(aerospike_client):
    """
    Creates the store instance and cleans up the set before each test.
    """
    store = AerospikeStore(client=aerospike_client, namespace=TEST_NAMESPACE, set=TEST_SET)

    # --- Cleanup / Truncate Set before test starts ---
    # Note: truncate is asynchronous, so we wait briefly or just use scan/remove
    # for strictly clean state if truncate is slow. For unit tests, scan+remove is safer.
    try:
        scan = aerospike_client.scan(TEST_NAMESPACE, TEST_SET)

        def callback(input_tuple):
            key, _, _ = input_tuple
            aerospike_client.remove(key)

        scan.foreach(callback)
    except Exception:
        pass  # Ignore if set is empty

    return store


def test_basic_put_and_get(store):
    """Test simple write and read operations."""
    namespace = ("users", "profiles")
    key = "user_123"
    data = {"name": "Alice", "age": 30}

    # 1. Put Data
    op = PutOp(namespace=namespace, key=key, value=data)
    store.batch([op])

    # 2. Get Data
    item = store.get(namespace, key)

    assert item is not None
    assert item.value == data
    assert item.key == key
    assert item.namespace == namespace
    assert item.created_at is not None
    assert item.updated_at is not None


def test_get_missing_item(store):
    """Test getting a key that doesn't exist."""
    item = store.get(("ghost", "town"), "non_existent")
    assert item is None


def test_batch_operations(store):
    """Test mixing Put and Get in a batch."""
    ns = ("memories",)

    ops = [
        PutOp(namespace=ns, key="k1", value={"v": 1}),
        PutOp(namespace=ns, key="k2", value={"v": 2}),
    ]
    store.batch(ops)

    # Now retrieve them in a batch
    read_ops = [GetOp(namespace=ns, key="k1"), GetOp(namespace=ns, key="k2")]
    results = store.batch(read_ops)

    assert len(results) == 2
    assert results[0].value == {"v": 1}
    assert results[1].value == {"v": 2}


def test_delete_item(store):
    """Test that putting None deletes the item."""
    ns = ("temp",)
    key = "to_delete"

    # Create
    store.batch([PutOp(namespace=ns, key=key, value={"data": "here"})])
    assert store.get(ns, key) is not None

    # Delete (Put None)
    store.batch([PutOp(namespace=ns, key=key, value=None)])

    # Verify
    assert store.get(ns, key) is None


def test_search_exact_match(store):
    """Test searching with filter expressions."""
    ns = ("documents", "reports")

    # Setup data
    ops = [
        PutOp(namespace=ns, key="doc1", value={"status": "draft", "author": "bob"}),
        PutOp(namespace=ns, key="doc2", value={"status": "published", "author": "alice"}),
        PutOp(namespace=ns, key="doc3", value={"status": "draft", "author": "charlie"}),
    ]
    store.batch(ops)

    # Search for status=draft
    search_op = SearchOp(namespace_prefix=ns, filter={"status": {"$eq": "draft"}}, limit=10)
    results = store.batch([search_op])[0]

    assert len(results) == 2
    authors = sorted([r.value["author"] for r in results])
    assert authors == ["bob", "charlie"]


def test_list_namespaces(store):
    """Test listing namespaces with prefixes."""
    # Setup hierarchy
    # ("root", "branch_a", "leaf_1")
    # ("root", "branch_a", "leaf_2")
    # ("root", "branch_b", "leaf_3")

    data = {"dummy": True}
    ops = [
        PutOp(namespace=("root", "branch_a", "leaf_1"), key="k", value=data),
        PutOp(namespace=("root", "branch_a", "leaf_2"), key="k", value=data),
        PutOp(namespace=("root", "branch_b", "leaf_3"), key="k", value=data),
    ]
    store.batch(ops)

    # List with prefix ("root", "branch_a")
    list_op = ListNamespacesOp(
        match_conditions=[MatchCondition(match_type="prefix", path=("root", "branch_a"))],
        max_depth=3,
    )

    # Depending on how the user code parses MatchConditions, we might need to verify inputs
    # But based on your implementation:
    results = store.batch([list_op])[0]

    # Should find 2 namespaces: leaf_1 and leaf_2
    assert len(results) == 2
    assert ("root", "branch_a", "leaf_1") in results
    assert ("root", "branch_a", "leaf_2") in results


def test_search_numeric_operators(store):
    """Test $gt, $lt, $gte, $lte with integers."""
    ns = ("game", "scores")

    # Setup data
    ops = [
        PutOp(namespace=ns, key="p1", value={"score": 10, "rank": "C"}),
        PutOp(namespace=ns, key="p2", value={"score": 20, "rank": "B"}),
        PutOp(namespace=ns, key="p3", value={"score": 30, "rank": "A"}),
        PutOp(namespace=ns, key="p4", value={"score": 40, "rank": "S"}),
    ]
    store.batch(ops)

    # 1. Test Greater Than ($gt)
    # Expect scores > 20 -> 30, 40
    res_gt = store.batch([SearchOp(namespace_prefix=ns, filter={"score": {"$gt": 20}})])[0]
    assert len(res_gt) == 2
    assert {item.value["score"] for item in res_gt} == {30, 40}

    # 2. Test Less Than or Equal ($lte)
    # Expect scores <= 20 -> 10, 20
    res_lte = store.batch([SearchOp(namespace_prefix=ns, filter={"score": {"$lte": 20}})])[0]
    assert len(res_lte) == 2
    assert {item.value["score"] for item in res_lte} == {10, 20}


def test_search_float_comparisons(store):
    """Test comparisons with floating point numbers to ensure type inference works."""
    ns = ("sensors", "temp")

    ops = [
        PutOp(namespace=ns, key="t1", value={"temperature": 98.6}),
        PutOp(namespace=ns, key="t2", value={"temperature": 100.5}),
        PutOp(namespace=ns, key="t3", value={"temperature": 102.1}),
    ]
    store.batch(ops)

    # Test Greater Than with Float
    # Note: It is crucial that the input filter value (100.0) is a float
    # so the Store infers exp.ResultType.FLOAT
    res_float = store.batch(
        [SearchOp(namespace_prefix=ns, filter={"temperature": {"$gt": 100.0}})]
    )[0]

    assert len(res_float) == 2
    assert {item.key for item in res_float} == {"t2", "t3"}


def test_search_not_equal(store):
    """Test the $ne operator."""
    ns = ("catalog", "fruits")

    ops = [
        PutOp(namespace=ns, key="f1", value={"color": "red", "type": "apple"}),
        PutOp(namespace=ns, key="f2", value={"color": "yellow", "type": "banana"}),
        PutOp(namespace=ns, key="f3", value={"color": "red", "type": "strawberry"}),
    ]
    store.batch(ops)

    # Search for anything that is NOT red
    res_ne = store.batch([SearchOp(namespace_prefix=ns, filter={"color": {"$ne": "red"}})])[0]

    assert len(res_ne) == 1
    assert res_ne[0].value["type"] == "banana"


def test_search_multiple_conditions(store):
    """Test combining multiple keys (Implicit AND)."""
    ns = ("hr", "employees")

    ops = [
        # Match matches department but not active
        PutOp(namespace=ns, key="e1", value={"dept": "engineering", "active": False, "years": 5}),
        # Match matches active but not department
        PutOp(namespace=ns, key="e2", value={"dept": "sales", "active": True, "years": 5}),
        # Matches Both
        PutOp(namespace=ns, key="e3", value={"dept": "engineering", "active": True, "years": 2}),
        # Matches Both Dept/Active but fails Years condition
        PutOp(namespace=ns, key="e4", value={"dept": "engineering", "active": True, "years": 10}),
    ]
    store.batch(ops)

    # Filter: Engineering AND Active AND Years < 5
    search_filter = {"dept": "engineering", "active": True, "years": {"$lt": 5}}

    results = store.batch([SearchOp(namespace_prefix=ns, filter=search_filter)])[0]

    assert len(results) == 1
    assert results[0].key == "e3"


def test_search_mixed_operators_on_same_field(store):
    """Test range queries (e.g., 10 < price < 20) if supported by the implementation structure."""
    # Note: Your implementation iterates over keys. If a user provides:
    # {"price": {"$gt": 10, "$lt": 20}}
    # Your loop `for op, val in condition.items()` handles both $gt and $lt for the same key.

    ns = ("shop", "items")
    ops = [
        PutOp(namespace=ns, key="i1", value={"price": 5}),
        PutOp(namespace=ns, key="i2", value={"price": 15}),
        PutOp(namespace=ns, key="i3", value={"price": 25}),
    ]
    store.batch(ops)

    # Range query: 10 < price < 20
    results = store.batch(
        [SearchOp(namespace_prefix=ns, filter={"price": {"$gt": 10, "$lt": 20}})]
    )[0]

    assert len(results) == 1
    assert results[0].value["price"] == 15
