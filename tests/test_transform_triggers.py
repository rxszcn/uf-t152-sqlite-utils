"""
Tests that ``table.transform()`` preserves the triggers sqlite-utils creates
for full-text search and the cached counts table.

The rebuild performed by ``transform()`` drops and recreates the underlying
table, which in SQLite also removes every trigger attached to it. These tests
pin the behaviour on three sides for both the plain rebuild and the
``keep_table=`` variant:

1. the trigger inventory before and after the rebuild (names and SQL text),
2. full-text search continuing to index inserted, updated and deleted rows,
3. the cached counts staying in step with the real row count.
"""

import pytest

from sqlite_utils import Database
from sqlite_utils.db import TransformError


@pytest.fixture
def fts_db(fts_version):
    db = Database(memory=True)
    db["articles"].insert_all(
        [
            {"id": 1, "title": "alpha dog", "weight": "10"},
            {"id": 2, "title": "beta cat", "weight": "20"},
        ],
        pk="id",
    )
    db["articles"].enable_fts(
        ["title"], fts_version=fts_version, create_triggers=True
    )
    db["articles"].enable_counts()
    return db


@pytest.fixture
def fts5_db():
    db = Database(memory=True)
    db["articles"].insert_all(
        [
            {"id": 1, "title": "alpha dog", "weight": "10"},
            {"id": 2, "title": "beta cat", "weight": "20"},
        ],
        pk="id",
    )
    db["articles"].enable_fts(["title"], create_triggers=True)
    db["articles"].enable_counts()
    return db


FTS_TRIGGER_NAMES = {"articles_ai", "articles_ad", "articles_au"}
COUNTS_TRIGGER_NAMES = {
    "articles_counts_insert",
    "articles_counts_delete",
}
ALL_TRIGGER_NAMES = FTS_TRIGGER_NAMES | COUNTS_TRIGGER_NAMES


# ---------------------------------------------------------------------------
# Side 1: trigger inventory before vs after the rebuild
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("fts_version", ["FTS5", "FTS4"])
def test_transform_preserves_trigger_inventory_and_sql(fts_db):
    table = fts_db["articles"]
    triggers_before = table.triggers_dict
    assert set(triggers_before) == ALL_TRIGGER_NAMES

    table.transform(types={"weight": int})

    # Same trigger names, attached to the replacement table.
    assert set(table.triggers_dict) == ALL_TRIGGER_NAMES
    assert {trigger.table for trigger in table.triggers} == {"articles"}
    # The regenerated managed triggers are byte-for-byte identical.
    assert table.triggers_dict == triggers_before


@pytest.mark.parametrize("fts_version", ["FTS5", "FTS4"])
def test_transform_keep_table_splits_triggers_between_old_and_new(fts_db):
    table = fts_db["articles"]
    old_trigger_sql = dict(table.triggers_dict)

    table.transform(types={"weight": int}, keep_table="articles_backup")

    backup = fts_db["articles_backup"]
    # The original triggers stay attached to the backup table under their
    # original names.
    assert set(backup.triggers_dict) == ALL_TRIGGER_NAMES
    assert {trigger.table for trigger in backup.triggers} == {
        "articles_backup"
    }
    # The replacement table gets a fresh, distinctly named set.
    new_triggers = table.triggers_dict
    assert len(new_triggers) == len(ALL_TRIGGER_NAMES)
    assert not (set(new_triggers) & ALL_TRIGGER_NAMES)
    assert {trigger.table for trigger in table.triggers} == {"articles"}
    # Bodies are semantically identical apart from the table/trigger names.
    for name in old_trigger_sql:
        matching = [
            new_sql
            for new_name, new_sql in new_triggers.items()
            if new_name.startswith(name)
        ]
        assert len(matching) == 1
        assert 'ON "articles"' in matching[0]


def test_transform_without_triggers_creates_none(fresh_db):
    fresh_db["plain"].insert({"id": 1, "name": "Cleo"}, pk="id")
    fresh_db["plain"].transform(types={"id": str})
    assert fresh_db["plain"].triggers == []


def test_transform_keeps_fts_virtual_table_when_no_sync_triggers(fresh_db):
    fresh_db["t"].insert({"id": 1, "a": "searchable text"}, pk="id")
    fresh_db["t"].enable_fts(["a"], create_triggers=False)
    fresh_db["t"].transform(types={"id": int})
    assert fresh_db["t"].detect_fts() == "t_fts"
    assert fresh_db["t"].triggers == []


def test_transform_preserves_custom_trigger_with_rewritten_columns(fresh_db):
    fresh_db.execute("CREATE TABLE log (id INTEGER PRIMARY KEY, msg TEXT)")
    fresh_db.execute("CREATE TABLE log_audit (note TEXT)")
    fresh_db.execute(
        """
        CREATE TRIGGER log_tr AFTER INSERT ON log BEGIN
            INSERT INTO log_audit (note) VALUES ('logged ' || new.msg);
        END
        """
    )
    fresh_db["log"].transform(rename={"msg": "message"})

    assert [trigger.name for trigger in fresh_db["log"].triggers] == [
        "log_tr"
    ]
    trigger_sql = fresh_db["log"].triggers_dict["log_tr"]
    assert "new.message" in trigger_sql
    assert "new.msg" not in trigger_sql
    # The string literal inside the trigger must be untouched.
    assert "'logged '" in trigger_sql
    fresh_db.execute(
        "INSERT INTO log (id, message) VALUES (1, 'hello')"
    )
    assert fresh_db.execute("SELECT note FROM log_audit").fetchall() == [
        ("logged hello",)
    ]


def test_transform_custom_trigger_keeps_other_tables_columns(fresh_db):
    # The trigger table and another table both have a column "a". Only the
    # new./old. references to the trigger table's "a" may be renamed; the
    # other table's column must be left exactly as written.
    fresh_db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, a TEXT)")
    fresh_db.execute("CREATE TABLE other (a TEXT)")
    fresh_db.execute(
        """
        CREATE TRIGGER t_tr AFTER UPDATE OF a ON t
        WHEN old.a <> new.a
        BEGIN
            INSERT INTO other (a) VALUES (new.a);
        END
        """
    )
    fresh_db["t"].transform(rename={"a": "alpha"})

    trigger_sql = fresh_db["t"].triggers_dict["t_tr"]
    assert "UPDATE OF alpha ON t" in trigger_sql
    assert "old.alpha" in trigger_sql
    assert "new.alpha" in trigger_sql
    assert "INSERT INTO other (a)" in trigger_sql
    fresh_db.execute("INSERT INTO t (id, alpha) VALUES (1, 'first')")
    fresh_db.execute("UPDATE t SET alpha = 'second' WHERE id = 1")
    assert fresh_db.execute("SELECT a FROM other").fetchall() == [
        ("second",)
    ]


def test_transform_refuses_to_drop_fts_indexed_column(fts5_db):
    with pytest.raises(TransformError, match="full-text search index"):
        fts5_db["articles"].transform(drop=["title"])
    # Nothing was applied.
    assert "title" in fts5_db["articles"].columns_dict
    assert set(fts5_db["articles"].triggers_dict) == ALL_TRIGGER_NAMES


def test_transform_keeps_custom_trigger_using_managed_name(fresh_db):
    # A custom trigger named like one FTS trigger, but without the full FTS
    # set, must be treated as a custom trigger rather than silently dropped.
    fresh_db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, a TEXT)")
    fresh_db.execute(
        'CREATE TRIGGER t_ai AFTER INSERT ON t BEGIN SELECT 1; END'
    )
    fresh_db["t"].transform(types={"id": int})
    assert [trigger.name for trigger in fresh_db["t"].triggers] == ["t_ai"]


def test_transform_keep_table_custom_trigger_using_managed_name(fresh_db):
    fresh_db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, a TEXT)")
    fresh_db.execute(
        'CREATE TRIGGER t_ai AFTER INSERT ON t BEGIN SELECT 1; END'
    )
    fresh_db["t"].transform(types={"id": int}, keep_table="t_old")
    # The backup keeps the original name; the new table gets a suffixed one.
    assert [t.name for t in fresh_db["t_old"].triggers] == ["t_ai"]
    new_triggers = [t.name for t in fresh_db["t"].triggers]
    assert len(new_triggers) == 1
    assert new_triggers[0].startswith("t_ai_")


@pytest.mark.parametrize("fts_version", ["FTS5", "FTS4"])
def test_transform_fts_column_rename_for_both_versions(
    fresh_db, fts_version
):
    fresh_db["t"].insert_all(
        [{"id": 1, "a": "alpha dog"}, {"id": 2, "a": "beta cat"}],
        pk="id",
    )
    fresh_db["t"].enable_fts(
        ["a"], fts_version=fts_version, create_triggers=True
    )
    fresh_db["t"].transform(rename={"a": "b"})

    # The FTS table keeps its own column name "a", now fed from content
    # column "b" via the rebuilt triggers.
    trigger_sql = fresh_db["t"].triggers_dict["t_ai"]
    assert '(rowid, "a")' in trigger_sql
    assert 'new."b"' in trigger_sql
    fresh_db["t"].insert({"id": 3, "b": "gamma bird"})
    assert [row["rowid"] for row in fresh_db["t"].search("gamma")] == [3]
    assert [row["rowid"] for row in fresh_db["t"].search("alpha")] == [1]


# ---------------------------------------------------------------------------
# Side 2: full-text search stays in sync
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("fts_version", ["FTS5", "FTS4"])
def test_transform_fts_indexes_newly_inserted_rows(fts_db):
    fts_db["articles"].transform(types={"weight": int})
    fts_db["articles"].insert(
        {"id": 3, "title": "gamma bird", "weight": 30}
    )
    assert [row["rowid"] for row in fts_db["articles"].search("gamma")] == [3]
    # Pre-existing rows remain searchable.
    assert [
        row["rowid"] for row in fts_db["articles"].search("alpha")
    ] == [1]


def test_transform_fts_handles_column_rename(fts5_db):
    fts5_db["articles"].transform(rename={"title": "headline"})
    fts5_db["articles"].insert(
        {"id": 3, "headline": "gamma bird", "weight": "30"}
    )
    assert [
        row["rowid"] for row in fts5_db["articles"].search("gamma")
    ] == [3]
    assert [
        row["rowid"] for row in fts5_db["articles"].search("alpha")
    ] == [1]
    # Updates propagate through the rebuilt UPDATE trigger.
    fts5_db["articles"].update(3, {"headline": "gamma eagle"})
    assert [
        row["rowid"] for row in fts5_db["articles"].search("eagle")
    ] == [3]
    assert list(fts5_db["articles"].search("bird")) == []


def test_transform_fts_rename_keys_resolve_case_insensitively(fresh_db):
    fresh_db["t"].insert_all(
        [{"id": 1, "MyTitle": "alpha"}, {"id": 2, "MyTitle": "beta"}],
        pk="id",
    )
    fresh_db["t"].enable_fts(["MyTitle"], create_triggers=True)
    # A differently-cased rename key must resolve against the real column.
    fresh_db["t"].transform(rename={"mytitle": "headline"})
    fresh_db["t"].insert({"id": 3, "headline": "gamma"})
    assert [row["rowid"] for row in fresh_db["t"].search("gamma")] == [3]
    assert [row["rowid"] for row in fresh_db["t"].search("alpha")] == [1]


def test_transform_fts_delete_keeps_index_in_sync(fts5_db):
    fts5_db["articles"].transform(types={"weight": int})
    fts5_db["articles"].insert(
        {"id": 3, "title": "gamma bird", "weight": 30}
    )
    fts5_db["articles"].delete(3)
    assert list(fts5_db["articles"].search("gamma")) == []
    assert [
        row["rowid"] for row in fts5_db["articles"].search("alpha")
    ] == [1]


def test_transform_keep_table_fts_serves_rebuilt_table(fts5_db):
    fts5_db["articles"].transform(
        types={"weight": int}, keep_table="articles_backup"
    )
    fts5_db["articles"].insert(
        {"id": 3, "title": "gamma bird", "weight": 30}
    )
    assert [
        row["rowid"] for row in fts5_db["articles"].search("gamma")
    ] == [3]
    assert fts5_db["articles_backup"].exists()


# ---------------------------------------------------------------------------
# Side 3: cached counts stay in step
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("fts_version", ["FTS5", "FTS4"])
def test_transform_reprimes_counts_cache(fts_db):
    table = fts_db["articles"]
    table.transform(types={"weight": int})
    # The rebuild bypassed the triggers, so the cache must be re-primed.
    assert fts_db.cached_counts(["articles"]) == {"articles": 2}
    assert table.count == 2


@pytest.mark.parametrize("fts_version", ["FTS5", "FTS4"])
def test_transform_counts_follow_inserts(fts_db):
    table = fts_db["articles"]
    table.transform(types={"weight": int})
    table.insert({"id": 3, "title": "gamma bird", "weight": 30})
    assert table.count == 3
    assert fts_db.cached_counts(["articles"]) == {"articles": 3}


def test_transform_counts_follow_deletes(fts5_db):
    table = fts5_db["articles"]
    table.transform(types={"weight": int})
    table.insert({"id": 3, "title": "gamma bird", "weight": 30})
    table.delete(3)
    assert table.count == 2
    assert fts5_db.cached_counts(["articles"]) == {"articles": 2}


def test_transform_keep_table_counts_track_rebuilt_table(fts5_db):
    table = fts5_db["articles"]
    table.transform(
        types={"weight": int}, keep_table="articles_backup"
    )
    assert table.count == 2
    table.insert({"id": 3, "title": "gamma bird", "weight": 30})
    assert table.count == 3
    assert fts5_db.cached_counts(["articles"]) == {"articles": 3}


# ---------------------------------------------------------------------------
# Index, foreign key and transaction regressions
# ---------------------------------------------------------------------------


def test_transform_still_recreates_explicit_indexes(fts5_db):
    fts5_db["articles"].create_index(["weight"])
    index_names_before = {
        index.name for index in fts5_db["articles"].indexes
    }
    fts5_db["articles"].transform(types={"weight": int})
    index_names_after = {
        index.name for index in fts5_db["articles"].indexes
    }
    assert index_names_after == index_names_before


def test_transform_foreign_key_protection_unchanged(fresh_db):
    fresh_db["authors"].insert({"id": 1, "name": "Jane"}, pk="id")
    fresh_db["books"].insert(
        {"id": 1, "title": "Book", "author_id": 1}, pk="id",
        foreign_keys=[("author_id", "authors", "id")],
    )
    fresh_db["authors"].transform(types={"id": int})
    fk = fresh_db["books"].foreign_keys[0]
    assert fk.other_table == "authors"
    assert tuple(fk.columns) == ("author_id",)
