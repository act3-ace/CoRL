import pytest


class ToyClass1:
    ...


class ToyClass2:
    ...


class ToyClass3:
    ...


class ToyClass4:
    ...


def test_plugin_library():
    from corl.libraries.plugin_library import PluginLibrary

    PluginLibrary.AddClassToGroup(ToyClass1, "class_1", {"condition1": "test"})
    PluginLibrary.AddClassToGroup(ToyClass2, "class_1", {"condition1": "test", "condition2": "test2"})
    PluginLibrary.AddClassToGroup(ToyClass3, "class_1", {"condition1": "test", "condition2": ["test4", "test3"]})
    PluginLibrary.AddClassToGroup(ToyClass4, "class_4", {})
    PluginLibrary.AddClassToGroup(ToyClass2, "class_2", {"condition1": "test"})

    # test doubling up on conditions not causing an error
    PluginLibrary.AddClassToGroup(ToyClass2, "class_2", {"condition1": "test"})

    # test wrong type for conditions
    with pytest.raises(RuntimeError):
        PluginLibrary.AddClassToGroup(ToyClass2, "class_2", [])

    # test wrong type for callable
    with pytest.raises(RuntimeError):
        PluginLibrary.AddClassToGroup({}, "class_2", {"condition1": "test"})

    # test access with no conditions
    assert PluginLibrary.FindMatch("class_4", {}) == ToyClass4
    # test access with 1 condition
    assert PluginLibrary.FindMatch("class_1", {"condition1": "test"}) == ToyClass1
    # test access with 1 condition with different group name than above
    assert PluginLibrary.FindMatch("class_2", {"condition1": "test"}) == ToyClass2
    # test access with 2 correct matches
    assert PluginLibrary.FindMatch("class_1", {"condition1": "test", "condition2": "test2"}) == ToyClass2
    # test access with 1 condition match and then a condition miss, so that it falls through to a the single match case
    assert PluginLibrary.FindMatch("class_1", {"condition1": "test", "condition2": "test123434"}) == ToyClass1
    # test access with 1 condition match and then a condition miss; however, no single match case exists
    with pytest.raises(RuntimeError):
        PluginLibrary.FindMatch("class_1", {"condition1": "test123123", "condition2": "test2"})
    # test access with 2 correct matches using product
    assert PluginLibrary.FindMatch("class_1", {"condition1": "test", "condition2": "test3"}) == ToyClass3
    assert PluginLibrary.FindMatch("class_1", {"condition1": "test", "condition2": "test4"}) == ToyClass3

    # test accessing with no condition matches
    with pytest.raises(RuntimeError):
        PluginLibrary.FindMatch("class_1", {"condition1": "test123123", "condition2": "test412312"})

    # test accessing a non registered class
    with pytest.raises(RuntimeError):
        PluginLibrary.FindMatch("class_100", {})
