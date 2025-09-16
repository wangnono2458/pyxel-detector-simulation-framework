#  Copyright (c) European Space Agency, 2020.
#
#  This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#  is part of this Pyxel package. No part of the package, including
#  this file, may be copied, modified, propagated, or distributed except according to
#  the terms contained in the file ‘LICENCE.txt’.


from copy import copy, deepcopy

import pytest

from pyxel.pipelines import ModelFunction, ModelGroup


@pytest.fixture
def model_function1() -> ModelFunction:
    return ModelFunction(
        func="test_model_function.my_func1",
        name="first_function",
        arguments={"param1": 2, "param2": "foo"},
        enabled=True,
    )


@pytest.fixture
def model_function2() -> ModelFunction:
    return ModelFunction(
        func="test_model_function.my_func2",
        name="second_function",
        arguments=None,
        enabled=True,
    )


@pytest.fixture
def model_group_foo(
    model_function1: ModelFunction, model_function2: ModelFunction
) -> ModelGroup:
    return ModelGroup([model_function1, model_function2], name="foo")


def test_model_group_repr(model_group_foo: ModelGroup):
    """Test 'ModelGroup.__repr__'."""
    model_group = model_group_foo
    assert isinstance(model_group_foo, ModelGroup)

    # Test .__repr__
    assert repr(model_group).startswith("ModelGroup<")


def test_model_group_iter(
    model_group_foo: ModelGroup,
    model_function1: ModelFunction,
    model_function2: ModelFunction,
):
    """Test 'ModelGroup.__iter__'."""
    model_group = model_group_foo
    assert isinstance(model_group_foo, ModelGroup)

    lst_models = list(model_group)
    exp_models = [model_function1, model_function2]
    assert lst_models == exp_models


def test_model_group_dir(model_group_foo: ModelGroup):
    """Test 'ModelGroup.__dir__'."""
    model_group = model_group_foo
    assert isinstance(model_group, ModelGroup)

    assert "first_function" in dir(model_group)
    assert "second_function" in dir(model_group)


def test_model_group_set_get_state(model_group_foo: ModelGroup):
    """Test 'ModelGroup.__getstate__' and '.__setstate__'."""
    model_group = model_group_foo
    assert isinstance(model_group, ModelGroup)

    copied_model_group = copy(model_group)
    assert model_group is not copied_model_group
    assert copied_model_group.name == "foo"


def test_model_group_deepcopy(model_group_foo: ModelGroup):
    """Test 'ModelGroup.__deepcopy__'."""
    model_group = model_group_foo
    assert isinstance(model_group, ModelGroup)

    deepcopied_model_group = deepcopy(model_group)
    assert model_group is not deepcopied_model_group
    assert deepcopied_model_group.name == "foo"


def test_model_group_getattr(model_group_foo: ModelGroup):
    """Test 'ModelGroup.__getattr__'."""
    model_group = model_group_foo
    assert isinstance(model_group, ModelGroup)

    assert isinstance(model_group.first_function, ModelFunction)
    assert isinstance(model_group.second_function, ModelFunction)


def test_model_group_getattr_bad_input(model_group_foo: ModelGroup):
    """Test 'ModelGroup.__getattr__'."""
    model_group = model_group_foo
    assert isinstance(model_group, ModelGroup)

    with pytest.raises(AttributeError, match=r"Cannot find model"):
        _ = model_group.unknown
