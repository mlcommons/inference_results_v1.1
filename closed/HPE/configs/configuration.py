# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import sys
sys.path.insert(0, os.getcwd())

from code.common.system_list import System
from code.common.constants import Benchmark, Scenario, HarnessType, AccuracyTarget, PowerSetting, CPUArch, WorkloadSetting
from code.common.utils import Tree
from code.common.fields import get_applicable_fields
from configs.error import *

import ast
import inspect
import textwrap

from collections import namedtuple
from enum import Enum, unique
from importlib import import_module


def _parse_sourcelines(lines):
    """
    Returns the Abstract Syntax Tree of the source code represented in `lines`. Since source may be local to a scope and
    have indents, it is possible that the parsing of `lines` may fail due to improper indentation.

    This method will normalize the indentation of the sourcelines so that the first line in `lines` will not have any
    indentation.
    """
    if len(lines) > 0:
        indent_str_len = len(lines[0]) - len(textwrap.dedent(lines[0]))
    normalized = [s[indent_str_len:] for s in lines]
    return ast.parse("".join(normalized))


class ConfigRegistry(object):
    """
    Singleton, global data structure to store and index benchmark-system configurations and provide metadata about them.

    A config cannot be registered unless it passes certain validation checks, and will not be able to be used at runtime
    unless it is registered. This serves as a 'security check' at runtime to ensure best practices when writing configs.

    Internal structure for storing configs. Should not be accessed outside the class. This Tree will have the structure:
        {
            <benchmark>: {
                <scenario>: {
                    <system id>: {
                        <setting>: <config>, # 'setting' is a string that describes the workload setting
                        ...
                    },
                    ...
                },
                ...
            },
            ...
        }
    """

    @staticmethod
    def load_configs(benchmark: Benchmark, scenario: Scenario) -> bool:
        """
        Bulk registers all the configs for a given benchmark-scenario pair by importing its file from configs/. If a
        module for benchmark.scenario does not exist, will return False.

        Args:
            benchmark (Benchmark):
                The benchmark to load configs for
            scenario (Scenario):
                The scenario to load configs for

        Returns:
            bool: Whether or not the module was loaded successfully
        """
        try:
            import_module(f"configs.{str(benchmark.value)}.{str(scenario.value)}")
            return True
        except ModuleNotFoundError:
            return False

    _registry = Tree()

    @classmethod
    def _reset(cls):
        """
        Clears the registry.
        """
        cls._registry = Tree()

    @classmethod
    def register(cls, harness_type, accuracy_target, power_setting):
        """
        Used as a decorator to register a config for the given workload setting.

        Returns:
            A func that will attempt to register the config, and will return the config.

            This func will attempts to register the config if the config passes validation checks. If the config is
            invalid or is already exists in the registry, this returned func will error.
        """
        workload_setting = WorkloadSetting(harness_type, accuracy_target, power_setting)

        def _do_register(config):
            cls.validate(config, workload_setting)  # Will raise error if validation fails

            keyspace = [config.benchmark, config.scenario, config.system, workload_setting]
            if cls._registry.get(keyspace) != None:
                raise KeyError("Config for {} is already registered.".format("/".join(map(str, keyspace))))
            cls._registry.insert(keyspace, config)
            return config
        return _do_register

    @classmethod
    def get(cls, benchmark, scenario, system, harness_type=HarnessType.Custom, accuracy_target=AccuracyTarget.k_99,
            power_setting=PowerSetting.MaxP):
        """
        Returns the config specified, None if it doesn't exist.
        """
        workload_setting = WorkloadSetting(harness_type, accuracy_target, power_setting)
        return cls._registry.get([benchmark, scenario, system, workload_setting])

    @classmethod
    def available_workload_settings(cls, benchmark, scenario):
        """
        Returns the registered WorkloadSettings for a given benchmark-scenario pair. None if the benchmark-scenario
        pair does not exist.
        """
        workloads = cls._registry.get([benchmark, scenario])
        if workloads is None:
            return None
        return list(workloads.keys())

    @classmethod
    def validate(cls, config, workload_setting):
        """
        Validates the config's settings based on certain rules. The config must satisfy the following:
            1. Must have 'BenchmarkConfiguration' in its inheritance parent classes
            2. Can only inherit from configs that use the same:
                - Benchmark
                - Scenario
                - Accelerator
            3. Does not include any config fields that are not applicable to the workload
            4. Contains all required keys necessary to run the workload
            5. Does not define fields multiple times within the class
            6. Does not extend multiple classes at any point within its inheritance chain

        If the config fails any of these criteria, it will raise the corresponding configs.error.ConfigFieldError.
        This method will finish successfully without errors raised if the config is valid.
        """
        # Criteria 1,2,6
        cls._check_inheritance_constraints(config)

        # Criteria 3,4 require arguments.py refactor, as it is too clunky to determine the correct key set at
        cls._check_field_descriptions(config, workload_setting)

        # Criteria 5: Cannot define the same field multiple times within the class
        cls._check_field_reassignment(config)

    @classmethod
    def _check_field_descriptions(cls, config, workload_setting, enforce_full=True):
        """
        Checks the names and types of all the fields in config.

        This method will:
            1. Checks if the config has 'benchmark', 'scenario', and 'system'
            2. Builds a list of mandatory and optional Fields given the 'benchmark', 'scenario', and 'system'
            3. Check if any fields exist in config that are not in these mandatory and optional sets.

        This method will run successfully if and only if:
            - config.fields is a subset of (mandatory UNION optional)
            - If enforce_full is True, mandatory is a subset of config.fields

        Otherwise, this method will throw a ConfigFieldMissingError or ConfigFieldInvalidTypeError.
        """
        # Check for benchmark, scenario, and system:
        identifiers = {
            "action": None,
            "benchmark": None,
            "scenario": None,
            "system": None,
            "workload_setting": workload_setting,
        }

        ConfigFieldDescription = namedtuple("ConfigFieldDescription", ("name", "type"))
        for f in (
            ConfigFieldDescription(name="benchmark", type=Benchmark),
            ConfigFieldDescription(name="scenario", type=Scenario),
            ConfigFieldDescription(name="system", type=System)
        ):
            if not hasattr(config, f.name):
                if enforce_full:
                    raise ConfigFieldMissingError(config.__name__, f.name)
            elif type(getattr(config, f.name)) != f.type:
                raise ConfigFieldInvalidTypeError(config.__name__, f.name, f.type)
            else:
                identifiers[f.name] = getattr(config, f.name)

        # Grab a set of mandatory and optional Fields from fields.py.
        mandatory, optional = get_applicable_fields(**identifiers)
        mandatory = set([f.name for f in mandatory])
        optional = set([f.name for f in optional])
        possible_fields = mandatory.union(optional)
        # Remove 'action' and 'harness_type', as these are metadata about the config, not fields.
        declared_fields = set(config.all_fields()) - {"action", "harness_type"}

        disallowed_fields = declared_fields - possible_fields
        if len(disallowed_fields) > 0:
            raise ConfigFieldInvalidError(config.__name__, list(disallowed_fields)[0])

        missing_fields = mandatory - declared_fields
        if enforce_full and len(missing_fields) > 0:
            raise ConfigFieldMissingError(config.__name__, list(missing_fields)[0])

        if HarnessType.Triton == workload_setting.harness_type:
            if config.use_triton != True:
                raise ConfigFieldError(f"<config '{config.__name__}'> uses Triton, but use_triton is not set to True")

    @classmethod
    def _check_inheritance_constraints(cls, config):
        """
        Checks that the config satisfies the following rules:
            1. Must have 'BenchmarkConfiguration' in its inheritance parent classes
            2. Can only inherit from configs that use the same:
                - Benchmark
                - Scenario
                - Accelerator
            3. Does not extend multiple classes at any point within its inheritance chain
        """
        # Check inheritance rules via MRO (method resolution order)
        parents = inspect.getmro(config)

        # Criteria 1:
        if BenchmarkConfiguration not in parents:
            raise ConfigInvalidTypeError(config.__name__)

        # Criteria 3:
        for curr in parents[:-2]:
            # Search for the field in the current class:
            source, start_lineno = inspect.getsourcelines(curr)
            syntax_tree = _parse_sourcelines(source)
            classdefs = [node for node in ast.walk(syntax_tree) if isinstance(node, ast.ClassDef)]
            assert(len(classdefs) > 0)

            # We only care about the 0th class definition, as any class definition after that would be an internal class
            # defined within the Configuration
            if len(classdefs[0].bases) != 1:
                raise ConfigMultipleExtendsError(curr.__name__)

        # Criteria 2: Check every parent for validity
        for parent in parents[:-2]:
            # Must have the same benchmark if a parent defined a benchmark.
            if hasattr(parent, 'benchmark'):
                if config.benchmark != parent.benchmark:
                    raise ConfigFieldInheritanceError(config.__name__, parent.__name__, "benchmark", config.benchmark,
                                                      parent.benchmark)

            # Must have the same scenario if a parent defined a scenario.
            if hasattr(parent, 'scenario'):
                if config.scenario != parent.scenario:
                    raise ConfigFieldInheritanceError(config.__name__, parent.__name__, "scenario", config.scenario,
                                                      parent.scenario)

            # Must have the same accelerator if a parent defined an accelerator.
            if hasattr(config, 'system'):
                if type(config.system) != System:
                    raise ConfigFieldInvalidTypeError(config.__name__, "system", System.__name__)

                if hasattr(parent, 'system'):
                    if type(parent.system) != System:
                        raise ConfigFieldInvalidTypeError(parent.__name__, "system", System.__name__)

                    # TODO: code.common.system_list.System does not explicitly support non-GPU accelerators.
                    if config.system.gpu != parent.system.gpu:
                        raise ConfigFieldInheritanceError(config.__name__, parent.__name__, "system.gpu",
                                                          config.system.gpu, parent.system.gpu)

    @classmethod
    def _check_field_reassignment(cls, config):
        """
        Checks that the config does not define fields multiple times within the class
        """
        trace = config.get_field_trace()
        for field, assignments in trace.trace.items():
            # Note that since assignments is an ordered sequence, we can just iterate once to check for duplicates
            for i in range(len(assignments) - 1):
                if assignments[i].klass == assignments[i + 1].klass:
                    raise ConfigFieldReassignmentError(assignments[i].klass.__name__,
                                                       field,
                                                       (assignments[i].lineno, assignments[i + 1].lineno))


class BenchmarkConfiguration(object):
    """
    Describes a configuration for a BenchmarkScenario pair for a given system. If the config is meant to be used as a
    full config, the derived class must be registered with the @ConfigRegistry.register decorator. For example, there
    might be an INT8Configuration with generic, default settings for an INT8 benchmark that is not registered, which is
    extended by ResNet50_INT8_Configuration that will be registered.

    Fields for the configuration are all non-callable class variables that are not prefixed with '_'. For instance, if
    you wish for a variable to be hidden from code that uses the config, prefix it with '_'.

    A configuration is defined like follows:

        class MyConfiguration(<must have BenchmarkConfiguration somewhere in its inheritance chain>):
            system = System(...)
            scenario = Scenario.<...>
            _N = 12 # This field will not be visible when a list of fields is requested
            batch_size = 64 * _N # This field uses __N to calculate it, and will be visible
            depth = _N + 1
            some_field = "value"
            ...

    For best practices, Configurations should **NOT** have multiple inherits at any level of their inheritance chain.
    This makes it easier to detect where fields are introduced along the inheritance chain.

    Example:
        class Foo(BenchmarkConfiguration):
            ...

        class Bar(Foo): # This is fine
            ...

        class Baz(Bar, SomeOtherClass): # This is not best practices
            ...

        class Boo(Bar): # This is still fine
            ...

        class Faz(Baz): # Even though Faz only inherits Baz, Baz has multiple inherits which makes this not advised.
            ...
    """

    @classmethod
    def all_fields(cls):
        """
        Returns all visible fields in this config. Visible fields are all non-callable class variables that are not
        prepended with '_'.
        """
        return tuple([
            attr
            for attr in dir(cls)
            if not callable(getattr(cls, attr)) and not attr.startswith("_")
        ])

    @classmethod
    def as_dict(cls):
        """
        Returns all fields of the class as a shallow dict of {key:value}.
        """
        return {
            attr: getattr(cls, attr)
            for attr in cls.all_fields()
        }

    @classmethod
    def get_field_trace(cls):
        """
        Returns a FieldTrace of this class. See FieldTrace documentation for more details. This is useful to track down
        where fields are overridden or inherited from.
        """
        return FieldTrace(cls)


class FieldTrace(object):
    """
    Represents the trace of all the fields of a BenchmarkConfiguration as they are declared in Method Resolution Order.

    {
        <field>: [ list of namedtuple(klass=<class>, lineno=int, value=<ast.Value object>) ]
        ...
    }
    The list is ordered in method resolution order, where the first element is the true resolution of the field at
    runtime, and subsequent elements are resolutions of the field further down the resolution chain.

    i.e. Given
        class A(object):
            foo = 1

        class B(A):
            foo = 2

        class C(B):
            foo = 3

    The FieldTrace of C will be:
        {
            "foo": [(klass=C, lineno=..., value=ast.Num(3)),
                    (klass=B, lineno=..., value=ast.Num(2)),
                    (klass=A, lineno=..., value=ast.Num(1))]
        }
    """

    def __init__(self, config_cls):
        """
        Initializes a FieldTrace for the given class.
        """
        self.cls_name = config_cls.__name__

        Assignment = namedtuple("Assignment", ["klass", "lineno", "value"])
        parents = inspect.getmro(config_cls)
        trace = dict()

        def _add_trace(k, v):
            if k not in trace:
                trace[k] = []
            trace[k].append(v)
        fields = config_cls.all_fields()

        # The last 2 fields in MRO will be <BenchmarkConfiguration> and <object>, which we don't want to check.
        for curr in parents[:-2]:
            # Search for the field in the current class:
            source, start_lineno = inspect.getsourcelines(curr)
            syntax_tree = _parse_sourcelines(source)
            assignments = [node for node in ast.walk(syntax_tree) if isinstance(node, ast.Assign)]

            # Later assignments will take precendence in resolution, so reverse the order of assignments
            for assignment in assignments[::-1]:
                for target in assignment.targets:
                    if target.id in fields:
                        _add_trace(
                            target.id,
                            Assignment(
                                klass=curr,
                                lineno=(start_lineno + assignment.lineno - 1),
                                value=assignment.value
                            ))
        self.trace = trace

    def __str__(self):
        return self.dump()

    def dump(self, sort_fields=True, indent=4):
        """
        Returns a string representing this FieldTrace in a human-readable format.
        """
        keys = self.trace.keys()
        if sort_fields:
            keys = list(sorted(keys))

        indent_str = " " * indent
        lines = [f"FieldTrace({self.cls_name})" + "{"]
        for key in keys:
            lines.append(f"{indent_str}'{key}': [")
            for assignment in self.trace[key]:
                lines.append(f"{indent_str * 2}Assignment(")
                lines.append(f"{indent_str * 3}klass={assignment.klass}")
                lines.append(f"{indent_str * 3}lineno={assignment.lineno}")
                lines.append(f"{indent_str * 3}value={ast.dump(assignment.value)}")
                lines.append(f"{indent_str * 2}),")
            lines.append(f"{indent_str}]")
        lines.append("}")
        return "\n".join(lines)
