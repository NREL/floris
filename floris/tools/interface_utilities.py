# Copyright 2020 NREL

# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.

import inspect


def show_params(
    fi,
    params=None,
    verbose=False,
    wake_velocity_model=True,
    wake_deflection_model=True,
    turbulence_model=True,
):

    if wake_velocity_model:
        obj = "fi.floris.farm.wake.velocity_model"
        props = get_props(obj, fi)

        if verbose:
            print("=".join(["="] * 39))
        else:
            print("=".join(["="] * 19))
        print(
            "Wake Velocity Model Parameters:",
            fi.floris.farm.wake.velocity_model.model_string,
            "model",
        )

        if params is not None:
            props_subset = get_props_subset(params, props)
            if not verbose:
                print_props(obj, fi, props_subset)
            else:
                print_prop_docs(obj, fi, props_subset)

        else:
            if not verbose:
                print_props(obj, fi, props)
            else:
                print_prop_docs(obj, fi, props)

    if wake_deflection_model:
        obj = "fi.floris.farm.wake.deflection_model"
        props = get_props(obj, fi)

        if verbose:
            print("=".join(["="] * 39))
        else:
            print("=".join(["="] * 19))
        print(
            "Wake Deflection Model Parameters:",
            fi.floris.farm.wake.deflection_model.model_string,
            "model",
        )

        if params is not None:
            props_subset = get_props_subset(params, props)
            if props_subset:  # true if the subset is not empty
                if not verbose:
                    print_props(obj, fi, props_subset)
                else:
                    print_prop_docs(obj, fi, props_subset)

        else:
            if not verbose:
                print_props(obj, fi, props)
            else:
                print_prop_docs(obj, fi, props)

    if turbulence_model:
        obj = "fi.floris.farm.wake.turbulence_model"
        props = get_props(obj, fi)

        if verbose:
            print("=".join(["="] * 39))
        else:
            print("=".join(["="] * 19))
        print(
            "Wake Turbulence Model Parameters:",
            fi.floris.farm.wake.turbulence_model.model_string,
            "model",
        )

        if params is not None:
            props_subset = get_props_subset(params, props)
            if props_subset:  # true if the subset is not empty
                if not verbose:
                    print_props(obj, fi, props_subset)
                else:
                    print_prop_docs(obj, fi, props_subset)

        else:
            if not verbose:
                print_props(obj, fi, props)
            else:
                print_prop_docs(obj, fi, props)


def get_params(
    fi,
    params=None,
    wake_velocity_model=True,
    wake_deflection_model=True,
    turbulence_model=True,
):
    model_params = {}

    if wake_velocity_model:
        wake_vel_vals = {}
        obj = "fi.floris.farm.wake.velocity_model"
        props = get_props(obj, fi)
        if params is not None:
            props_subset = get_props_subset(params, props)
            wake_vel_vals = get_prop_values(obj, fi, props_subset)
        else:
            wake_vel_vals = get_prop_values(obj, fi, props)
        model_params["Wake Velocity Parameters"] = wake_vel_vals
        del model_params["Wake Velocity Parameters"]["logger"]

    if wake_deflection_model:
        wake_defl_vals = {}
        obj = "fi.floris.farm.wake.deflection_model"
        props = get_props(obj, fi)
        if params is not None:
            props_subset = get_props_subset(params, props)
            wake_defl_vals = get_prop_values(obj, fi, props_subset)
        else:
            wake_defl_vals = get_prop_values(obj, fi, props)
        model_params["Wake Deflection Parameters"] = wake_defl_vals
        del model_params["Wake Deflection Parameters"]["logger"]

    if turbulence_model:
        wake_turb_vals = {}
        obj = "fi.floris.farm.wake.turbulence_model"
        props = get_props(obj, fi)
        if params is not None:
            props_subset = get_props_subset(params, props)
            wake_turb_vals = get_prop_values(obj, fi, props_subset)
        else:
            wake_turb_vals = get_prop_values(obj, fi, props)
        model_params["Wake Turbulence Parameters"] = wake_turb_vals
        del model_params["Wake Turbulence Parameters"]["logger"]

    return model_params


def set_params(fi, params, verbose=True):
    for param_dict in params:
        if param_dict == "Wake Velocity Parameters":
            obj = "fi.floris.farm.wake.velocity_model"
            props = get_props(obj, fi)
            for prop in params[param_dict]:
                if prop in [val[0] for val in props]:
                    exec(obj + "." + prop + " = " + str(params[param_dict][prop]))
                    if verbose:
                        print(
                            "Wake velocity parameter "
                            + prop
                            + " set to "
                            + str(params[param_dict][prop])
                        )
                else:
                    raise Exception(
                        (
                            "Wake deflection parameter '{}' "
                            + "not part of current model. Value '{}' was not "
                            + "used."
                        ).format(prop, params[param_dict][prop])
                    )

        if param_dict == "Wake Deflection Parameters":
            obj = "fi.floris.farm.wake.deflection_model"
            props = get_props(obj, fi)
            for prop in params[param_dict]:
                if prop in [val[0] for val in props]:
                    exec(obj + "." + prop + " = " + str(params[param_dict][prop]))
                    if verbose:
                        print(
                            "Wake deflection parameter "
                            + prop
                            + " set to "
                            + str(params[param_dict][prop])
                        )
                else:
                    raise Exception(
                        (
                            "Wake deflection parameter '{}' "
                            + "not part of current model. Value '{}' was not "
                            + "used."
                        ).format(prop, params[param_dict][prop])
                    )

        if param_dict == "Wake Turbulence Parameters":
            obj = "fi.floris.farm.wake.turbulence_model"
            props = get_props(obj, fi)
            for prop in params[param_dict]:
                if prop in [val[0] for val in props]:
                    exec(obj + "." + prop + " = " + str(params[param_dict][prop]))
                    if verbose:
                        print(
                            "Wake turbulence parameter "
                            + prop
                            + " set to "
                            + str(params[param_dict][prop])
                        )
                else:
                    raise Exception(
                        (
                            "Wake turbulence parameter '{}' "
                            + "not part of current model. Value '{}' was not "
                            + "used."
                        ).format(prop, params[param_dict][prop])
                    )


def get_props_subset(params, props):
    prop_names = [prop[0] for prop in props]
    try:
        props_subset_inds = [prop_names.index(param) for param in params]
    except:
        props_subset_inds = []
        print("Parameter(s)", ", ".join(params), "does(do) not exist.")
    props_subset = [props[i] for i in props_subset_inds]
    return props_subset


def get_props(obj, fi):
    return inspect.getmembers(
        eval(obj + ".__class__"), lambda obj: isinstance(obj, property)
    )


def get_prop_values(obj, fi, props):
    prop_val_dict = {}
    for val in props:
        prop_val_dict[val[0]] = eval(obj + "." + val[0])
    return prop_val_dict


def print_props(obj, fi, props):
    print("-".join(["-"] * 19))
    for val in props:
        print(val[0] + " = " + str(eval(obj + "." + val[0])))
    print("-".join(["-"] * 19))


def print_prop_docs(obj, fi, props):
    for val in props:
        print(
            "-".join(["-"] * 39) + "\n",
            val[0] + " = " + str(eval(obj + "." + val[0])),
            "\n",
            eval(obj + ".__class__." + val[0] + ".__doc__"),
        )
    print("-".join(["-"] * 39))
