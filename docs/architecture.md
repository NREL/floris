
# Architecture and Design

At the outset of the design of the FLORIS software, a few fundamental ideas were identified
that should continue to guide future design decisions. These characteristics should never
be violated, and ongoing work should strive to meet these ideas and expand on them as much
as possible.

- Modularity in wake model formulation:
    - New mathematical formulation should be straightforward to incorporate.
    - Requisite solver and grid data structures should not conflict with other existing
        wake models.
- Any new feature or work should not affect an existing feature:
    - Low level code should be reused as much as possible, but high level code should rarely
        be repurposed.
    - It is expected that a new feature will include a new user entry point
        at the highest level.
    - Avoid flags and if-statements that allow using one high-level routine for multiple unrelated
        tasks.
    - When in doubt, create a new pipeline from the user-level API to the low level implementation
        and refactor to consolidate, if necessary, afterwards.
- Management of abstraction:
    - Low level code is opaque but well tested and exercised; it should be very computationally
        efficient with low algorithmic complexity.
    - High level code should be expressive and clear even if it results in verbose or less
        efficient code.

The FLORIS software consists of two primary high-level packages and a few other low level
packages. The internal structure and hierarchy is described below.

```{mermaid}
classDiagram

    class tools {
        +FlorisInterface
    }

    class simulation {
        +Floris
    }

    class logging_manager
    class type_dec
    class utilities

    tools <-- logging_manager
    simulation <-- logging_manager
    tools <-- type_dec
    simulation <-- type_dec
    tools <-- utilities
    simulation <-- utilities
    tools <-- simulation
```

## floris.tools

This is the user interface. Most operations at the user level will happen through `floris.tools`.
This package contains a wide variety of functionality including but not limited to:

- Initializing and driving a simulation with `tools.floris_interface`
- Wake field visualization through `tools.visualization`
- Yaw and layout optimization in `tools.optimization`
- Parallelizing work load with `tools.parallel_computing_interface`

## floris.simulation

This is the core simulation package. This should primarily be used within `floris.simulation` and
`floris.tools`, and user scripts generally won't interact directly with this package.

```{mermaid}
classDiagram

    class Floris

    class Farm

    class FlowField {
        array u
        array v
        array w
    }

    class Grid {
        <<interface>>
    }
    class TurbineGrid
    class FlowFieldPlanarGrid

    class WakeModelManager {
        <<interface>>
    }
    class WakeCombination {
        dict parameters
        function()
    }
    class WakeDeflection {
        dict parameters
        function()
    }
    class WakeTurbulence {
        dict parameters
        function()
    }
    class WakeVelocity {
        dict parameters
        function()
    }

    class Solver {
        <<interface>>
        dict parameters
    }

    Floris o-- Farm
    Floris o-- FlowField
    Floris o-- Grid
    Floris o-- WakeModelManager
    Floris *-- Solver
    WakeModelManager o-- WakeCombination
    WakeModelManager o-- WakeDeflection
    WakeModelManager o-- WakeTurbulence
    WakeModelManager o-- WakeVelocity

    Grid <|-- TurbineGrid
    Grid <|-- FlowFieldPlanarGrid

    Solver --> Farm
    Solver --> FlowField
    Solver --> Grid
    Solver --> WakeModelManager
```
