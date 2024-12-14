
# Architecture and Design

At the outset of the design of the FLORIS software, a few fundamental ideas were identified
that should continue to guide future design decisions. These characteristics should never
be violated, and ongoing work should strive to meet these ideas and expand on them as much
as possible.

- Modularity in wake model formulation:
    - New mathematical formulation should be straightforward to incorporate by non-expert
        software developers.
    - Solver and grid data structures for one wake model should not conflict with the data
        structures for other wake models.
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

    class core["floris.core"] {
        +Core
    }

    class floris["floris"] {
        +FlorisModel
    }

    class logging_manager
    class type_dec
    class utilities

    tools <-- logging_manager
    simulation <-- logging_manager
    simulation <-- type_dec
    simulation <-- utilities
    tools <-- simulation
```

## floris

This is the user interface. Most operations at the user level will happen through `floris`.
This package contains a wide variety of functionality including but not limited to:

- Initializing and driving a simulation with `floris_model`
- Wake field visualization through `flow_visualization`
- Yaw and layout optimization in `optimization`
- Wind data handling in `wind_data`

## floris.core

This is the core simulation package. This should primarily be used within `floris.core` and
`floris`, and user scripts generally won't interact directly with this package.

```{mermaid}
classDiagram

    class Core

    class Farm

    class FlowField {
        u: NDArrayFloat
        v: NDArrayFloat
        w: NDArrayFloat
    }

    class Grid {
        <<interface>>
        x: NDArrayFloat
        y: NDArrayFloat
        z: NDArrayFloat
    }
    class TurbineGrid
    class TurbineCubatureGrid
    class FlowFieldPlanarGrid
    class PointsGrid

    class WakeModelManager {
        <<interface>>
    }
    class WakeCombination {
        parameters: dict
        function()
    }
    class WakeDeflection {
        parameters: dict
        function()
    }
    class WakeTurbulence {
        parameters: dict
        function()
    }
    class WakeVelocity {
        parameters: dict
        function()
    }

    class Solver {
        <<interface>>
        parameters: dict
    }

    Core *-- Farm
    Core *-- FlowField
    Core *-- Grid
    Core *-- WakeModelManager
    Core --> Solver
    WakeModelManager *-- WakeCombination
    WakeModelManager *-- WakeDeflection
    WakeModelManager *-- WakeTurbulence
    WakeModelManager *-- WakeVelocity

    Grid <|-- TurbineGrid
    Grid <|-- TurbineCubatureGrid
    Grid <|-- FlowFieldPlanarGrid
    Grid <|-- PointsGrid

    Solver --> Farm
    Solver --> FlowField
    Solver --> Grid
    Solver --> WakeModelManager
```
