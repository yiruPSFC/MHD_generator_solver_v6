# v6_casadi_v2 Flowchart

This document summarizes the execution flow of the `v6_casadi_v2` workflow.

## 1. Module-Level Flow

```mermaid
flowchart TD
    A["validation/smoke_test_casadi_area_opt_v2.py"] --> B["optimize_area_profile(...)"]
    C["run_casadi_continuation_v2.py main()"] --> D["run_continuation(...)"]
    E["runners/run_relaxed_continuation_v2.py main()"] --> F["build relaxed stage schedule"]
    F --> D
    D --> B
    B --> G["OptimizedAreaProfile"]
    D --> H["stage artifacts (.npz/.png/.json)"]
    D --> I["continuation_summary.json"]
```

## 2. Continuation Driver Flow

Source: `run_casadi_continuation_v2.py`

```mermaid
flowchart TD
    A["run_continuation(...)"] --> B["load stage schedule"]
    B --> C["initialize warm_profile=None"]
    C --> D{"for each stage"}

    D --> E{"adaptive bridging enabled and\nwarm start available?"}
    E -->|yes| F["build bridge stages between\nlast adopted stage and target stage"]
    F --> G["run each bridge stage via _run_stage()"]
    G --> H["_save_stage_record(...)"]
    H --> I{"warm start gate passed?"}
    I -->|yes| J["promote bridge result to warm start"]
    I -->|no| K{"can increase bridge count?"}
    K -->|yes| F
    K -->|no| L["continue with original target stage"]
    E -->|no| L
    J --> L

    L --> M["run target stage via _run_stage()"]
    M --> N["_save_stage_record(...)"]
    N --> O{"first stage?"}
    O -->|yes| P["store baseline_result"]
    O -->|no| Q["keep previous baseline"]
    P --> R{"warm start gate passed?"}
    Q --> R
    R -->|yes| S["make warm profile for next stage"]
    R -->|no| T["keep current warm profile"]
    S --> U{"result acceptable?"}
    T --> U
    U -->|no and stop_on_unacceptable| V["stop early"]
    U -->|otherwise| W["next stage"]
    W --> D
    V --> X["assemble payload"]
    D -->|done| X
    X --> Y{"out_dir provided?"}
    Y -->|yes| Z["save final plots, npz, summary json"]
    Y -->|no| AA["return payload only"]
```

## 3. Single-Stage NLP Flow

Source: `optimize_area_profile_casadi_v2.py`

```mermaid
flowchart TD
    A["optimize_area_profile(...)"] --> B["validate scalar inputs and bounds"]
    B --> C["build x grid and stage ODE/closure function"]
    C --> D{"warm profile provided?"}
    D -->|yes| E["resample warm profile to current grid"]
    D -->|no| F["build constant warm start"]
    E --> G["project warm start into bounds"]
    F --> G

    G --> H["create CasADi Opti variables:\nX, U, inlet vars, optional slack S"]
    H --> I["apply inlet bounds and normalized state/control bounds"]
    I --> J["build inlet algebraic relations:\nn_e, beta, v_in, dot_N, T_p_in, Mach, G"]
    J --> K["enforce inlet constraints:\nT_p_in >= tp_min, G_in == 0, optional Mach bounds"]

    K --> L["build objective:\nslack penalty + smoothness + curvature\n+ warm-start tracking - design score"]
    L --> M["for each interval, evaluate stage()"]
    M --> N{"transcription"}
    N -->|trapezoid| O["add trapezoid dynamics constraints"]
    N -->|hermite-simpson| P["build midpoint state and add HS constraints"]
    O --> Q["apply node/midpoint path constraints"]
    P --> Q
    Q --> R["apply final-node path constraints"]
    R --> S["set initial guess from warm start"]
    S --> T["solve with IPOPT via solve_limited()"]
    T --> U{"solve raises RuntimeError?"}
    U -->|yes| V["fallback to opti.debug.value"]
    U -->|no| W["use opti.value"]
    V --> X["recover solution arrays and inlet design"]
    W --> X
    X --> Y["evaluate full numeric profile"]
    Y --> Z["compute feasibility and regularity diagnostics"]
    Z --> AA["package OptimizedAreaProfile and return"]
```

## 4. Relaxed Runner Flow

Source: `runners/run_relaxed_continuation_v2.py`

```mermaid
flowchart TD
    A["main()"] --> B{"schedule_json provided?"}
    B -->|yes| C["load custom stage list from json"]
    B -->|no| D["build built-in relaxed schedule:\nconservative / balanced / aggressive"]
    C --> E["call run_continuation(...)"]
    D --> E
    E --> F["append schedule metadata to payload"]
    F --> G{"out_json provided?"}
    G -->|yes| H["write payload to out_json"]
    G -->|no| I["write relaxed_schedule.json into out_dir"]
    H --> J["print payload"]
    I --> J
```
