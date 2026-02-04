# Draft: Project3 Optimization Plan

## Requirements (confirmed)
- Create a precise plan to (a) review project issues, (b) propose an optimization roadmap, (c) start optimization by creating project3_v2.ipynb in E:\programs\APEXUSTech_Inter\project3.
- Provide a step-by-step plan with parallel task graph, explicit deliverables, and required files to read/modify.
- Must include success criteria and test/validation steps for notebook changes.
- Do not implement yet; plan only.
- Required tools for later execution: read, glob, apply_patch, write (if needed). No bash.

## Technical Decisions
- project3_v2.ipynb should be a full refactor using project3 package imports (not a minimal wrapper).
- Optimization priorities: speed, reproducibility (fixed seed), modularization, strategy correctness/accuracy.
- Success criteria: end-to-end run without errors; key metrics consistent with existing (no material deviation).
- No environment constraints; fixed random seed required.

## Research Findings
- Project3 optimized framework exists in:
  - E:\programs\APEXUSTech_Inter\project3\optimized_backtest.py
  - E:\programs\APEXUSTech_Inter\project3\project3\optimized_backtest.py
  - E:\programs\APEXUSTech_Inter\project3\project3\optimized_integration.py
  - E:\programs\APEXUSTech_Inter\project3\benchmark_optimization.py
- Existing project3 notebooks:
  - E:\programs\APEXUSTech_Inter\project3\test1.ipynb
  - E:\programs\APEXUSTech_Inter\project3\test2.ipynb
  - E:\programs\APEXUSTech_Inter\project3\test3.ipynb
  - E:\programs\APEXUSTech_Inter\project3\tmp\temp.ipynb
- Baseline notebook to refactor:
  - E:\programs\APEXUSTech_Inter\YiboSun_Project3_15_08.ipynb
- Data CSVs for project3 exist under E:\programs\APEXUSTech_Inter\project3\ (multiple datasets).

## Open Questions
- Awaiting external best-practice research for notebook optimization and roadmap structure (librarian agent).

## Scope Boundaries
- INCLUDE: issue review, optimization roadmap, creation of project3_v2.ipynb plan steps, validation steps.
- EXCLUDE: implementation or code edits at this stage.
