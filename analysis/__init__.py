"""Analysis and plotting pipeline for the RL round-scheduling experiments.

Modules:
    config   -- experiment constants (episode length, weights, paths, ...)
    methods  -- registry of compared methods (labels, colors, markers)
    io       -- parser for Alchemist CSV exports
    data     -- tidy tables built from the raw runs
    stats    -- confidence intervals, significance tests, effect sizes
    pareto   -- comparisons against the fixed-period frontier
    style    -- shared IEEE paper style
    figures  -- one module per figure family
    tables   -- LaTeX booktabs tables
    report   -- machine-generated summary.md
"""
