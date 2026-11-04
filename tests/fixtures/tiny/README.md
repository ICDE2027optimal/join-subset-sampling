# `stats-undirected-v1` loader fixture

This four-relation path has five full tuple-ID witnesses. The
relation order is `ph,v,c,pl`. Edge IDs are undirected and each edge's key IDs
are interned in a namespace local to that edge.  `weights/uniform/scale-1`
uses the split key/weight layout consumed by the formal C++ loader.
The `boundary` weight set repeats the validated three-factor counterexample
weights `0.9169340963554261`, `0.9992149193767528`, and
`0.5457238974081999`. Every witness has canonical preorder product exactly
`0.5` in the existing double implementation.  The fourth relation has unit
weight.

Expected witnesses, in relation order, are:

```text
10,20,30,40
10,21,31,40
10,21,32,41
11,22,31,40
11,22,32,41
```
